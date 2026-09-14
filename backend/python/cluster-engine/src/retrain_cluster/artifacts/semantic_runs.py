"""分片化的语义运行产物仓库。

同步接口把明细整份塞进 HTTP 响应；作业路径必须在磁盘上分页读，因为一次
聚类可能有几十万条明细，全量载入内存正是要避免的事。本模块因此把产物拆成三部分：

* ``result.json``：摘要 / 簇 / 聚合 / 抽样坐标——一个请求就够，且很小；
* ``detail/shard-XXXXX.json``：明细。**按簇连续分片**，因此"只看某个簇"
  是一段连续区间，分页只读命中的那几片，内存峰值不随 offset 增长；
* ``manifest.json``：提交标记 + 分片哈希 + 簇区间索引。**最后写**：
  看到它才说明内容完整，中途被杀不会留下可服务的半成品。

任何一片被改动，读取时都会被哈希校验抓住并报 ``ARTIFACT_INVALID``；
按文本筛选若达到扫描上限，必须如实回报 ``truncated``，不能假装搜过了全量。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .runs import atomic_json
from ..errors import ClusterError

__all__ = ["MAX_PAGE_SIZE", "SemanticRunStore"]

#: run_id 参与路径拼接，必须白名单校验。
_RUN_ID = re.compile(r"run_[0-9a-f]{32}")

#: 单页上限是硬约束：调用方不能靠调大 limit 把分页变成全量导出。
MAX_PAGE_SIZE = 500

#: 默认分片大小；仅在调用方未指定时生效。
DEFAULT_SHARD_SIZE = 500


class SemanticRunStore:
    """``<root>/<run_id>/`` 下的分片产物仓库。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    # ------------------------------------------------------------------ 路径与校验

    def path(self, run_id):
        """返回 run 目录；非法 run_id 直接报"未找到"，不去碰文件系统。"""

        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ClusterError("RUN_NOT_FOUND", "Run not found", 404)
        return self.directory / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self.path(run_id) / "manifest.json"

    def exists(self, run_id) -> bool:
        """manifest 是提交标记：它存在才说明这份产物可服务。"""

        try:
            path = self.path(run_id)
        except ClusterError:
            return False
        return (path / "manifest.json").is_file()

    def _read_manifest(self, run_id):
        path = self.path(run_id)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ClusterError("RUN_NOT_FOUND", "Run not found", 404)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500) from None
        if not isinstance(manifest, dict) or manifest.get("artifact_kind") != "sharded":
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        return path, manifest

    # ------------------------------------------------------------------ 写

    def save(
        self,
        *,
        summary,
        clusters,
        aggregate,
        items,
        visualization,
        manifest,
        shard_size: int = DEFAULT_SHARD_SIZE,
    ) -> dict:
        """落盘一次运行，返回分页所需的元信息。

        写入顺序刻意是：撤下旧 manifest → result.json → 分片 → manifest.json。
        这样任何一步崩溃，读者都看不到"半份但可服务"的产物。
        """

        summary = dict(summary or {})
        manifest = dict(manifest or {})
        run_id = summary.get("run_id") or manifest.get("run_id")
        path = self.path(run_id)
        size = max(1, int(shard_size))

        # 撤销旧的提交标记，避免"新内容只写了一半、旧 manifest 还在"的窗口
        manifest_path = path / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        path.mkdir(parents=True, exist_ok=True)

        # 按簇连续排序：同一簇的明细在磁盘上是一整段，簇索引指向其起点与长度
        groups: dict[int, list] = {}
        for item in items or []:
            cluster_id = int(item.get("cluster_id", -1))
            groups.setdefault(cluster_id, []).append(item)
        ordered: list = []
        cluster_index: list[dict] = []
        start = 0
        for cluster_id in sorted(groups):
            members = groups[cluster_id]
            cluster_index.append({"cluster_id": cluster_id, "start": start, "size": len(members)})
            ordered.extend(members)
            start += len(members)

        result_payload = {
            "summary": summary,
            "clusters": list(clusters or []),
            "aggregate": dict(aggregate or {}),
            "visualization": list(visualization or []),
        }
        result_path = path / "result.json"
        atomic_json(result_path, result_payload)

        detail = path / "detail"
        detail.mkdir(parents=True, exist_ok=True)
        for stale in detail.glob("shard-*.json"):
            stale.unlink()
        shards: list[dict] = []
        for index in range(0, len(ordered), size):
            chunk = ordered[index : index + size]
            shard_path = detail / f"shard-{index // size:05d}.json"
            atomic_json(shard_path, chunk)
            shards.append(
                {
                    "file": f"detail/{shard_path.name}",
                    "sha256": _bytes_hash(shard_path.read_bytes()),
                    "count": len(chunk),
                }
            )

        committed = {
            **manifest,
            "schema_version": manifest.get("schema_version", 1),
            "run_id": run_id,
            "artifact_kind": "sharded",
            "status": "completed",
            "item_count": len(ordered),
            "shard_count": len(shards),
            "shard_size": size,
            "clusters": cluster_index,
            "shards": shards,
            "result_sha256": _bytes_hash(result_path.read_bytes()),
        }
        # manifest 最后写
        atomic_json(manifest_path, committed)
        return {"run_id": run_id, "item_count": len(ordered), "shard_count": len(shards), "shard_size": size}

    # ------------------------------------------------------------------ 读

    def load(self, run_id) -> dict:
        """读回摘要/簇/聚合/抽样坐标，并附带 manifest。"""

        path, manifest = self._read_manifest(run_id)
        result_path = path / "result.json"
        if not result_path.is_file():
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        try:
            raw = result_path.read_bytes()
        except OSError:
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500) from None
        if _bytes_hash(raw) != manifest.get("result_sha256"):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500) from None
        payload["manifest"] = manifest
        return payload

    def index(self, run_id) -> dict:
        """簇区间与分片索引，全部来自 manifest，不读明细。"""

        _, manifest = self._read_manifest(run_id)
        return {
            "run_id": manifest.get("run_id"),
            "shard_size": int(manifest.get("shard_size") or 1),
            "item_count": int(manifest.get("item_count") or 0),
            "shard_count": int(manifest.get("shard_count") or 0),
            "clusters": list(manifest.get("clusters") or []),
            "shards": list(manifest.get("shards") or []),
        }

    def filter_options(self, run_id) -> dict:
        """筛选取值清单只读索引，因此与样本量无关。"""

        _, manifest = self._read_manifest(run_id)
        return {
            "runId": manifest.get("run_id"),
            "itemCount": int(manifest.get("item_count") or 0),
            "clusterIds": [
                {"clusterId": int(entry["cluster_id"]), "size": int(entry["size"])}
                for entry in manifest.get("clusters") or []
            ],
        }

    # ------------------------------------------------------------------ 分页

    def page(
        self,
        run_id,
        *,
        offset: int = 0,
        limit=None,
        cluster_id=None,
        assignment_status=None,
        keyword=None,
        scan_limit=None,
    ) -> dict:
        """分页读取明细。

        * 无筛选时只是对连续区间切片，只读命中的分片；
        * 按簇筛选时命中一段连续区间；
        * 按状态/关键词筛选时 total 需要数完整个区间，但只读当前页的数据，
          且关键词扫描若被 ``scan_limit`` 截断会如实回报 ``truncated``。
        """

        _, manifest = self._read_manifest(run_id)
        offset = max(0, int(offset))
        requested = MAX_PAGE_SIZE if limit is None else int(limit)
        page_limit = max(1, min(requested, MAX_PAGE_SIZE))
        item_count = int(manifest.get("item_count") or 0)

        if cluster_id is not None:
            entry = next(
                (item for item in manifest.get("clusters") or [] if int(item["cluster_id"]) == int(cluster_id)),
                None,
            )
            if entry is None:
                return self._page(run_id, [], 0, page_limit, offset, truncated=False)
            start = int(entry["start"])
            end = start + int(entry["size"])
        else:
            start, end = 0, item_count

        predicate = _build_predicate(assignment_status, keyword)
        truncated = False
        if predicate is None:
            total = max(0, end - start)
            positions = list(range(start + offset, min(end, start + offset + page_limit)))
        else:
            matches: list[int] = []
            scanned = 0
            cap = None if scan_limit is None else max(0, int(scan_limit))
            for position, item in self._iter_range(run_id, manifest, start, end):
                if cap is not None and scanned >= cap:
                    truncated = True
                    break
                scanned += 1
                if predicate(item):
                    matches.append(position)
            total = len(matches)
            positions = matches[offset : offset + page_limit]

        items = self._read_items(run_id, manifest, positions)
        page_items = [items[position] for position in positions]
        return self._page(run_id, page_items, total, page_limit, offset, truncated=truncated)

    @staticmethod
    def _page(run_id, items, total, limit, offset, *, truncated) -> dict:
        returned = len(items)
        return {
            "runId": run_id,
            "total": int(total),
            "returned": returned,
            "limit": int(limit),
            "offset": int(offset),
            "hasMore": (offset + returned) < int(total),
            "items": items,
            "truncated": bool(truncated),
        }

    # ------------------------------------------------------------------ 低层读取

    def _read_shard(self, run_id: str, manifest: dict, shard_index: int) -> list:
        shards = manifest.get("shards") or []
        if shard_index < 0 or shard_index >= len(shards):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        record = shards[shard_index]
        shard_path = self.path(run_id) / record["file"]
        try:
            raw = shard_path.read_bytes()
        except (OSError, KeyError, TypeError):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500) from None
        if _bytes_hash(raw) != record.get("sha256"):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500) from None
        if not isinstance(payload, list):
            raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
        return payload

    def _iter_range(self, run_id: str, manifest: dict, start: int, end: int):
        """按顺序流式遍历 ``[start, end)``，同一时刻只持有一片。"""

        shard_size = max(1, int(manifest.get("shard_size") or 1))
        if end <= start:
            return
        first = start // shard_size
        last = (end - 1) // shard_size
        for shard_index in range(first, last + 1):
            shard = self._read_shard(run_id, manifest, shard_index)
            base = shard_index * shard_size
            low = max(0, start - base)
            high = min(len(shard), end - base)
            for local in range(low, high):
                yield base + local, shard[local]

    def _read_items(self, run_id: str, manifest: dict, positions) -> dict:
        """只读命中位置所在的分片，每片最多读一次。"""

        shard_size = max(1, int(manifest.get("shard_size") or 1))
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position in positions:
            grouped.setdefault(position // shard_size, []).append((position, position % shard_size))
        result: dict[int, dict] = {}
        for shard_index in sorted(grouped):
            shard = self._read_shard(run_id, manifest, shard_index)
            for position, local in grouped[shard_index]:
                if local >= len(shard):
                    raise ClusterError("ARTIFACT_INVALID", "Semantic run integrity check failed", 500)
                result[position] = shard[local]
        return result


def _build_predicate(assignment_status, keyword):
    """把两个可选筛选合成一个谓词；都为空则返回 ``None``（走快速切片路径）。"""

    predicates = []
    if assignment_status is not None:
        expected = str(assignment_status)
        predicates.append(lambda item: str(item.get("assignment_status")) == expected)
    if keyword:
        token = str(keyword)
        predicates.append(lambda item: token in str(item.get("text") or ""))
    if not predicates:
        return None
    if len(predicates) == 1:
        return predicates[0]

    def combined(item):
        return all(predicate(item) for predicate in predicates)

    return combined


def _bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
