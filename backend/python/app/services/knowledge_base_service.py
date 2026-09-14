"""偏差数据库（聚类语义知识库）服务。

「偏差数据库」是检索增强聚类（nr1 / ``spear_purified_retrieval``）所用的
语义知识库：由一批偏差文本经向量模型编码后写入 ChromaDB，聚类时按近邻检索
补偿表示。它住在 cluster-engine 的 ``artifacts/knowledge_bases/<id>/``，
清单（``manifest.json``）是唯一事实来源。

本服务提供三件事：

* **查看**：列出已有知识库、分页读取某个库的条目。读取刻意不加载向量模型
  与索引（见 ``retrain_cluster.retrieval.registry``），列表页毫秒级返回。
* **生成**：上传语料 →（可选）Qwen 逐条语义净化 → 向量化 → 写入新索引。
  净化是逐条 LLM 推理，两万条要一到三小时，因此**必须异步**：提交立刻返回
  ``jobId``，前端轮询进度；作业完成前不会出现"半成品"知识库（清单只有
  ``status=completed`` 才会被检索端接受）。
* **选择**：聚类页把 ``knowledgeBaseId`` 放进运行选项，由引擎在解析 profile 后
  覆盖其默认知识库。

并发约束（很关键）：生成与聚类共用同一块 GPU 和同一份模型内存。这里用
``_build_lock`` 保证同一时刻只有一个生成任务；提交时若已有任务在跑，直接
返回 429 而不是排队——排队只会把 OOM 从"立刻拒绝"推迟成"跑一半崩掉"。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import threading
import uuid
from typing import Any

from app.core.config import GatewaySettings, get_settings
from app.core.constants import (
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    MAX_ITEMS,
    MAX_TEXT_CHARS,
    SUPPORTED_UPLOAD_EXTENSIONS,
)
from app.core.errors import InvalidInputError, JobNotFoundError, ServiceBusyError, UnsupportedFileError
from app.core.logger import get_logger
from app.processors.tabular_reader import read_records
from app.services.clustering_service import ClusteringGatewayService, _from_engine_error

logger = get_logger(__name__)

#: 知识库标识白名单：与 cluster-engine 的 ``chroma.kb_path`` 保持一致，
#: 避免"页面能建、检索用不了"。
_IDENT_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")

#: 语料处理方式。
KIND_PURIFIED = "purified"
KIND_RAW = "raw"
KNOWN_KINDS = (KIND_PURIFIED, KIND_RAW)

#: 生成作业的终态集合。
_TERMINAL = frozenset({JOB_SUCCEEDED, JOB_FAILED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slugify(name: str) -> str:
    """把可读名称压成 ASCII 标识片段；中文等非 ASCII 字符会被丢弃。"""

    return re.sub(r"[^A-Za-z0-9]+", "-", name or "").strip("-").lower()


class KnowledgeBaseService:
    """偏差数据库的只读查询 + 异步生成。"""

    def __init__(self, settings: GatewaySettings, cluster_service: ClusteringGatewayService) -> None:
        self.settings = settings
        self.cluster = cluster_service
        #: 保护作业记录文件的读写
        self._lock = threading.Lock()
        #: 同一时刻只允许一个生成任务（GPU/显存约束）
        self._build_lock = threading.Lock()

    # ------------------------------------------------------------------ 路径

    @property
    def job_root(self) -> Path:
        """生成作业记录的目录（不存在时自动创建）。"""

        path = self.settings.knowledge_base_store_path / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _kb_root(self) -> Path:
        """知识库本体根目录（由引擎配置决定，与检索端完全一致）。"""

        engine = self.cluster.require_engine()
        return Path(engine.settings.artifacts_dir) / "knowledge_bases"

    def _source_candidates(self, root: Path) -> list[Path]:
        """历史知识库的来源语料候选（按 ``source_sha256`` 匹配）。

        仓库里不分发 ``database.txt``，现场把语料放在 ``cluster-engine/data/``。
        优先用引擎配置里声明的 ``experiment.knowledge_texts``，再退回
        ``<cluster-engine>/data/*.txt``（``artifacts_dir`` 的同级 data 目录）。
        匹配失败只会退化成"看不到条目内容"，不影响元信息展示。
        """

        candidates: list[Path] = []
        engine = self.cluster.require_engine()
        experiment = getattr(engine.settings, "experiment", None) or {}
        declared = experiment.get("knowledge_texts")
        if declared:
            candidates.append(Path(declared))
        data_dir = Path(engine.settings.artifacts_dir).parent / "data"
        if data_dir.is_dir():
            candidates.extend(sorted(data_dir.glob("*.txt")))
        sources = root / "_sources"
        if sources.is_dir():
            candidates.extend(sorted(sources.glob("*")))
        return candidates

    # ------------------------------------------------------------------ 查询

    def list(self) -> list[dict[str, Any]]:
        """列出全部知识库（新→旧）。"""

        from retrain_cluster.retrieval.registry import list_knowledge_bases

        try:
            return list_knowledge_bases(self._kb_root())
        except Exception as exc:  # noqa: BLE001 - 目录不存在时返回空列表，不报 500
            logger.warning("知识库列表读取失败：%s", exc)
            return []

    def detail(self, ident: str, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """读取某个知识库的元信息与一页条目内容。"""

        from retrain_cluster.retrieval.registry import read_knowledge_detail

        root = self._kb_root()
        try:
            return read_knowledge_detail(
                root,
                ident,
                offset=offset,
                limit=limit,
                source_candidates=self._source_candidates(root),
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成网关错误
            raise _from_engine_error(exc) from exc

    # ------------------------------------------------------------------ 语料

    def parse_corpus(self, data: bytes, filename: str) -> dict[str, Any]:
        """把上传的语料解析成去重后的文本列表。

        只做"取文本 + 去首尾空白 + 按长度截断 + 精确去重"，**不做** NFKC /
        标点归一化：知识库条目要尽量保留语料原貌，聚类查询侧怎么清洗是另一回事。
        """

        if Path(filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            allowed = " / ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
            raise UnsupportedFileError(f"不支持的文件类型：{filename}。请上传 {allowed} 文件。")
        try:
            records, read_warnings = read_records(data, filename)
        except Exception as exc:  # noqa: BLE001 - FileParseError 等统一成网关错误
            from app.core.errors import FileParseError

            if isinstance(exc, FileParseError):
                raise
            raise InvalidInputError(f"语料解析失败：{exc}") from exc

        maximum = min(self.settings.knowledge_base_max_items, MAX_ITEMS)
        texts: list[str] = []
        seen: set[str] = set()
        duplicates = 0
        truncated = 0
        for record in records:
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            if len(text) > MAX_TEXT_CHARS:
                text = text[: MAX_TEXT_CHARS - 1].rstrip() + "…"
                truncated += 1
            if text in seen:
                duplicates += 1
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= maximum:
                break
        if not texts:
            raise InvalidInputError("语料中没有可用的非空文本，请检查文件内容与文本列。")

        warnings = list(read_warnings)
        if duplicates:
            warnings.append(f"已对 {duplicates} 条完全重复的文本去重。")
        if truncated:
            warnings.append(f"{truncated} 条文本超过 {MAX_TEXT_CHARS} 字符，已截断。")
        if len(texts) >= maximum:
            warnings.append(f"语料条目已达上限 {maximum} 条，其余被忽略。")
        return {"source_name": filename, "texts": texts, "warnings": warnings}

    def allocate_ident(self, root: Path, name: str, explicit: str | None = None) -> str:
        """分配一个未被占用的知识库标识。"""

        if explicit:
            candidate = explicit.strip()
            if not _IDENT_RE.fullmatch(candidate):
                raise InvalidInputError(
                    "知识库标识只能包含字母、数字、下划线、点和短横线，且以字母或数字开头（最长 128 字符）。"
                )
            if (root / candidate).exists():
                raise InvalidInputError(f"知识库标识「{candidate}」已存在，请换一个。")
            return candidate

        slug = _slugify(name)[:48]
        stem = slug or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        candidate = f"kb-{stem}"[:128]
        if not (root / candidate).exists():
            return candidate
        for index in range(2, 1000):
            alternative = f"{candidate[:120]}-{index}"
            if not (root / alternative).exists():
                return alternative
        raise InvalidInputError("无法分配知识库标识，请手动指定一个。")

    # ------------------------------------------------------------------ 生成

    def _default_model_id(self, engine: Any) -> str:
        models = list(engine.catalog.models)
        if not models:
            raise InvalidInputError("引擎未配置任何向量模型，无法构建知识库。")
        # 优先沿用 SPEAR profile 使用的模型，保证与检索端口径一致
        for profile in engine.catalog.profiles.values():
            if profile.knowledge_base_id:
                return str(profile.model_id)
        return str(models[0])

    @staticmethod
    def _purification_spec(engine: Any) -> Any:
        """取一份"强制走本地 LLM"的净化配置。

        刻意不用 profile 里的 ``backend=auto``：``auto`` 会优先命中预计算的
        ``purify_map.json``，未命中的文本交给**规则**兜底——用户上传的新语料
        几乎必然全部未命中，于是"让大模型生成"会静默变成"规则生成"。
        这里改为 ``backend=qwen``：要么真的用大模型，要么在状态里如实报告降级。
        """

        from dataclasses import replace

        for profile in engine.catalog.profiles.values():
            spec = getattr(profile, "purification", None)
            if spec is not None:
                return replace(spec, enabled=True, backend="qwen", cache_file=None)
        raise InvalidInputError(
            "引擎没有配置净化模型（Qwen），无法按「大模型生成」建库；请改用「原文直接入库」。"
        )

    def submit_build(
        self,
        *,
        data: bytes,
        filename: str,
        display_name: str,
        kind: str,
        ident: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """提交一次知识库生成任务并立刻返回作业状态。"""

        if kind not in KNOWN_KINDS:
            raise InvalidInputError(f"未知的语料处理方式「{kind}」，可选：{', '.join(KNOWN_KINDS)}。")
        if not data:
            raise InvalidInputError("上传的语料文件是空的。")
        if len(data) > self.settings.upload_max_bytes:
            raise InvalidInputError(
                f"语料文件 {len(data) / 1024 / 1024:.1f} MB 超过上限 "
                f"{self.settings.upload_max_bytes / 1024 / 1024:.0f} MB。"
            )

        if not self._build_lock.acquire(blocking=False):
            raise ServiceBusyError("已有知识库生成任务在执行，请等待其完成后再试。")

        try:
            engine = self.cluster.require_engine()
            root = self._kb_root()
            root.mkdir(parents=True, exist_ok=True)
            parsed = self.parse_corpus(data, filename)
            target_id = self.allocate_ident(root, display_name or filename, explicit=ident)
            target_model = model_id or self._default_model_id(engine)
            # 提前校验模型存在，避免任务跑起来才发现模型名写错
            engine.catalog.model(target_model)
        except Exception:
            self._build_lock.release()
            raise

        job = {
            "job_id": "kbjob_" + uuid.uuid4().hex[:12],
            "status": JOB_RUNNING,
            "stage": "queued",
            "processed": 0,
            "total": len(parsed["texts"]),
            "message": "任务已提交，等待开始…",
            "knowledge_base_id": target_id,
            "display_name": display_name or target_id,
            "mode": kind,
            "model_id": target_model,
            "source_name": parsed["source_name"],
            "created_at": _now(),
            "updated_at": _now(),
            "finished_at": None,
            "error": None,
            "warnings": list(parsed["warnings"]),
            "terminal": False,
        }
        self._write_job(job)
        thread = threading.Thread(
            target=self._run_build,
            kwargs={
                "job_id": job["job_id"],
                "target_id": target_id,
                "texts": parsed["texts"],
                "kind": kind,
                "model_id": target_model,
                "source_sha256": _sha256(data),
            },
            name=f"kb-build-{job['job_id']}",
            daemon=True,
        )
        thread.start()
        return job

    def _run_build(
        self,
        *,
        job_id: str,
        target_id: str,
        texts: list[str],
        kind: str,
        model_id: str,
        source_sha256: str,
    ) -> None:
        """后台执行：净化 → 向量化 → 建索引。任何异常都落成 failed 状态。"""

        try:
            engine = self.cluster.require_engine()
            root = self._kb_root()
            record = self._read_job(job_id)
            model = engine.catalog.model(model_id)
            from retrain_cluster.config import model_fingerprint

            model_hash = model_fingerprint(model)
            dimension = int(model["dimension"])

            processed = texts
            warnings: list[str] = []
            if kind == KIND_PURIFIED:
                spec = self._purification_spec(engine)
                purifier, status = engine.purifier_registry().get(spec)
                if status.degraded:
                    warnings.append(
                        f"净化模型不可用，已降级为规则净化（{status.reason or 'unknown'}）；"
                        "入库文本并非大模型产出。"
                    )
                logger.info(
                    "知识库 %s：开始净化 %d 条（后端 %s）", target_id, len(texts), status.effective_backend
                )
                self._update(job_id, stage="purify", processed=0, message="正在调用大模型净化语料…")
                processed = self._purify_in_chunks(job_id, purifier, texts)
                # 净化结果为空时退回原文：索引里放空串没有语义，还会污染检索
                processed = [value if str(value).strip() else raw for value, raw in zip(processed, texts)]

            self._update(
                job_id,
                stage="embed",
                processed=0,
                total=len(processed),
                message="正在向量化语料…",
            )
            values, _, _ = engine.embeddings(processed, model_id)

            self._update(
                job_id,
                stage="index",
                processed=0,
                total=len(processed),
                message="正在写入向量索引…",
            )
            from retrain_cluster.retrieval.build import build_knowledge_base

            meta = build_knowledge_base(
                root,
                target_id,
                processed,
                None,
                model_hash,
                dimension,
                values=values,
                processing="purified-qwen-v1" if kind == KIND_PURIFIED else "raw-upload-v1",
                display_name=record.get("display_name") or target_id,
                source_name=record.get("source_name"),
                source_sha256=source_sha256,
                corpus_texts=texts,
                created_at=_now(),
                extra_meta={"model_id": model_id, "mode": kind},
            )
            self._update(
                job_id,
                status=JOB_SUCCEEDED,
                stage="done",
                processed=len(processed),
                total=len(processed),
                message=f"知识库「{target_id}」已生成，共 {meta.get('count', len(processed))} 条。",
                knowledge_base_id=target_id,
                warnings=warnings,
                terminal=True,
                finished_at=_now(),
            )
            logger.info("知识库 %s 生成完成：%d 条", target_id, meta.get("count", len(processed)))
        except Exception as exc:  # noqa: BLE001 - 后台线程不能把异常抛给调用方
            logger.exception("知识库 %s 生成失败：%s", target_id, exc)
            self._discard_partial(target_id)
            converted = _from_engine_error(exc) if hasattr(exc, "code") or hasattr(exc, "message") else None
            self._update(
                job_id,
                status=JOB_FAILED,
                stage="failed",
                message="生成失败。",
                error={
                    "code": getattr(converted, "code", None) or getattr(exc, "code", "KNOWLEDGE_BASE_BUILD_FAILED"),
                    "message": getattr(converted, "message", None) or str(exc) or "未知错误",
                },
                terminal=True,
                finished_at=_now(),
            )
        finally:
            self._release_build_lock()

    def _discard_partial(self, ident: str) -> None:
        """清理生成失败留下的半成品目录。

        ``build_knowledge_base`` 先建目录再写清单，写索引失败时会留下一个
        ``status=building`` 的目录：检索端本来就不会接受它，但用户也无法再用
        同一个标识重试。这里把它删掉，让"失败"等于"没发生过"，可原样重来。
        只删未完成的目录——已完成的知识库绝不动。
        """

        import json
        import shutil

        path = self._kb_root() / ident
        manifest = path / "manifest.json"
        try:
            if manifest.is_file():
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                if meta.get("status") == "completed":
                    return
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                logger.info("已清理未完成的知识库目录：%s", ident)
        except (OSError, ValueError) as exc:  # noqa: BLE001 - 清理失败不影响作业状态
            logger.warning("清理未完成知识库 %s 失败：%s", ident, exc)

    def _purify_in_chunks(self, job_id: str, purifier: Any, texts: list[str]) -> list[str]:
        """分块净化并逐块上报进度（一块一落盘，进度不会长时间不动）。"""

        chunk = max(1, int(self.settings.knowledge_base_purify_chunk))
        results: list[str] = []
        total = len(texts)
        for start in range(0, total, chunk):
            piece = texts[start : start + chunk]
            results.extend(purifier.purify(piece))
            done = min(start + chunk, total)
            self._update(
                job_id,
                stage="purify",
                processed=done,
                total=total,
                message=f"正在净化语料（{done}/{total}）…",
            )
        return results

    # ------------------------------------------------------------------ 作业存储

    def _job_path(self, job_id: str) -> Path:
        return self.job_root / f"{job_id}.json"

    def _write_job(self, record: dict[str, Any]) -> None:
        from retrain_cluster.artifacts.runs import atomic_json

        with self._lock:
            atomic_json(self._job_path(record["job_id"]), record)

    def _read_job(self, job_id: str) -> dict[str, Any]:
        import json

        path = self._job_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"知识库生成任务 {job_id} 不存在。")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JobNotFoundError(f"知识库生成任务 {job_id} 的记录已损坏。") from exc
        return payload

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            import json

            path = self._job_path(job_id)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            record.update(changes)
            record["updated_at"] = _now()
            from retrain_cluster.artifacts.runs import atomic_json

            atomic_json(path, record)

    def _release_build_lock(self) -> None:
        try:
            self._build_lock.release()
        except RuntimeError:  # pragma: no cover - 重复释放不应影响作业状态
            pass

    def get_job(self, job_id: str) -> dict[str, Any]:
        """读取生成任务状态。"""

        record = self._read_job(job_id)
        record["terminal"] = record.get("status") in _TERMINAL
        return record

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """列出生成任务（新→旧）。"""

        import json

        records: list[dict[str, Any]] = []
        if self.job_root.is_dir():
            for path in sorted(self.job_root.glob("kbjob_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                payload["terminal"] = payload.get("status") in _TERMINAL
                records.append(payload)
                if len(records) >= max(1, limit):
                    break
        return records

    def recover_stale_jobs(self) -> None:
        """进程重启后，把遗留的 running 作业标记为失败。

        这类作业的线程已经随进程消失，若不标记，页面会看到一个永远转圈的
        "生成中"。标记为失败比假装成功诚实：产物目录不存在，清单也不会是
        completed，检索端本来就不会接受它。
        """

        for record in self.list_jobs(limit=100):
            if record.get("status") == JOB_RUNNING and not record.get("terminal"):
                self._update(
                    record["job_id"],
                    status=JOB_FAILED,
                    stage="failed",
                    message="服务重启，任务已中断。",
                    error={"code": "KNOWLEDGE_BASE_BUILD_INTERRUPTED", "message": "服务重启导致生成中断，请重新提交。"},
                    terminal=True,
                    finished_at=_now(),
                )
        # 进程重启后锁必然空闲
        self._release_build_lock()


__all__ = ["KnowledgeBaseService", "KIND_PURIFIED", "KIND_RAW", "KNOWN_KINDS"]
