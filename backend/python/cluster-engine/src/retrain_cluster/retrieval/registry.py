"""知识库清单的读取与校验（只读，不加载向量模型与 ChromaDB 索引）。

「偏差数据库」页面需要在不付出「加载 1.3GB 向量模型」代价的前提下，列出
现有知识库、查看某个库的条目内容。这里因此刻意只做**文件系统级**读取：

* 清单 ``manifest.json`` 是唯一事实来源，校验状态/维度/条目数都取自它；
* 条目内容读取优先级：``entries.jsonl``（构建时旁路快照）→ ``corpus.txt``
  （上传语料快照）→ 调用方给出的候选来源文件（按 ``source_sha256`` 匹配，
  历史 ``bge-large-raw-lines-v1`` 走这条）；
* 任何缺失都不抛异常，而是如实返回 ``entries_source``，让界面说明"只能看到
  元信息，看不到内容"，而不是假装库是空的。

真正的向量一致性校验发生在检索时（:mod:`retrain_cluster.retrieval.chroma`），
本模块不替代它。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re

from .build import CORPUS_FILE, ENTRIES_FILE
from .chroma import kb_path

#: 与 chroma.kb_path 共用同一份标识白名单，避免"列得出来、检索用不了"。
_IDENT_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def read_manifest_file(directory: Path, ident: str) -> dict | None:
    """读取单个知识库清单；不存在或不可解析时返回 None（列表要能容忍坏目录）。"""

    if not _IDENT_RE.fullmatch(ident):
        return None
    path = Path(directory) / ident
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def summarize(directory: Path, manifest: dict) -> dict:
    """把清单压成列表页需要的摘要（不含条目内容）。"""

    ident = str(manifest.get("knowledge_base_id") or "")
    path = Path(directory) / ident
    created_at = manifest.get("created_at")
    if not created_at:
        try:
            created_at = _iso(path.stat().st_mtime)
        except OSError:
            created_at = None
    return {
        "knowledge_base_id": ident,
        "display_name": manifest.get("display_name") or ident,
        "status": manifest.get("status") or "unknown",
        "verification": manifest.get("verification") or "unknown",
        "processing": manifest.get("processing") or "unknown",
        "entry_count": int(manifest.get("count") or 0),
        "dimension": int(manifest.get("dimension") or 0),
        "metric": manifest.get("metric") or "",
        "model_fingerprint": manifest.get("model_fingerprint") or "",
        "model_id": manifest.get("model_id"),
        "source_name": manifest.get("source_name"),
        "source_sha256": manifest.get("source_sha256") or "",
        "created_at": created_at,
        "size_bytes": _dir_size_bytes(path),
        "has_entries": (path / ENTRIES_FILE).is_file(),
        "has_corpus": (path / CORPUS_FILE).is_file(),
    }


def list_knowledge_bases(directory) -> list[dict]:
    """列出全部知识库（新→旧）。坏目录被跳过而不是让整个列表失败。"""

    root = Path(directory)
    if not root.is_dir():
        return []
    items: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        manifest = read_manifest_file(root, child.name)
        if manifest is None:
            continue
        items.append(summarize(root, manifest))
    # 新→旧；created_at 缺失的排在最后，避免被当成"刚建的"
    items.sort(key=lambda item: (item.get("created_at") or "", item["knowledge_base_id"]), reverse=True)
    return items


def _count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            total += 1
    return total


def _read_entries_page(path: Path, offset: int, limit: int) -> list[str]:
    entries: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        index = 0
        for line in handle:
            if index < offset:
                index += 1
                continue
            if len(entries) >= limit:
                break
            line = line.strip()
            if not line:
                index += 1
                continue
            if path.name == ENTRIES_FILE:
                try:
                    payload = json.loads(line)
                    text = str(payload.get("text", ""))
                except ValueError:
                    text = line
            else:
                text = line
            entries.append(text)
            index += 1
    return entries


def _locate_source(directory: Path, ident: str, manifest: dict, candidates) -> Path | None:
    """按 ``source_sha256`` 在候选文件里定位历史来源语料。"""

    expected = str(manifest.get("source_sha256") or "")
    if not expected:
        return None
    from ..artifacts.fingerprints import file_hash

    for candidate in candidates or []:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            if file_hash(path) == expected:
                return path
        except OSError:
            continue
    return None


def read_knowledge_detail(
    directory,
    ident: str,
    *,
    offset: int = 0,
    limit: int = 50,
    source_candidates=None,
) -> dict:
    """读取知识库详情：清单 + 一页条目内容。

    ``entries_source`` 取值：
      * ``snapshot`` —— 构建时写入的 ``entries.jsonl``（索引里的真实文本）；
      * ``corpus`` —— 上传语料的原文快照 ``corpus.txt``；
      * ``source`` —— 按指纹匹配到的历史来源文件；
      * ``unavailable`` —— 只能看到元信息，看不到条目内容。
    """

    root = Path(directory)
    manifest = read_manifest_file(root, ident)
    if manifest is None:
        from ..errors import ClusterError

        raise ClusterError("KNOWLEDGE_BASE_UNAVAILABLE", "Knowledge base is missing", 404)

    path = kb_path(root, ident)
    detail = summarize(root, manifest)
    detail["warnings"] = []

    entries_path: Path | None = None
    source_kind = "unavailable"
    if (path / ENTRIES_FILE).is_file():
        entries_path, source_kind = path / ENTRIES_FILE, "snapshot"
    elif (path / CORPUS_FILE).is_file():
        entries_path, source_kind = path / CORPUS_FILE, "corpus"
    else:
        located = _locate_source(root, ident, manifest, source_candidates)
        if located is not None:
            entries_path, source_kind = located, "source"

    total = 0
    entries: list[str] = []
    if entries_path is not None:
        try:
            total = _count_lines(entries_path)
            entries = _read_entries_page(entries_path, max(0, offset), max(1, limit))
        except OSError as exc:  # noqa: BLE001 - 读不到内容不影响元信息展示
            detail["warnings"].append(f"条目内容读取失败：{exc}")
            source_kind, total, entries = "unavailable", 0, []
    else:
        total = detail["entry_count"]
        detail["warnings"].append(
            "该知识库构建时未保留文本快照，只能查看元信息（条目内容不在索引中）。"
        )

    detail.update(
        {
            "entries_source": source_kind,
            "entry_total": total,
            "offset": max(0, offset),
            "limit": max(1, limit),
            "entries": [{"index": offset + position, "text": text} for position, text in enumerate(entries)],
        }
    )
    return detail


__all__ = [
    "list_knowledge_bases",
    "read_knowledge_detail",
    "read_manifest_file",
    "summarize",
]
