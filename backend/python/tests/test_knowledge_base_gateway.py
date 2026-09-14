"""偏差数据库网关服务的契约测试。

不加载向量模型、ChromaDB 与 Qwen：用替身引擎 + 临时目录，钉住
「上传语料 → 提交异步作业 → 查看进度/列表/详情」这条链路的准入与读取口径。

真实建库（净化 + 向量化 + Chroma 写入）由 cluster-engine 的
``tests/unit/test_knowledge_bases.py`` 覆盖。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import GatewaySettings
from app.core.errors import InvalidInputError, JobNotFoundError, ServiceBusyError
from app.services.knowledge_base_service import KnowledgeBaseService


class _Catalog:
    profiles: dict = {}

    def __init__(self) -> None:
        self.models = {"fixture": {"dimension": 4, "model_id": "fixture"}}

    def model(self, ident: str):
        if ident not in self.models:
            raise KeyError(ident)
        return self.models[ident]


class _Engine:
    def __init__(self, artifacts: Path) -> None:
        self.settings = SimpleNamespace(artifacts_dir=artifacts)
        self.catalog = _Catalog()


class _ClusterService:
    def __init__(self, artifacts: Path) -> None:
        self._engine = _Engine(artifacts)

    def require_engine(self):
        return self._engine


@pytest.fixture
def service(tmp_path):
    artifacts = tmp_path / "cluster-engine" / "artifacts"
    (artifacts / "knowledge_bases").mkdir(parents=True)
    settings = GatewaySettings(knowledge_base_store=tmp_path / "gw-artifacts")
    return KnowledgeBaseService(settings, _ClusterService(artifacts))


# ------------------------------------------------------------------ 语料解析


def test_parse_corpus_dedupes_and_drops_blank_lines(service):
    data = "支架安装偏差\n\n模板龙骨间距过大\n支架安装偏差\n".encode("utf-8")
    parsed = service.parse_corpus(data, "corpus.txt")

    assert parsed["texts"] == ["支架安装偏差", "模板龙骨间距过大"]
    assert any("重复" in warning for warning in parsed["warnings"])


def test_parse_corpus_rejects_empty_and_unsupported(service):
    # 空语料可能由底层解析器（FileParseError）或网关（InvalidInputError）拦下，
    # 两者都是 400 级的 ClusterGatewayError；这里只钉住"必须被拒绝"。
    from app.core.errors import ClusterGatewayError

    with pytest.raises(ClusterGatewayError):
        service.parse_corpus(b"\n\n", "empty.txt")
    from app.core.errors import UnsupportedFileError

    with pytest.raises(UnsupportedFileError):
        service.parse_corpus(b"abc", "corpus.pdf")


def test_parse_corpus_reads_csv_text_column(service):
    data = "id,隐患描述\n1,支架安装偏差\n2,模板间距过大\n".encode("utf-8")
    parsed = service.parse_corpus(data, "corpus.csv")
    assert parsed["texts"] == ["支架安装偏差", "模板间距过大"]


# ------------------------------------------------------------------ 标识分配


def test_allocate_ident_slugifies_and_avoids_collision(service):
    root = service._kb_root()
    root.mkdir(parents=True, exist_ok=True)

    first = service.allocate_ident(root, "2026 Q1 巡检")
    assert first == "kb-2026-q1"
    (root / first).mkdir()
    second = service.allocate_ident(root, "2026 Q1 巡检")
    assert second == f"{first}-2"


def test_allocate_ident_falls_back_to_timestamp_for_cjk(service):
    root = service._kb_root()
    root.mkdir(parents=True, exist_ok=True)
    ident = service.allocate_ident(root, "核电偏差语料")
    assert ident.startswith("kb-20")  # 中文无法 slug，用时间戳兜底


def test_allocate_ident_rejects_bad_explicit_id(service):
    root = service._kb_root()
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidInputError):
        service.allocate_ident(root, "x", explicit="../escape")


# ------------------------------------------------------------------ 列表与详情


def _write_kb(root: Path, ident: str, entries: list[str]) -> None:
    path = root / ident
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        '{'
        f'"knowledge_base_id": "{ident}", "status": "completed", "verification": "verified",'
        '"processing": "purified-qwen-v1", "count": %d, "dimension": 1024, "metric": "l2",'
        '"model_fingerprint": "fp", "display_name": "测试库", "model_id": "fixture",'
        '"source_name": "corpus.txt", "created_at": "2026-01-01T00:00:00+00:00"'
        '}' % len(entries),
        encoding="utf-8",
    )
    with (path / "entries.jsonl").open("w", encoding="utf-8") as handle:
        for text in entries:
            handle.write('{"text": "%s"}\n' % text)


def test_list_and_detail(service):
    root = service._kb_root()
    root.mkdir(parents=True, exist_ok=True)
    _write_kb(root, "kb-one", ["甲", "乙", "丙"])

    items = service.list()
    assert len(items) == 1
    assert items[0]["display_name"] == "测试库"
    assert items[0]["entry_count"] == 3

    detail = service.detail("kb-one", offset=0, limit=2)
    assert detail["entries_source"] == "snapshot"
    assert [entry["text"] for entry in detail["entries"]] == ["甲", "乙"]
    assert detail["entry_total"] == 3


# ------------------------------------------------------------------ 生成作业


def test_submit_build_rejects_unknown_mode(service):
    with pytest.raises(InvalidInputError):
        service.submit_build(data=b"a\nb\n", filename="c.txt", display_name="x", kind="magic")


def test_submit_build_is_exclusive(service, monkeypatch):
    monkeypatch.setattr(service, "_run_build", lambda **kwargs: None)
    assert service._build_lock.acquire(blocking=False)
    try:
        with pytest.raises(ServiceBusyError):
            service.submit_build(data=b"a\nb\n", filename="c.txt", display_name="x", kind="raw")
    finally:
        service._build_lock.release()


def test_submit_build_starts_job_and_persists_record(service, monkeypatch):
    released: list[bool] = []

    def fake_run_build(**kwargs):
        # 真实实现会在 finally 里释放锁；替身也必须释放，否则后续用例全被卡住
        released.append(True)
        service._release_build_lock()

    monkeypatch.setattr(service, "_run_build", fake_run_build)

    job = service.submit_build(data=b"one\ntwo\n", filename="c.txt", display_name="巡检语料", kind="raw")
    assert job["status"] == "running"
    assert job["knowledge_base_id"].startswith("kb-")
    assert job["total"] == 2

    # 后台线程会调用替身并释放锁（给它一点点调度时间）
    import time

    for _ in range(100):
        if released:
            break
        time.sleep(0.01)
    assert released, "后台线程没有执行 _run_build"

    stored = service.get_job(job["job_id"])
    assert stored["job_id"] == job["job_id"]
    assert stored["terminal"] is False


def test_get_job_missing_raises(service):
    with pytest.raises(JobNotFoundError):
        service.get_job("kbjob_missing")


def test_discard_partial_removes_incomplete_but_keeps_completed(service):
    root = service._kb_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "kb-partial").mkdir()
    (root / "kb-partial" / "manifest.json").write_text('{"status": "building"}', encoding="utf-8")
    (root / "kb-done").mkdir()
    (root / "kb-done" / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")

    service._discard_partial("kb-partial")
    service._discard_partial("kb-done")

    assert not (root / "kb-partial").exists()
    assert (root / "kb-done").exists()


def test_recover_stale_jobs_marks_running_as_failed(service):
    service._write_job(
        {
            "job_id": "kbjob_stale",
            "status": "running",
            "stage": "purify",
            "processed": 1,
            "total": 10,
            "message": "…",
            "knowledge_base_id": "kb-x",
            "display_name": "x",
            "mode": "raw",
            "model_id": "fixture",
            "source_name": "c.txt",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
            "error": None,
            "warnings": [],
            "terminal": False,
        }
    )
    service.recover_stale_jobs()
    recovered = service.get_job("kbjob_stale")
    assert recovered["status"] == "failed"
    assert recovered["terminal"] is True
    assert recovered["error"]["code"] == "KNOWLEDGE_BASE_BUILD_INTERRUPTED"
