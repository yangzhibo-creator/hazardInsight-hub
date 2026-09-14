"""偏差数据库（语义知识库）的构建、列举与请求级覆盖。

三件事必须被钉住，否则「偏差数据库」页面会看起来正常、实际用不了：

1. ``build_knowledge_base`` 建出的库能被 ``list`` / ``detail`` 读出来，
   且条目内容来自旁路快照——ChromaDB 只存向量，没有快照就只能看元信息；
2. 请求级 ``knowledge_base_id`` 只对检索增强 profile 生效，对纯向量 profile
   必须显式报错，而不是静默忽略；
3. 覆盖后 profile 的指纹随之变化，保证策略缓存不会把两个知识库的结果混用。
"""

from __future__ import annotations

import numpy as np
import pytest

from retrain_cluster.config import Settings
from retrain_cluster.errors import ClusterError
from retrain_cluster.services.clustering import ClusteringService
from retrain_cluster.types import ResolvedPipelineConfig


def _spear_profile(**overrides) -> ResolvedPipelineConfig:
    raw = {
        "profile_id": "spear_retrieval",
        "algorithm": "agglomerative",
        "model_id": "fixture",
        "algorithm_params": {"distance_threshold": 0.5},
        "features": {"beta": 0.5, "n_results": 6, "pca_dim": 0, "version": "spear-v1", "reduction": "PCA"},
        "purification": {"enabled": True, "backend": "rule"},
        "knowledge_base_id": "kb-default",
        "implementation_version": "spear-v1",
    }
    raw.update(overrides)
    return ResolvedPipelineConfig.from_dict(raw)


def _service(tmp_path) -> ClusteringService:
    """构造一个只依赖临时目录的引擎实例（不加载模型）。"""

    (tmp_path / "profiles").mkdir(exist_ok=True)
    (tmp_path / "models.toml").write_text(
        '[[models]]\nmodel_id = "fixture"\nprovider = "openai_compatible"\nsource = "fixture"\n'
        'base_url = "http://127.0.0.1:9/v1"\nrevision = "fixture-v1"\ndimension = 8\n',
        encoding="utf-8",
    )
    (tmp_path / "app.toml").write_text(
        '[app]\nmodels_file="models.toml"\nprofiles_dir="profiles"\nartifacts_dir="artifacts"\n',
        encoding="utf-8",
    )
    return ClusteringService(Settings.load(tmp_path / "app.toml"))


def _write_manifest(root, ident: str, count: int) -> None:
    path = root / ident
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        '{"knowledge_base_id": "%s", "status": "completed", "verification": "verified",'
        ' "processing": "raw-upload-v1", "count": %d, "dimension": 8, "metric": "l2",'
        ' "model_fingerprint": "fp"}' % (ident, count),
        encoding="utf-8",
    )


# ------------------------------------------------------------------ 请求级覆盖


def test_knowledge_base_override_replaces_id_and_fingerprint(tmp_path):
    service = _service(tmp_path)
    profile = _spear_profile()
    overridden = service.with_knowledge_base(profile, "kb-new")

    assert overridden.knowledge_base_id == "kb-new"
    # 指纹必须变化：策略实例按指纹缓存，指纹不变会让两个库共用同一份检索器
    assert overridden.fingerprint != profile.fingerprint


def test_knowledge_base_override_is_noop_when_absent_or_same(tmp_path):
    service = _service(tmp_path)
    profile = _spear_profile()
    assert service.with_knowledge_base(profile, None) is profile
    assert service.with_knowledge_base(profile, "kb-default") is profile


def test_knowledge_base_override_on_pure_vector_profile_is_rejected(tmp_path):
    service = _service(tmp_path)
    profile = _spear_profile(
        features={"beta": 1.0, "n_results": 0, "pca_dim": 0, "version": "spear-v1", "reduction": "PCA"},
        knowledge_base_id=None,
    )
    with pytest.raises(ClusterError) as excinfo:
        service.with_knowledge_base(profile, "kb-new")
    assert excinfo.value.code == "INVALID_PROFILE"


def test_knowledge_base_override_clamps_neighbors_to_kb_size(tmp_path):
    """演示用的小知识库必须能用：k > 条目数时把 k 下调为条目数，而不是直接拒绝。"""

    service = _service(tmp_path)
    root = service.settings.artifacts_dir / "knowledge_bases"
    _write_manifest(root, "kb-small", count=3)

    overridden = service.with_knowledge_base(_spear_profile(), "kb-small")
    assert overridden.features.n_results == 3
    assert overridden.knowledge_base_id == "kb-small"


def test_knowledge_base_count_reads_manifest(tmp_path):
    service = _service(tmp_path)
    root = service.settings.artifacts_dir / "knowledge_bases"
    _write_manifest(root, "kb-x", count=17)
    assert service.knowledge_base_count("kb-x") == 17
    assert service.knowledge_base_count("kb-missing") == 0


# ------------------------------------------------------------------ 构建与列举


class _FakeEncoder:
    """固定维度、可复现的假编码器：构建流程不该依赖真实模型。"""

    def encode(self, texts):
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0


def test_build_list_and_detail_roundtrip(tmp_path):
    pytest.importorskip("chromadb")
    from retrain_cluster.retrieval.build import build_knowledge_base
    from retrain_cluster.retrieval.registry import list_knowledge_bases, read_knowledge_detail

    root = tmp_path / "knowledge_bases"
    root.mkdir()
    meta = build_knowledge_base(
        root,
        "kb-test",
        ["支架安装偏差", "模板龙骨间距过大", "名称与系统不一致"],
        _FakeEncoder(),
        "fp-test",
        4,
        None,
        source_sha256="deadbeef",
        display_name="测试库",
        source_name="corpus.txt",
        corpus_texts=["支架安装偏差", "模板龙骨间距过大", "名称与系统不一致"],
        created_at="2026-01-01T00:00:00+00:00",
        extra_meta={"model_id": "fixture"},
    )
    assert meta["status"] == "completed"
    assert meta["count"] == 3
    assert meta["display_name"] == "测试库"

    items = list_knowledge_bases(root)
    assert len(items) == 1
    summary = items[0]
    assert summary["knowledge_base_id"] == "kb-test"
    assert summary["display_name"] == "测试库"
    assert summary["entry_count"] == 3
    assert summary["has_entries"] is True
    assert summary["has_corpus"] is True

    detail = read_knowledge_detail(root, "kb-test", offset=0, limit=2)
    assert detail["entries_source"] == "snapshot"
    assert [entry["text"] for entry in detail["entries"]] == ["支架安装偏差", "模板龙骨间距过大"]

    # 分页：第二页只剩一条，且序号连续
    rest = read_knowledge_detail(root, "kb-test", offset=2, limit=5)
    assert [entry["text"] for entry in rest["entries"]] == ["名称与系统不一致"]
    assert rest["entries"][0]["index"] == 2


def test_list_skips_broken_directories(tmp_path):
    """坏目录（无清单 / 清单损坏）不能让整个列表 500。"""

    root = tmp_path / "knowledge_bases"
    (root / "kb-good").mkdir(parents=True)
    (root / "kb-good" / "manifest.json").write_text(
        '{"knowledge_base_id": "kb-good", "status": "completed", "verification": "verified",'
        ' "processing": "raw-lines-v1", "count": 2, "dimension": 4, "metric": "l2",'
        ' "model_fingerprint": "fp"}',
        encoding="utf-8",
    )
    (root / "kb-broken").mkdir()
    (root / "kb-broken" / "manifest.json").write_text("{not json", encoding="utf-8")

    from retrain_cluster.retrieval.registry import list_knowledge_bases

    items = list_knowledge_bases(root)
    assert [item["knowledge_base_id"] for item in items] == ["kb-good"]


def test_detail_without_snapshot_reports_unavailable(tmp_path):
    """历史库（只有清单、没有快照）必须如实说明看不到内容，而不是返回空列表装正常。"""

    root = tmp_path / "knowledge_bases"
    path = root / "kb-legacy"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        '{"knowledge_base_id": "kb-legacy", "status": "completed", "verification": "verified",'
        ' "processing": "raw-lines-v1", "count": 18170, "dimension": 1024, "metric": "l2",'
        ' "model_fingerprint": "fp", "source_sha256": "unknown"}',
        encoding="utf-8",
    )

    from retrain_cluster.retrieval.registry import read_knowledge_detail

    detail = read_knowledge_detail(root, "kb-legacy")
    assert detail["entries_source"] == "unavailable"
    assert detail["entry_total"] == 18170
    assert detail["entries"] == []
    assert detail["warnings"]
