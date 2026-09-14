"""SpearStrategy 的端到端工程契约（替身编码器 + 精确近邻，默认不需要 GPU/模型）。

最重要的一条是验收标准 4：**净化关闭时必须与改动前逐位一致**。
这里用"同一份输入分别跑 legacy profile 与净化关闭的 spear profile，
断言标签逐位相等"来钉住它——这不是近似比较，而是 ``==`` 全等。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retrain_cluster.clustering.registry import ALGORITHMS
from retrain_cluster.config import Catalog, Settings
from retrain_cluster.purification import GuardedPurifier, PurifierRegistry, RulePurifier
from retrain_cluster.services.clustering import ClusteringService
from retrain_cluster.services.strategies import SPEAR_VERSION, build_strategy, list_strategy_versions
from tests.conftest import ExactNeighbors, FixedRegistry

MODELS_TOML = """[[models]]
model_id = "fixture"
provider = "openai_compatible"
source = "fixture"
base_url = "http://127.0.0.1:9/v1"
revision = "fixture-v1"
dimension = 8
"""

ITEMS = [
    {"id": "a", "text": "2025.11.03 巡检发现焊机二维码上名称与系统不一致，已要求更换"},
    {"id": "b", "text": "2025.10.16 巡查不锈钢车间发现已用鞋套未及时回收。"},
    {"id": "c", "text": "1C16 段模板龙骨间距过大"},
    {"id": "d", "text": "14-1 连续段凿毛不到位"},
]

PURIFICATION = {
    "enabled": True,
    "backend": "rule",
    "guard": True,
    "batch_size": 8,
    "max_new_tokens": 32,
}


def _settings(tmp_path: Path, *, n_results: int) -> Settings:
    """搭一份同时含 legacy 与 spear profile 的最小配置。"""

    (tmp_path / "profiles").mkdir(exist_ok=True)
    (tmp_path / "models.toml").write_text(MODELS_TOML, encoding="utf-8")
    (tmp_path / "app.toml").write_text(
        '[app]\nmodels_file="models.toml"\nprofiles_dir="profiles"\nartifacts_dir="artifacts"\n', encoding="utf-8"
    )
    features = {"beta": 1.0 if n_results == 0 else 0.4974, "n_results": n_results, "pca_dim": 0}
    kb = None if n_results == 0 else "fixture-kb"
    legacy = {
        "profile_id": "legacy",
        "algorithm": "agglomerative",
        "model_id": "fixture",
        "algorithm_params": {"distance_threshold": 1.0},
        "features": {**features, "version": "legacy-v1"},
        "knowledge_base_id": kb,
    }
    spear = {
        "profile_id": "spear",
        "algorithm": "agglomerative",
        "model_id": "fixture",
        "algorithm_params": {"distance_threshold": 1.0},
        "features": {**features, "version": "spear-v1"},
        "purification": PURIFICATION,
        "knowledge_base_id": kb,
        "implementation_version": "spear-v1",
    }
    (tmp_path / "profiles" / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    (tmp_path / "profiles" / "spear.json").write_text(json.dumps(spear), encoding="utf-8")
    return Settings.load(tmp_path / "app.toml")


def _service(settings, *, knowledge=None):
    factory = None
    if knowledge is not None:
        factory = lambda *args: ExactNeighbors(knowledge)  # noqa: E731 - 测试替身
    return ClusteringService(settings, encoders=FixedRegistry(), retriever_factory=factory)


def test_purification_off_is_bit_identical_to_the_legacy_path(tmp_path, arrays):
    _, _, knowledge = arrays
    settings = _settings(tmp_path, n_results=0)
    service = _service(settings, knowledge=knowledge)
    legacy = service.cluster({"profile_id": "legacy", "items": ITEMS})
    # 关掉净化：spear 必须退化成改动前的纯向量路径
    profile = service.catalog.profile("spear")
    from dataclasses import replace

    from retrain_cluster.types import PurificationSpec

    service._strategies.clear()
    service.catalog.profiles["spear"] = replace(
        profile, purification=replace(profile.purification, enabled=False)
    )
    spear = service.cluster({"profile_id": "spear", "items": ITEMS})

    assert spear["assignments"] == legacy["assignments"]
    assert spear["n_clusters"] == legacy["n_clusters"]
    assert spear["purification"]["enabled"] is False
    assert spear["purification"]["backend"] == "identity"


def test_retrieval_off_is_bit_identical_to_the_legacy_retrieval_path(tmp_path, arrays):
    _, _, knowledge = arrays
    settings = _settings(tmp_path, n_results=6)
    service = _service(settings, knowledge=knowledge)
    legacy = service.cluster({"profile_id": "legacy", "items": ITEMS})

    profile = service.catalog.profile("spear")
    from dataclasses import replace

    service._strategies.clear()
    service.catalog.profiles["spear"] = replace(profile, purification=replace(profile.purification, enabled=False))
    spear = service.cluster({"profile_id": "spear", "items": ITEMS})

    assert spear["assignments"] == legacy["assignments"]
    assert spear["purification"]["enabled"] is False


def test_spear_with_rule_purification_reports_condensed_samples(tmp_path, arrays):
    _, _, knowledge = arrays
    service = _service(_settings(tmp_path, n_results=0), knowledge=knowledge)
    result = service.cluster({"profile_id": "spear", "items": ITEMS})

    report = result["purification"]
    assert report["enabled"] is True
    assert report["backend"] == "rule"
    assert report["degraded"] is False
    assert report["guarded"] is True
    samples = {sample["id"]: sample for sample in report["samples"]}
    assert "2025.11.03" not in samples["a"]["condensed"]  # 时空信息被剥离
    assert "不一致" in samples["a"]["condensed"]  # 极性词被保留
    assert result["implementation_version"] == SPEAR_VERSION


def test_spear_degrades_to_rule_when_the_local_llm_is_missing(tmp_path, arrays):
    _, _, knowledge = arrays
    settings = _settings(tmp_path, n_results=0)
    service = _service(settings, knowledge=knowledge)
    profile = service.catalog.profile("spear")
    from dataclasses import replace

    service._strategies.clear()
    service.catalog.profiles["spear"] = replace(
        profile, purification=replace(profile.purification, backend="qwen", model_path="/nonexistent/qwen")
    )
    result = service.cluster({"profile_id": "spear", "items": ITEMS})

    assert result["purification"]["degraded"] is True
    assert result["purification"]["backend"] == "rule"
    assert "PURIFIER_DEGRADED_TO_RULE" in result["warnings"]


def test_guard_is_applied_inside_the_strategy(tmp_path, arrays):
    """策略层也必须过护栏：注入一个会反转语义的假净化器，结果不得被污染。"""

    _, _, knowledge = arrays
    settings = _settings(tmp_path, n_results=0)
    service = _service(settings, knowledge=knowledge)
    profile = service.catalog.profile("spear")

    class _Reversing:
        name = "reversing-fake"

        def purify(self, texts):
            return [str(text).replace("不一致", "一致") for text in texts]

    class _StubRegistry(PurifierRegistry):
        def get(self, spec):
            purifier = GuardedPurifier(_Reversing(), fallback=RulePurifier())
            return purifier, self.status(spec)

    strategy = build_strategy(
        profile,
        settings=settings,
        catalog=service.catalog,
        encoders=FixedRegistry(),
        embedder=service.embeddings,
        retriever_for=service.retriever,
        purifiers=_StubRegistry(),
    )
    result = strategy.run(ITEMS)
    condensed = {sample["id"]: sample["condensed"] for sample in result["purification"]["samples"]}
    assert "不一致" in condensed["a"]
    assert "POLARITY_GUARD_TRIGGERED" in result["warnings"]


def test_request_level_override_can_turn_purification_off(tmp_path, arrays):
    """对照实验入口：同一个 profile，一次请求内开/关净化。"""

    _, _, knowledge = arrays
    service = _service(_settings(tmp_path, n_results=0), knowledge=knowledge)
    on = service.cluster({"profile_id": "spear", "items": ITEMS})
    assert on["purification"]["enabled"] is True

    service._strategies.clear()
    off = service.cluster({"profile_id": "spear", "items": ITEMS, "purification": {"enabled": False}})
    assert off["purification"]["enabled"] is False
    assert off["purification"]["backend"] == "identity"


def test_request_level_override_is_rejected_for_profiles_without_purification(tmp_path, arrays):
    _, _, knowledge = arrays
    service = _service(_settings(tmp_path, n_results=0), knowledge=knowledge)
    from retrain_cluster.errors import ClusterError

    with pytest.raises(ClusterError, match="Purification override requires"):
        service.cluster({"profile_id": "legacy", "items": ITEMS, "purification": {"enabled": False}})


def test_spear_capability_is_declared_and_algorithms_come_from_the_registry():
    versions = {entry["implementation_version"]: entry for entry in list_strategy_versions()}
    assert SPEAR_VERSION in versions
    assert set(versions[SPEAR_VERSION]["algorithms"]) <= set(ALGORITHMS)
    assert versions[SPEAR_VERSION]["purification"] is True
