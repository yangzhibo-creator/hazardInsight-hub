"""spear-v1 的网关契约测试。

网关在 SPEAR 路径上的职责与 semantic 路径一致：**原样透传引擎结果**，
不二次编码、不重算净化。本模块钉住四件事：

1. 引擎的净化报告与逐条浓缩文本能映射到 ``ClusteringData``；
2. 请求级 ``purify`` 开关被原样透传给引擎（关闭对照走的是同一条代码路径）；
3. ``profile`` 列表如实报告净化降级状态，前端据此显示提示；
4. 离线基准结果可读取且相对增益口径与后端一致。

引擎用替身，因此不需要本地模型或 GPU。
"""

from __future__ import annotations

import copy

import pytest

from app.core.config import get_settings
from app.schemas.clustering import BaselineData, ClusteringData
from app.services.clustering_service import ClusteringGatewayService, _capability_of

SPEAR_PROFILE_ID = "spear_purified"
SPEAR_RETRIEVAL_PROFILE_ID = "spear_purified_retrieval"


def _purification_report() -> dict:
    return {
        "enabled": True,
        "backend": "rule",
        "requested_backend": "auto",
        "degraded": True,
        "reason": "purification_model_missing",
        "guarded": True,
        "guard_hits": 1,
        "total": 4,
        "elapsed_ms": 0.5,
        "samples": [
            {"id": "i1", "raw": "2025.11.03 巡检发现名称与系统不一致", "condensed": "名称与系统不一致"},
            {"id": "i2", "raw": "1C16 段模板龙骨间距过大", "condensed": "模板龙骨间距过大"},
        ],
    }


def _spear_payload(items: list[dict]) -> dict:
    """按请求条数生成结构合法的 SPEAR 载荷（网关有守恒断言）。"""

    ids = [item.get("id") or f"row-{index}" for index, item in enumerate(items, start=1)]
    return {
        "run_id": "run_" + "a" * 32,
        "profile_id": SPEAR_PROFILE_ID,
        "algorithm": "agglomerative",
        "implementation_version": "spear-v1",
        "n_samples": len(ids),
        "n_clusters": 1,
        "n_noise": 0,
        "assignments": [{"id": ident, "cluster_id": 0} for ident in ids],
        "warnings": ["PURIFIER_DEGRADED_TO_RULE"],
        "elapsed_ms": 42,
        "purification": _purification_report(),
    }


class StubEngine:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.embeddings_calls = 0

    def cluster(self, request, *, enforce_api_limits: bool = False):
        self.requests.append({"request": copy.deepcopy(request), "enforce_api_limits": enforce_api_limits})
        return copy.deepcopy(_spear_payload(request.get("items") or []))

    def embeddings(self, texts, model_id):  # pragma: no cover - 被调用即测试失败
        self.embeddings_calls += 1
        raise AssertionError("SPEAR 路径不得二次编码：网关不该调用 engine.embeddings")


@pytest.fixture
def service(monkeypatch):
    from retrain_cluster.config import Catalog, Settings as EngineSettings

    settings = get_settings()
    engine_settings = EngineSettings.load(str(settings.engine_config_path))
    gateway = ClusteringGatewayService(settings)
    monkeypatch.setattr(gateway, "_load_engine", lambda: None)
    monkeypatch.setattr(gateway, "_profile_status", lambda profile: (True, None))
    gateway._catalog = Catalog(engine_settings)
    gateway._engine_settings = engine_settings
    gateway._engine = StubEngine()
    return gateway


def _items(count: int = 4) -> list[dict]:
    return [
        {"id": f"i{index}", "text": f"第 {index} 条隐患描述文本", "metadata": {"area": "A区"}}
        for index in range(1, count + 1)
    ]


def test_spear_payload_maps_onto_the_shared_contract(service):
    result = service.run({"items": _items(), "options": {"profile_id": SPEAR_PROFILE_ID}})
    data = ClusteringData(**result)  # 字段不符会在这里直接报错

    assert data.summary.implementation_version == "spear-v1"
    assert data.summary.purification is not None
    assert data.summary.purification.backend == "rule"
    assert data.summary.purification.degraded is True
    assert data.summary.purification.guard_hits == 1
    assert data.summary.purification.samples[0].condensed == "名称与系统不一致"
    # 逐条浓缩文本挂到明细行上，前端可直接做前后对照
    assert data.items[0].purified_text == "名称与系统不一致"
    # SPEAR 路径网关不复现聚类空间，因此不生成散点坐标（如实为空）
    assert data.visualization == []
    assert "SPEAR_VIZ_NOT_COMPUTED" in data.summary.warnings
    # 绝不二次编码
    assert service._engine.embeddings_calls == 0


def test_purify_override_is_forwarded_verbatim(service):
    service.run({"items": _items(), "options": {"profile_id": SPEAR_PROFILE_ID, "purify": False}})
    request = service._engine.requests[0]["request"]
    assert request["purification"] == {"enabled": False}

    service.run({"items": _items(), "options": {"profile_id": SPEAR_PROFILE_ID}})
    assert "purification" not in service._engine.requests[1]["request"]


def test_profile_listing_reports_purification_degradation(service):
    profiles = {profile["profile_id"]: profile for profile in service.list_profiles(refresh=True)}
    for profile_id in (SPEAR_PROFILE_ID, SPEAR_RETRIEVAL_PROFILE_ID):
        profile = profiles[profile_id]
        assert profile["purification"] is not None
        # 本机没有 55 GB 权重：必须如实报告降级，而不是假装净化可用
        assert profile["purification"]["degraded"] is True
        assert profile["purification"]["effective_backend"] == "rule"
        assert "PURIFIER_DEGRADED_TO_RULE" in profile["warnings"]
        assert profile["capability"]["purification"] is True
        # 对外载荷不含宿主机路径
        assert "model_path" not in profile["purification"]

    # legacy profile 没有净化口径
    assert profiles["bge-large-agglomerative-nr0-legacy-v1"]["purification"] is None


def test_capability_table_advertises_spear_purification():
    capability = _capability_of("spear-v1")
    assert capability["purification"] is True
    assert capability["supports_single_item"] is False
    assert _capability_of("legacy-v1")["purification"] is False


def test_offline_baseline_matches_the_archived_metrics():
    service = ClusteringGatewayService()
    payload = service.load_offline_baseline()
    data = BaselineData(**payload)
    assert data.full_run is True
    assert data.rows["nr0_legacy"].ari == pytest.approx(0.1545)
    # 关键区分：净化关（nr1）与净化开（完整框架）是两行，不能混为一谈
    assert data.rows["nr1_legacy_purification_off"].ari == pytest.approx(0.2754)
    assert data.rows["spear_purified_retrieval_purification_on"].ari == pytest.approx(0.2465)
    assert data.rows["spear_purified_retrieval_purification_on"].ari < data.rows["nr1_legacy_purification_off"].ari
    assert data.comparison is not None
    assert data.comparison["ari"] == pytest.approx(0.092, abs=1e-3)
    # 自由字典的键必须是 camelCase，才能与 shared/clustering.ts 对齐
    assert data.comparison["ariRelative"] == pytest.approx(0.5955, abs=1e-3)
    assert data.source
    # 验收冲突必须写进 notes，避免界面把 0.2465 当成"达标"
    assert any("0.26" in note for note in data.notes)
