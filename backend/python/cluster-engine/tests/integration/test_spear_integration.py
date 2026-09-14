"""SPEAR 全链路集成测试（``-m integration``，默认跳过）。

需要现场环境才具备的东西：本地 bge 权重、可选的本地 Qwen 权重、以及
已构建的检索知识库。任何一项缺失都 skip，而不是失败——否则 CI/预演机上
"没带 55 GB 权重"会被误报成代码缺陷。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from retrain_cluster.config import Catalog, Settings
from retrain_cluster.services.clustering import ClusteringService

pytestmark = pytest.mark.integration

ENGINE_ROOT = Path(__file__).resolve().parents[2]


def _settings():
    settings = Settings.load(ENGINE_ROOT / "configs" / "app.toml")
    catalog = Catalog(settings)
    return settings, catalog


def test_configured_embedding_model_is_present_and_complete():
    """本地向量模型必须存在且清单校验通过（否则现场第一步就会挂）。"""

    _, catalog = _settings()
    spec = catalog.model("bge-large-zh-v1.5")
    from retrain_cluster.config import verify_local_model

    verify_local_model(spec)


def test_spear_profiles_load_with_real_configuration():
    _, catalog = _settings()
    for profile_id in ("spear_purified", "spear_purified_retrieval"):
        profile = catalog.profile(profile_id)
        assert profile.purification is not None


def test_end_to_end_purified_run_on_a_small_real_batch():
    """真模型小批量端到端：净化 → 编码 → 聚类，闭环且不触网。"""

    settings, catalog = _settings()
    profile = catalog.profile("spear_purified")
    if not Path(profile.purification.model_path or "").is_dir():
        pytest.skip("local purification model is not present")
    service = ClusteringService(settings)
    items = [
        {"id": f"s{index}", "text": text}
        for index, text in enumerate(
            [
                "2025.11.03 巡检发现焊机二维码上名称与系统不一致，已要求更换二维码标签",
                "2025.10.16 巡查不锈钢车间发现已用鞋套未及时回收。",
                "1C16 段模板龙骨间距过大",
                "11 轴线东侧堆场钢筋拐头与地面接触",
            ]
        )
    ]
    result = service.cluster({"profile_id": "spear_purified", "items": items})
    assert result["n_samples"] == len(items)
    assert result["purification"]["enabled"] is True
    assert len(result["assignments"]) == len(items)
    condensed = {sample["id"]: sample["condensed"] for sample in result["purification"]["samples"]}
    # 极性保真：真模型也不得把否定词吃掉
    assert "不一致" in condensed["s0"]
