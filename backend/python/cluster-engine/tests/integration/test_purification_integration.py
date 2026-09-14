"""真模型净化用例（``-m integration``）。

从 ``-m integration`` 的语义出发，这里刻意**不使用硬编码路径**：模型位置从
仓库自己的 ``configs/app.toml`` 里解析（``spear_purified.purification.model_path``），
目录不存在就 skip。否则在没带 55 GB 权重的机器上会得到
``FileNotFoundError: /models/Qwen3.8-27B`` 这种"看似代码坏了、其实是没带权重"的误导。

真正的断言只有一条：**实测模型确实被加载**（``effective_backend == "qwen"``，
而不是悄悄降级成规则净化），并且护栏把已知会反转的样本拦下了。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from retrain_cluster.config import Catalog, Settings
from retrain_cluster.purification import PurifierRegistry

pytestmark = pytest.mark.integration

ENGINE_ROOT = Path(__file__).resolve().parents[2]

#: 论文/实测里已知会被裸 Qwen 反转的样本（陷阱 P1）。
POLARITY_SAMPLE = "2025.11.03 在工艺评定车间巡检发现焊机二维码上名称与系统不一致，已要求更换二维码标签"


def _purification_spec():
    settings = Settings.load(ENGINE_ROOT / "configs" / "app.toml")
    catalog = Catalog(settings)
    profile = catalog.profile("spear_purified")
    spec = profile.purification
    assert spec is not None
    # 允许用环境变量覆盖模型目录（现场与预演机路径不同）
    override = os.environ.get("SPEAR_LLM_MODEL_PATH")
    if override:
        from dataclasses import replace

        spec = replace(spec, model_path=override)
    return spec


def test_qwen_purifier_keeps_polarity_on_the_real_model():
    spec = _purification_spec()
    if not Path(spec.model_path or "").is_dir():
        pytest.skip(f"local purification model is not present: {spec.model_path}")

    registry = PurifierRegistry()
    try:
        purifier, status = registry.get(spec)
        # 权重在就位的前提下，不允许悄悄降级——降级说明加载真的失败了
        assert status.degraded is False, f"purifier degraded unexpectedly: {status.reason}"
        assert status.effective_backend == "qwen"
        assert "不一致" in purifier.purify_one(POLARITY_SAMPLE)
    finally:
        registry.release()
