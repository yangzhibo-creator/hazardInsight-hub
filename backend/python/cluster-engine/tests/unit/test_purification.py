"""SPEAR 阶段 1（语义净化）的工程契约测试。

这里只测**可判定的性质**，不测"净化质量"：

* 规则净化必须只删不改（构造上不可能反转语义）；
* 极性护栏必须拦下"把 ``不一致`` 改成 ``一致``"的假净化器并兜底修复（验收标准 3）；
* 降级链必须"失败但不抛异常"，并且如实报告降级原因（交付物 A2.5）；
* 缓存与实例复用必须可观察（陷阱 P3：Qwen 只能装一份）。

真实 Qwen 的用例放在 ``-m integration`` 下，默认不跑。
"""

from __future__ import annotations

import json

import pytest

from retrain_cluster.errors import ClusterError
from retrain_cluster.purification import (
    CachedPurifier,
    GuardedPurifier,
    IdentityPurifier,
    PurifierRegistry,
    QwenPurifier,
    RulePurifier,
    polarity_preserved,
    probe,
)
from retrain_cluster.types import PurificationSpec


def spec(**overrides) -> PurificationSpec:
    payload = {"enabled": True, "backend": "rule", "guard": True}
    payload.update(overrides)
    return PurificationSpec(**payload)


# --------------------------------------------------------------------- 规则净化
def test_rule_purifier_strips_noise_but_keeps_polarity():
    text = "2025.11.03 在工艺评定车间巡检发现焊机二维码上名称与系统不一致，已要求更换二维码标签"
    condensed = RulePurifier().purify_one(text)
    assert "不一致" in condensed  # 极性词必须原样保留
    assert "2025.11.03" not in condensed
    assert "已要求更换" not in condensed
    assert condensed and len(condensed) <= len(text)


def test_rule_purifier_never_returns_empty():
    """整条被规则删空时退回截断原文，而不是空串（空文本会让嵌入失去意义）。"""

    assert RulePurifier().purify_one("2025.11.03") != ""


# --------------------------------------------------------------------- 极性护栏
def test_polarity_preserved_requires_a_marker_on_both_sides():
    assert polarity_preserved("名称与系统不一致", "名称与系统不一致")
    assert polarity_preserved("名称与系统一致", "名称一致")  # 源无极性词，不约束
    assert not polarity_preserved("名称与系统不一致", "名称与系统一致")
    assert not polarity_preserved("已用鞋套未及时回收", "已用鞋套回收")


class _ReversingPurifier:
    """故意把否定词吃掉的假净化器——用来证明护栏不是摆设。"""

    name = "reversing-fake"

    def purify(self, texts):
        return [str(text).replace("不一致", "一致").replace("未及时", "及时") for text in texts]


def test_guard_catches_a_semantics_reversing_purifier_and_falls_back():
    guard = GuardedPurifier(_ReversingPurifier(), fallback=RulePurifier())
    source = "2025.11.03 巡检发现焊机二维码上名称与系统不一致，已要求更换二维码标签"
    output = guard.purify_one(source)
    assert "不一致" in output  # 护栏拦下并改用只删不改的规则净化
    assert guard.last_guard_count == 1
    assert guard.last_total == 1


def test_guard_leaves_correct_output_untouched():
    guard = GuardedPurifier(RulePurifier(), fallback=RulePurifier())
    source = "已用鞋套未及时回收"
    assert "未及时" in guard.purify_one(source)
    assert guard.last_guard_count == 0


def test_guard_falls_back_for_the_whole_batch_when_lengths_mismatch():
    class _ShortBatch:
        name = "short-batch"

        def purify(self, texts):
            return ["only-one"]

    guard = GuardedPurifier(_ShortBatch(), fallback=RulePurifier())
    outputs = guard.purify(["未及时回收", "不一致"])
    assert len(outputs) == 2
    assert guard.last_guard_count == 2


# --------------------------------------------------------------------- 预计算缓存
def test_cached_purifier_delegates_misses_and_memoizes(tmp_path):
    path = tmp_path / "purify_map.json"
    path.write_text(json.dumps({"原始": "浓缩"}, ensure_ascii=False), encoding="utf-8")
    cached = CachedPurifier.from_file(path, fallback=RulePurifier())
    outputs = cached.purify(["原始", "2025.11.03 巡检发现名称与系统不一致"])
    assert outputs[0] == "浓缩"
    assert "不一致" in outputs[1]  # 未命中交给规则兜底
    assert cached.last_miss_count == 1
    assert "2025.11.03 巡检发现名称与系统不一致" in cached.mapping  # 回填


def test_cached_purifier_rejects_non_object_payload(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        CachedPurifier.from_file(path)


# --------------------------------------------------------------------- 降级链
def test_registry_degrades_to_rule_when_model_missing():
    """backend=qwen 且权重目录不存在：不抛异常、如实报告降级（交付物 A2.5）。"""

    registry = PurifierRegistry()
    missing = spec(backend="qwen", model_path="/nonexistent/qwen", guard=False)
    purifier, status = registry.get(missing)
    assert isinstance(purifier, RulePurifier)
    assert status.effective_backend == "rule"
    assert status.degraded is True
    assert status.reason is not None
    assert status.warning == "PURIFIER_DEGRADED_TO_RULE"
    # 同一个配置再次取用必须复用同一实例（陷阱 P3：Qwen 只能装一份）
    assert registry.get(missing)[0] is purifier


def test_registry_disabled_backend_is_an_identity_transform():
    registry = PurifierRegistry()
    purifier, status = registry.get(spec(enabled=False))
    assert isinstance(purifier, IdentityPurifier)
    assert purifier.purify_one("原样") == "原样"
    assert status.effective_backend == "identity"
    assert status.enabled is False


def test_probe_reports_cache_hit_and_missing_cache_without_side_effects(tmp_path):
    cache_file = tmp_path / "purify_map.json"
    cache_file.write_text("{}", encoding="utf-8")
    assert probe(spec(backend="cache", cache_file=str(cache_file))).effective_backend == "cache"
    missing = probe(spec(backend="cache", cache_file=str(tmp_path / "none.json")))
    assert missing.effective_backend == "rule" and missing.degraded


def test_purification_spec_rejects_unknown_backend_and_missing_dependencies():
    with pytest.raises(ClusterError, match="Unknown purification backend"):
        PurificationSpec(backend="magic")
    with pytest.raises(ClusterError, match="requires a model path"):
        PurificationSpec(backend="qwen")
    with pytest.raises(ClusterError, match="requires a cache file"):
        PurificationSpec(backend="cache")
    with pytest.raises(ClusterError, match="positive integer"):
        PurificationSpec(batch_size=0)


def test_qwen_purifier_reports_missing_weights_without_importing_torch():
    with pytest.raises(FileNotFoundError):
        QwenPurifier("/nonexistent/qwen").load()
