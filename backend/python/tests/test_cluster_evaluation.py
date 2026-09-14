"""外部指标评估（ARI / VM / FMS / AMI / HS / CS）的网关侧测试。

不加载向量模型：真值提取与差值比较是纯函数；指标计算只依赖 sklearn 与
cluster-engine 的评估器，因此这组用例跑得很快。
"""

from __future__ import annotations

import pytest

from app.services.cluster_evaluation import (
    compare_metrics,
    evaluate_metrics,
    extract_ground_truth,
)
from app.services.clustering_service import ClusteringGatewayService


def _meta(**fields: str) -> dict[str, str]:
    return dict(fields)


# ---------------------------------------------------------------- 真值提取


def test_ground_truth_uses_category_when_every_row_has_it():
    labels, field = extract_ground_truth(
        [_meta(category="焊接设备标识错误"), _meta(category="脚手架搭设缺失")]
    )
    assert field == "category"
    assert labels == ["焊接设备标识错误", "脚手架搭设缺失"]


def test_ground_truth_field_match_is_case_insensitive():
    labels, field = extract_ground_truth([_meta(Category="A"), _meta(Category="B")])
    assert field == "Category"
    assert labels == ["A", "B"]


def test_ground_truth_accepts_chinese_column_names():
    labels, field = extract_ground_truth([_meta(类别="一类"), _meta(类别="二类")])
    assert field == "类别"
    assert labels == ["一类", "二类"]


def test_partial_labels_are_rejected_instead_of_biasing_the_metrics():
    # 只有一半样本带标签时绝不能算：那会得到一个看似可信、实则偏移的 ARI
    labels, field = extract_ground_truth([_meta(category="A"), _meta(category="")])
    assert labels is None
    assert field is None
    assert extract_ground_truth([_meta(category="A"), _meta(other="x")]) == (None, None)


def test_no_metadata_or_no_candidate_field_returns_nothing():
    assert extract_ground_truth([]) == (None, None)
    assert extract_ground_truth([_meta(note="只有备注")]) == (None, None)


# ---------------------------------------------------------------- 指标计算


def test_perfect_partition_scores_one_and_keeps_camel_case_keys():
    metrics = evaluate_metrics(["a", "a", "b", "b"], [0, 0, 1, 1])
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["nClusters"] == 2
    # 键名与离线基准 comparison 对齐，前端才能用同一套渲染逻辑
    assert "n_clusters" not in metrics
    for key in ("vm", "fms", "ami", "hs", "cs"):
        assert key in metrics


def test_single_cluster_returns_sentinel_score_without_other_metrics():
    metrics = evaluate_metrics(["a", "b"], [0, 0])
    assert metrics["nClusters"] == 1
    assert metrics["score"] == pytest.approx(-1.0)
    assert "ari" not in metrics


def test_metrics_are_computed_only_on_non_noise_samples():
    # 噪声(-1)不参与外部指标，但噪声比例要如实记录
    metrics = evaluate_metrics(["a", "a", "b", "b"], [0, 0, 1, -1])
    assert metrics["noiseRatio"] == pytest.approx(0.25)
    assert metrics["ari"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 差值比较


def test_comparison_is_target_minus_base_with_relative_ari():
    base = {"ari": 0.155, "vm": 0.59, "fms": 0.16, "ami": 0.45, "hs": 0.585, "cs": 0.595}
    target = {"ari": 0.281, "vm": 0.685, "fms": 0.285, "ami": 0.588, "hs": 0.669, "cs": 0.702}
    diff = compare_metrics(base, target)
    assert diff["ari"] == pytest.approx(0.126)
    assert diff["fms"] == pytest.approx(0.125)
    assert diff["ami"] == pytest.approx(0.138)
    assert diff["hs"] == pytest.approx(0.084)
    assert diff["cs"] == pytest.approx(0.107)
    assert diff["ariRelative"] == pytest.approx(0.8129, abs=1e-3)


def test_comparison_skips_metrics_missing_on_either_side():
    diff = compare_metrics({"ari": 0.2}, {"ari": 0.3, "vm": 0.5})
    assert diff == {"ari": pytest.approx(0.1), "ariRelative": pytest.approx(0.5)}


# ---------------------------------------------------------------- 归档基准


def test_archived_baseline_ships_all_six_metrics_for_the_two_full_rows():
    from app.services.clustering_service import ClusteringGatewayService as _Service

    payload = _Service().load_offline_baseline()
    assert payload["full_run_item_count"] == 20198
    for key in ("nr0_legacy", "spear_purified_retrieval_purification_on", "paper_report"):
        row = payload["rows"][key]
        for metric in ("ari", "vm", "fms", "ami", "hs", "cs"):
            assert isinstance(row.get(metric), (int, float)), f"{key} 缺少 {metric}"
    # 六项指标齐了，差值比较才可能给出六项增量的完整视图
    comparison = payload["comparison"]
    for metric in ("ari", "vm", "fms", "ami", "hs", "cs"):
        assert metric in comparison
