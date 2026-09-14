"""semantic-v1 流水线的工程契约测试。

这里**不**验证聚类质量——质量由真实模型 + 校准配置决定，用替身编码器测不出来，
断言"语义上应该分成 3 类"只会写出一条脆弱的测试。本模块只钉住那些
"无论用哪个模型都必须成立"的性质：

* 计数守恒：每个输入 ID 恰好出现一次，非噪声数 + 噪声数 = 总数；
* 结构退化不加载模型：单条唯一文本走 singleton，`effectiveModelId` 为空；
* 精确去重：相同文本合并成一条参与聚类，但展开回结果时每一份都独立可回溯；
* 无效文本被识别（而不是被当作噪声混在一起），且仍然出现在结果里；
* 文本级缓存：第二次运行零推理、零 miss，并通过 `cacheOnly` 如实上报；
* 可复现：打乱输入顺序不改变"文本 → 簇"的映射（规范编号 + 哈希抽样）；
* 降维状态可核查：PCA 为什么启用/跳过必须写在 `reduction` 里；
* 版本能力表不自说自话：`semantic-v1` 声称支持的算法必须都是注册表里真实存在的。

替身编码器把文本哈希成 8 维向量，因此向量之间没有语义关系——
这正是这些断言只谈工程性质、不谈语义质量的原因。
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import pytest

from retrain_cluster.config import Catalog, Settings
from retrain_cluster.services.clustering import ClusteringService
from tests.conftest import FixedRegistry

ROOT = Path(__file__).resolve().parents[2]

#: 每个用例都用这份"小预算"参数：这些数字只影响算力，不影响被断言的性质。
SEMANTIC_PARAMS = {
    "k_cap": 16,
    "seed": 42,
    "sample_size": 64,
    "validation_size": 16,
    "max_iter": 20,
    "batch_size": 8,
    "n_init": 2,
    "pca_dim": 0,
    "pca_min_samples": 2,
    "min_cluster_unique": 3,
    "split_budget": 2,
    "merge_rounds": 1,
    "refine_rounds": 1,
}


def _build_settings(tmp_path, *, params: dict | None = None) -> Settings:
    """搭一份最小可用的语义配置：8 维替身模型 + 引用真实校准文件的 semantic profile。"""

    (tmp_path / "profiles").mkdir(exist_ok=True)
    (tmp_path / "models.toml").write_text(
        """[[models]]
model_id = "fixture"
provider = "openai_compatible"
source = "fixture"
base_url = "http://127.0.0.1:9/v1"
revision = "fixture-v1"
dimension = 8
""",
        encoding="utf-8",
    )
    (tmp_path / "app.toml").write_text(
        '[app]\nmodels_file="models.toml"\nprofiles_dir="profiles"\n'
        'artifacts_dir="artifacts"\ncalibration_file="semantic-calibration-v1.json"\n'
        "max_samples=1000\nmax_text_length=4000\n",
        encoding="utf-8",
    )
    # 直接复用仓库里的真实校准文件：测试不应该在别处再抄一份阈值，
    # 否则改了校准而测试仍在验证旧数字，两边会悄悄漂移。
    shutil.copy(ROOT / "configs" / "semantic-calibration-v1.json", tmp_path / "semantic-calibration-v1.json")
    profile = {
        "profile_id": "fixture-semantic",
        "algorithm": "semantic_auto_kmeans",
        "model_id": "fixture",
        "algorithm_params": {**SEMANTIC_PARAMS, **(params or {})},
        "features": {"version": "semantic-v1", "n_results": 0, "pca_dim": 0, "reduction": "PCA", "beta": 1.0},
        "knowledge_base_id": None,
        "implementation_version": "semantic-v1",
        "backend": "python",
        "compatibility_status": "semantic",
        "calibration_id": "semantic-calibration-v1",
        "max_samples": 1000,
    }
    (tmp_path / "profiles" / "fixture-semantic.json").write_text(json.dumps(profile), encoding="utf-8")
    return Settings.load(tmp_path / "app.toml")


@pytest.fixture
def semantic_service(tmp_path):
    settings = _build_settings(tmp_path)
    return ClusteringService(settings, encoders=FixedRegistry())


def run(service, items, **params):
    return service.cluster(
        {"profile_id": "fixture-semantic", "items": items},
        enforce_api_limits=True,
    )


def summarize(result):
    return result["semantic"]["summary"]


def test_catalog_accepts_semantic_profile_and_resolves_calibration(tmp_path):
    """Catalog 必须把 calibration_id 解析成真实的校准对象（否则运行时会 422）。"""

    catalog = Catalog(_build_settings(tmp_path))
    profile = catalog.profile("fixture-semantic")
    assert profile.implementation_version == "semantic-v1"
    calibration = catalog.calibration(profile.calibration_id)
    assert calibration is not None
    assert calibration.version == "semantic-calibration-v1"
    assert 0.0 < calibration.thresholds()["t_sem"] < 1.0


def test_single_item_is_singleton_and_does_not_load_a_model(semantic_service):
    """单条唯一文本走 singleton：不加载模型、不算质量分，但结果结构完整。"""

    result = run(semantic_service, [{"id": "only", "text": "配电箱门未上锁", "metadata": {}}])
    summary = summarize(result)
    assert summary["auto_k_status"] == "singleton"
    assert summary["cluster_count"] == 1
    assert summary["noise_count"] == 0
    assert summary["unique_count"] == 1
    # 结构退化时模型完全不该被触碰，指纹也如实留空
    assert summary["cache_only"] is True
    assert summary["effective_model_id"] is None
    assert summary["model_fingerprint"] is None
    assert summary["embedding_cache_hits"] == 0 and summary["embedding_cache_misses"] == 0
    assert [item["id"] for item in result["assignments"]] == ["only"]
    assert "STRUCTURAL_DEGENERATE_NO_MODEL" in result["warnings"]


def test_duplicate_texts_are_merged_then_expanded_back(semantic_service):
    """重复文本只参与一次聚类，但展开回结果时每一份都必须是独立可回溯的记录。"""

    items = [
        {"id": "a", "text": "脚手架连墙件缺失", "metadata": {"area": "A"}},
        {"id": "b", "text": "脚手架连墙件缺失", "metadata": {"area": "B"}},
        {"id": "c", "text": "配电箱门未上锁", "metadata": {"area": "A"}},
    ]
    result = run(semantic_service, items)
    summary = summarize(result)
    assert summary["total_samples"] == 3
    assert summary["duplicate_count"] == 1
    assert summary["unique_count"] == 2  # 两条不同文本

    by_id = {item["id"]: item for item in result["semantic"]["items"]}
    assert set(by_id) == {"a", "b", "c"}
    # 同文本的两份必须落在同一个簇里，且都保留各自的业务字段
    assert by_id["a"]["cluster_id"] == by_id["b"]["cluster_id"]
    assert by_id["b"]["metadata"] == {"area": "B"}
    assert "EXACT_DUPLICATES_MERGED" in result["warnings"]


def test_blank_and_punctuation_only_texts_are_invalid_not_noise(semantic_service):
    """空白/纯标点必须标成 invalid（而不是混进噪声），且仍然出现在结果里。"""

    items = [
        {"id": "ok1", "text": "脚手架连墙件缺失"},
        {"id": "ok2", "text": "脚手架连墙件被拆除"},
        {"id": "blank", "text": "   "},
        {"id": "punct", "text": "，，，。"},
    ]
    result = run(semantic_service, items)
    summary = summarize(result)
    assert summary["invalid_count"] == 2
    assert summary["unique_count"] == 2
    assert "INVALID_TEXT_EXCLUDED" in result["warnings"]

    by_id = {item["id"]: item for item in result["semantic"]["items"]}
    assert by_id["blank"]["assignment_status"] == "invalid"
    assert by_id["punct"]["assignment_status"] == "invalid"
    assert by_id["blank"]["cluster_id"] == -1
    assert by_id["blank"]["noise_reason"] == "blank"


def test_counts_conserve_and_every_id_appears_exactly_once(semantic_service):
    """守恒律：逐条结果覆盖全部输入 ID、无重复，且非噪声 + 噪声 = 总数。"""

    items = [{"id": f"i{index}", "text": f"第 {index} 条隐患描述文本"} for index in range(9)]
    items.append({"id": "dup", "text": "第 0 条隐患描述文本"})
    items.append({"id": "blank", "text": " "})
    result = run(semantic_service, items)
    summary = summarize(result)

    ids = [item["id"] for item in result["semantic"]["items"]]
    assert len(ids) == len(set(ids)) == len(items)
    assert set(ids) == {item["id"] for item in items}

    # 「未归类」桶的 size 必须等于 summary 的 noiseCount：两处口径不一致会让
    # 前端表格与统计卡片对不上，是历史上最容易悄悄漂移的地方。
    buckets = {cluster["cluster_id"]: cluster for cluster in result["semantic"]["clusters"]}
    retained = sum(cluster["size"] for cluster in buckets.values() if cluster["cluster_id"] >= 0)
    assert buckets[-1]["size"] == summary["noise_count"]
    assert retained + summary["noise_count"] == summary["total_samples"]
    # 噪声桶里既包含"不属任何主题"的样本，也包含无效文本，但两者在明细里可区分
    bucket_statuses = {
        item["assignment_status"]
        for item in result["semantic"]["items"]
        if item["cluster_id"] == -1
    }
    assert bucket_statuses <= {"noise", "invalid"}
    assert buckets[-1]["is_noise_bucket"] is True
    assert buckets[-1]["quality_score"] is None


def test_second_run_is_served_entirely_from_the_text_cache(semantic_service):
    """第二次运行必须零推理、零 miss，并通过 cacheOnly / fallbackReason 如实上报。"""

    items = [{"id": f"i{index}", "text": f"第 {index} 条隐患描述文本"} for index in range(9)]
    first = run(semantic_service, items)
    second = run(semantic_service, items)

    assert summarize(first)["embedding_cache_misses"] == summarize(first)["unique_count"]
    assert summarize(second)["embedding_cache_misses"] == 0
    assert summarize(second)["embedding_cache_hits"] == summarize(second)["unique_count"]
    assert summarize(second)["cache_only"] is True
    assert summarize(second)["embedding_cache_hit_ratio"] == 1.0
    assert "EMBEDDING_CACHE_ONLY" in second["warnings"]


def test_shuffling_input_does_not_change_the_text_to_cluster_mapping(semantic_service):
    """可复现性：输入顺序变化不得改变"哪条文本属于哪个簇"。"""

    items = [{"id": f"i{index}", "text": f"第 {index} 条隐患描述文本"} for index in range(9)]
    straight = run(semantic_service, items)
    reversed_run = run(semantic_service, list(reversed(items)))

    def mapping(result):
        # 按文本比对：ID 会变（这里不变），但真正的契约是"文本 → 簇号"稳定
        by_text = {}
        for item in result["semantic"]["items"]:
            by_text.setdefault(item["text"], set()).add(item["cluster_id"])
        return {text: sorted(labels) for text, labels in by_text.items()}

    assert mapping(straight) == mapping(reversed_run)
    assert summarize(straight)["final_k"] == summarize(reversed_run)["final_k"]


def test_pca_reduction_is_recorded_and_used_for_fitting(tmp_path):
    """启用 PCA 时必须在 reduction 里留下可核查的记录（而不是静默返回原空间）。"""

    settings = _build_settings(tmp_path, params={"pca_dim": 4, "pca_min_samples": 4})
    service = ClusteringService(settings, encoders=FixedRegistry())
    items = [{"id": f"i{index}", "text": f"第 {index} 条隐患描述文本"} for index in range(12)]
    summary = summarize(run(service, items))

    reduction = summary["reduction"]
    assert reduction["status"] == "applied"
    assert reduction["source_dim"] == 8  # 原始模型维度（X_sem 的维度）
    assert reduction["fit_dim"] == 4  # 拟合空间维度（X_fit 的维度）
    assert reduction["fit_samples"] >= 4


def test_low_thresholds_produce_a_stable_quality_report(semantic_service):
    """质量分与命名要么给出带版本号的数值，要么明确说明为什么没有——不能沉默。"""

    items = [{"id": f"i{index}", "text": f"第 {index} 条隐患描述文本"} for index in range(9)]
    summary = summarize(run(semantic_service, items))

    assert summary["quality_version"] == "semantic-quality-v1"
    assert summary["confidence_version"] == "semantic-confidence-v1"
    assert summary["naming_version"] == "semantic-naming-v1"
    assert summary["normalization_version"] == "zh-shorttext-v1"
    if summary["cluster_count"]:
        assert summary["quality_score"] is not None
    else:
        # 没有存活簇时质量分必须为 null，而不是 0（0 会被误读成"质量极差"）
        assert summary["quality_score"] is None
        assert summary["overall_quality"] is None


def test_capability_table_only_claims_registered_algorithms():
    """能力表是按实现版本给调用方看的承诺，不能出现"声称支持但没实现"的算法。

    网关的 `_capability_of()` 与 `/algorithms` 的实现版本判定都读这张表：
    一旦它比算法注册表多出一个名字，前端会把这个算法当成可选项放出去，
    单条准入也会按语义规则放行，错误要等运行时才暴露。
    """

    from retrain_cluster.clustering.registry import ALGORITHMS, SEMANTIC_ALGORITHMS as REGISTERED
    from retrain_cluster.services.strategies import (
        LEGACY_VERSION,
        SEMANTIC_ALGORITHMS,
        SEMANTIC_VERSION,
        SPEAR_ALGORITHMS,
        SPEAR_VERSION,
        list_strategy_versions,
        strategy_for_version,
    )

    assert SEMANTIC_ALGORITHMS == REGISTERED
    assert set(SEMANTIC_ALGORITHMS) <= set(ALGORITHMS)
    # spear-v1 复用 legacy 的算法集合，同样不能声称支持注册表里没有的算法
    assert set(SPEAR_ALGORITHMS) <= set(ALGORITHMS)
    assert not set(SPEAR_ALGORITHMS) & set(SEMANTIC_ALGORITHMS)

    versions = {entry["implementation_version"]: entry for entry in list_strategy_versions()}
    assert set(versions) == {LEGACY_VERSION, SEMANTIC_VERSION, SPEAR_VERSION}
    assert list(versions[SEMANTIC_VERSION]["algorithms"]) == list(SEMANTIC_ALGORITHMS)
    assert list(versions[SPEAR_VERSION]["algorithms"]) == list(SPEAR_ALGORITHMS)
    # legacy 用 None 表示"沿用算法注册表的 API 列表"，不能在这里复制一份算法名
    assert versions[LEGACY_VERSION]["algorithms"] is None

    # 分派按版本号查表：未知版本必须报错，不能悄悄落回 legacy 分支
    assert strategy_for_version(LEGACY_VERSION) is None
    assert strategy_for_version(SEMANTIC_VERSION) == "semantic"
    assert strategy_for_version(SPEAR_VERSION) == "spear"
    with pytest.raises(Exception) as excinfo:
        strategy_for_version("semantic-v9")
    assert getattr(excinfo.value, "code", None) == "INVALID_PROFILE"

