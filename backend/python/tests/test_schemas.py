"""接口契约测试。

重点验证三件事：

1. 前端 camelCase 与后端 snake_case 的双向兼容（`alias_generator` + `populate_by_name`）；
2. 请求体的强校验（样本数下限、拒绝未知字段）；
3. 自由字典（`metadata` / `features`）里的业务键名不会被别名机制改写
   —— 这是选 camelCase 别名方案而不是"递归改键"方案的关键原因。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.clustering import (
    ClusterResultItem,
    ClusterSummary,
    ClusteringRunRequest,
    DatasetItem,
    EngineStatus,
    HealthData,
    HealthEnvelope,
    ProfileInfo,
    ProfilesData,
    ProfilesEnvelope,
    PurificationReportInfo,
    PurificationStatusInfo,
    RunOptions,
)


def test_request_accepts_camel_case_payload():
    """前端按 camelCase 发送时能被正确解析。"""

    payload = {
        "items": [
            {"id": "a", "text": "脚手架缺少连墙件", "metadata": {"隐患级别": "A"}},
            {"id": "b", "text": "配电箱未上锁"},
        ],
        "options": {"profileId": "agglo-nr0", "visualize": False, "reduceMethod": "none"},
    }

    request = ClusteringRunRequest.model_validate(payload)

    assert request.options.profile_id == "agglo-nr0"
    assert request.options.reduce_method == "none"
    assert request.options.visualize is False
    assert request.items[0].metadata == {"隐患级别": "A"}


def test_request_accepts_snake_case_payload():
    """Python 侧用字段名构造同样可用，便于服务端与测试复用同一模型。"""

    request = ClusteringRunRequest.model_validate(
        {
            "items": [{"id": "a", "text": "文本一"}, {"id": "b", "text": "文本二"}],
            "options": {"profile_id": "agglo-nr0"},
        }
    )

    assert request.options.profile_id == "agglo-nr0"


def test_request_rejects_unknown_field():
    """契约之外的多余字段直接报错，避免前端拼错字段却静默生效。"""

    with pytest.raises(ValidationError):
        DatasetItem.model_validate({"id": "a", "text": "t", "texts": "typo"})


def test_request_requires_at_least_one_item():
    """契约层只挡住空列表。

    「至少 2 条」不再是契约级约束：`semantic-v1` 明确支持单条（返回
    ``autoKStatus=singleton`` 并如实标注低支持），而 `legacy-v1` 仍然要求 ≥2 条。
    版本相关的准入规则由网关在解析出 profile 之后执行
    （`ClusteringGatewayService.run` → `validate_item_count(minimum=...)`），
    因此这里只验证「0 条必拒、1 条可解析」这两件与版本无关的事。
    """

    with pytest.raises(ValidationError):
        ClusteringRunRequest.model_validate({"items": []})

    single = ClusteringRunRequest.model_validate({"items": [{"id": "a", "text": "只有一条"}]})
    assert len(single.items) == 1


def test_item_requires_non_empty_text():
    """空文本无法向量化，必须在契约层拦住。"""

    with pytest.raises(ValidationError):
        DatasetItem.model_validate({"id": "a", "text": ""})


def test_options_defaults():
    """默认开启可视化、使用 PCA，且不指定 profile。"""

    options = RunOptions()
    assert options.visualize is True
    assert options.reduce_method == "pca"
    assert options.profile_id is None
    assert options.algorithm is None


def test_response_serializes_to_camel_case():
    """响应外发字段为 camelCase，与 `shared/clustering.ts` 一一对应。"""

    profile = ProfileInfo(
        profile_id="agglo-nr0",
        algorithm="agglomerative",
        model_id="bge-large-zh-v1.5",
        implementation_version="legacy-v1",
        max_samples=500,
        available=True,
    )
    envelope = ProfilesEnvelope(success=True, data=ProfilesData(profiles=[profile]))

    dumped = envelope.model_dump(by_alias=True)
    dumped_profile = dumped["data"]["profiles"][0]

    assert dumped_profile["profileId"] == "agglo-nr0"
    assert dumped_profile["maxSamples"] == 500
    assert dumped_profile["implementationVersion"] == "legacy-v1"
    assert "profile_id" not in dumped_profile


def test_free_form_dict_keys_are_preserved():
    """业务元数据的键名原样保留，不被别名机制递归改写。"""

    item = DatasetItem(id="a", text="t", metadata={"隐患级别": "A", "area_name": "1 号机组"})

    assert item.model_dump(by_alias=True)["metadata"] == {"隐患级别": "A", "area_name": "1 号机组"}


def test_health_envelope_mirrors_status():
    """健康检查既满足统一包裹，又在顶层镜像 status 供探针直接读取。"""

    data = HealthData(
        status="ok",
        service="HazardInsight Clustering Gateway",
        version="1.0.0",
        engine=EngineStatus(loaded=True, profiles_total=20, profiles_available=20),
    )
    envelope = HealthEnvelope(success=True, data=data, error=None, status=data.status)

    dumped = envelope.model_dump(by_alias=True)
    assert dumped["status"] == "ok"
    assert dumped["data"]["engine"]["profilesAvailable"] == 20
    assert dumped["data"]["engine"]["profilesTotal"] == 20


def test_health_envelope_rejects_inconsistent_status():
    """顶层 status 与 data.status 共用同一枚举，拼错即报错。"""

    data = HealthData(status="ok", service="s", version="1", engine=EngineStatus(loaded=True))

    with pytest.raises(ValidationError):
        HealthEnvelope(success=True, data=data, error=None, status="healthy")


def test_purify_override_is_optional_and_typed():
    """净化开关是可选布尔：不传表示"按 profile"，传 false 才是对照实验。"""

    assert RunOptions().purify is None
    assert RunOptions.model_validate({"purify": False}).purify is False
    assert RunOptions.model_validate({"purify": True}).purify is True


def test_purification_contract_round_trips_between_camel_and_snake():
    """新增的净化字段必须能双向往返：前端 camelCase ↔ 后端 snake_case。"""

    profile = ProfileInfo(
        profile_id="spear_purified",
        algorithm="agglomerative",
        model_id="bge-large-zh-v1.5",
        implementation_version="spear-v1",
        max_samples=300000,
        available=True,
        purification=PurificationStatusInfo(
            enabled=True,
            requested_backend="auto",
            effective_backend="rule",
            degraded=True,
            reason="purification_model_missing",
            guarded=True,
        ),
    )
    dumped = profile.model_dump(by_alias=True)
    assert dumped["purification"]["requestedBackend"] == "auto"
    assert dumped["purification"]["effectiveBackend"] == "rule"
    assert "requested_backend" not in dumped["purification"]

    # camelCase 往返回来仍能构造出同一个对象
    restored = ProfileInfo.model_validate(dumped)
    assert restored.purification is not None
    assert restored.purification.effective_backend == "rule"

    summary = ClusterSummary(
        run_id="run_" + "0" * 32,
        total_samples=4,
        cluster_count=2,
        noise_count=0,
        largest_cluster_size=3,
        smallest_cluster_size=1,
        avg_cluster_size=2.0,
        algorithm="agglomerative",
        profile_id="spear_purified",
        model_id="bge-large-zh-v1.5",
        implementation_version="spear-v1",
        embedding_dimension=1024,
        cache_hit=False,
        elapsed_ms=10,
        gateway_ms=12,
        purification=PurificationReportInfo(
            enabled=True,
            backend="rule",
            requested_backend="auto",
            degraded=True,
            reason="purification_model_missing",
            guarded=True,
            guard_hits=1,
            total=4,
            elapsed_ms=0.5,
            samples=[{"id": "a", "raw": "原文", "condensed": "浓缩"}],
        ),
    )
    dumped_summary = summary.model_dump(by_alias=True)
    assert dumped_summary["purification"]["guardHits"] == 1
    assert dumped_summary["purification"]["samples"][0]["condensed"] == "浓缩"

    item = ClusterResultItem(id="a", text="原文", cluster_id=0, cluster_label="簇 0", purified_text="浓缩")
    assert item.model_dump(by_alias=True)["purifiedText"] == "浓缩"
