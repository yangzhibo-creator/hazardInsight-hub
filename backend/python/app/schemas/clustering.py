"""聚类接口的请求/响应契约。

这是前端 TypeScript 类型（`shared/clustering.ts`）的唯一事实来源，两边字段
必须逐一对齐。统一响应包裹（`success / data / error`）满足「错误可辨识、
不 silent fail」的要求。

字段命名：Python 内部一律 snake_case，对外（HTTP 报文）通过 `alias_generator`
统一序列化为 camelCase，与前端 TypeScript 习惯一致。`populate_by_name=True`
让服务端既能用字段名构造，也能同时接受前端传来的 camelCase 请求体。
注意别名只作用于「已声明的模型字段」，不会递归改写 `metadata` / `features`
这类自由字典里的业务键名。
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")

#: 对外 camelCase、对内 snake_case 的基础配置。
CAMEL_CONFIG = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ApiModel(BaseModel):
    """所有契约模型的基类：统一 camelCase 别名策略。"""

    model_config = CAMEL_CONFIG


class DatasetItem(ApiModel):
    """一条待聚类的样本。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    id: str = Field(min_length=1, description="样本唯一 ID")
    text: str = Field(min_length=1, description="参与聚类的文本")
    metadata: dict[str, Any] = Field(default_factory=dict, description="随样本展示的业务字段")


class RunOptions(ApiModel):
    """聚类执行选项。

    只暴露网关真正支持的开关：算法选择由 cluster-engine 的 profile 决定，
    因此这里不虚构 n_clusters / distance_threshold 之类的参数。
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    profile_id: str | None = Field(default=None, description="指定 profile；留空则自动挑选")
    algorithm: str | None = Field(default=None, description="按算法名自动挑选可用 profile")
    visualize: bool = Field(default=True, description="是否计算二维可视化坐标")
    reduce_method: Literal["pca", "none"] = Field(default="pca", description="二维降维方式")
    purify: bool | None = Field(
        default=None,
        description="spear-v1 专用：true/false 覆盖 profile 的净化开关；null 表示按 profile 执行",
    )
    knowledge_base_id: str | None = Field(
        default=None,
        description="检索增强 profile 专用：覆盖 profile 默认知识库，改用指定的偏差数据库；null 表示按 profile 执行",
    )
    control_purify: bool = Field(
        default=False,
        description=(
            "spear-v1 专用：同一次提交额外跑一遍「净化关」作为对照，配合数据集类别标签计算 6 项外部指标与差值；"
            "其它 profile 没有净化步骤，会被跳过并在 warnings 里说明"
        ),
    )


class PurificationStatusInfo(ApiModel):
    """净化依赖的可用性快照（不含文件系统路径）。"""

    enabled: bool
    requested_backend: str
    effective_backend: str
    degraded: bool
    reason: str | None = None
    guarded: bool = True


class PurificationSampleInfo(ApiModel):
    """一条"净化前 → 净化后"对照。"""

    id: str
    raw: str
    condensed: str


class PurificationReportInfo(ApiModel):
    """本次执行里阶段 1（语义净化）的真实状态。"""

    enabled: bool
    backend: str
    requested_backend: str
    degraded: bool
    reason: str | None = None
    guarded: bool = False
    guard_hits: int = 0
    total: int = 0
    elapsed_ms: float = 0.0
    samples: list[PurificationSampleInfo] = Field(default_factory=list)


class ClusteringRunRequest(ApiModel):
    """聚类请求体。

    下限放宽到 1 条：``semantic-v1`` 允许单条（它自己会判 ``singleton``），
    而 ``legacy-v1`` 仍要求 ≥2 条。这条版本相关的准入规则由网关在解析出
    profile 之后执行（见 ``ClusteringGatewayService.run``），pydantic
    只挡住「空列表」这种无论哪个版本都不可能成立的情况。
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    items: list[DatasetItem] = Field(min_length=1, description="样本列表，至少 1 条")
    options: RunOptions = Field(default_factory=RunOptions)


class Capability(ApiModel):
    """Strategy 能力描述（legacy 与 semantic 的准入规则不同）。"""

    implementation_version: str
    supports_single_item: bool = False
    supports_cache_only: bool = False
    calibration_version: str | None = None
    purification: bool = Field(default=False, description="该实现版本是否带输入层语义净化（spear-v1）")


class ProfileInfo(ApiModel):
    """一个可用 profile 的描述。"""

    profile_id: str
    algorithm: str
    model_id: str
    knowledge_base_id: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    algorithm_params: dict[str, Any] = Field(default_factory=dict)
    implementation_version: str
    max_samples: int
    available: bool
    unavailable_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    calibration_id: str | None = Field(default=None, description="语义 profile 引用的校准配置 ID")
    capability: Capability | None = None
    purification: PurificationStatusInfo | None = Field(
        default=None, description="spear-v1 的净化依赖状态（软依赖，缺失时降级而非不可用）"
    )


class AlgorithmInfo(ApiModel):
    """一种聚类算法的可用性。"""

    algorithm: str
    available: bool
    backend: str
    implementation_version: str
    max_samples: int
    warnings: list[str] = Field(default_factory=list)
    capability: Capability | None = None


class DatasetPreview(ApiModel):
    """数据文件解析结果。"""

    source_name: str
    total: int
    warnings: list[str] = Field(default_factory=list)
    items: list[DatasetItem] = Field(default_factory=list)
    dataset_id: str | None = Field(
        default=None,
        description="数据集引用 ID：提交作业时可只传它，不必重复传 items",
    )


class ValueCount(ApiModel):
    """元数据取值计数。"""

    value: str
    count: int


class RepresentativeSample(ApiModel):
    """簇的代表样本（离质心最近的若干条）。"""

    id: str
    text: str
    confidence: float | None = None
    distance: float | None = None


class NoiseExample(ApiModel):
    """「未归类」桶里的一条示例文本（噪声 / 无效，不构成语义簇）。"""

    id: str
    text: str
    reason: str | None = None


class ClusterGroup(ApiModel):
    """一个簇的聚合描述。

    `cluster_id == -1` 是「未归类」系统桶：它**不是一个语义簇**——
    没有质心、没有代表文本、`quality_score` 与 `confidence` 恒为 None。
    前端必须按 `is_noise_bucket` 区分，而不是把它当成一个普通簇渲染。

    semantic-v1 扩展字段（`cluster_name` 起）在 legacy 路径下缺席，
    因此全部带默认值，保证旧路径构造对象时不报错。
    """

    cluster_id: int = Field(description="-1 表示未归类（噪声 + 无效）")
    label: str
    size: int
    keywords: list[str] = Field(default_factory=list)
    cohesion: float | None = Field(default=None, description="簇内平均余弦相似度")
    representative_samples: list[RepresentativeSample] = Field(default_factory=list)
    metadata_distribution: dict[str, list[ValueCount]] = Field(default_factory=dict)

    # —— semantic-v1 扩展 ——
    cluster_name: str | None = Field(default=None, description="证据式命名；与 label 同值")
    name_source: str | None = Field(default=None, description="命名来源，如 representative_phrase")
    name_evidence: str | None = Field(default=None, description="命名依据的原文")
    naming_version: str | None = None
    unique_size: int | None = Field(default=None, description="去重后的不同文本条数")
    percentage: float | None = Field(default=None, description="占全部输入样本的百分比 0~100")
    separation: float | None = Field(default=None, description="与最强竞争簇的分离支持分")
    quality_score: float | None = Field(default=None, description="簇级质量分 0~1")
    quality_status: str | None = None
    quality_version: str | None = None
    confidence_version: str | None = None
    distance_metric: str | None = Field(default=None, description="距离口径，semantic-v1 为 cosine")
    representative_texts: list[str] = Field(default_factory=list)
    is_noise_bucket: bool = False
    noise_examples: list[NoiseExample] = Field(default_factory=list)


class ClusterResultItem(ApiModel):
    """单条样本的聚类归属。

    `assignment_status` 比 `cluster_id` 更细：同为 `cluster_id == -1`，
    `noise`（不属任何主题）与 `invalid`（规范化后无有效语义）是两个概念。
    """

    id: str
    text: str
    cluster_id: int
    cluster_label: str
    confidence: float | None = Field(default=None, description="与簇质心的余弦相似度映射到 0~1")
    distance: float | None = Field(default=None, description="到簇质心的距离")
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # —— semantic-v1 扩展 ——
    assignment_status: str | None = Field(
        default=None,
        description="core / borderline / small_coherent / duplicate_only / noise / invalid",
    )
    noise_reason: str | None = None
    confidence_version: str | None = None
    distance_metric: str | None = None
    # —— spear-v1 扩展：同一行的浓缩文本，用于"净化前后对照" ——
    purified_text: str | None = Field(default=None, description="净化后的浓缩文本（spear-v1 才有）")


class VisualizationPoint(ApiModel):
    """二维散点坐标（仅用于可视化，不参与聚类计算）。"""

    id: str
    x: float
    y: float
    cluster_id: int


class ClusterSummary(ApiModel):
    """聚类统计。

    前 16 个字段是 legacy / semantic 两条路径共有的兼容字段；
    之后的 semantic-v1 扩展字段在 legacy 路径下保持默认值（None/0/False），
    因此旧调用方与旧前端无需改动。
    """

    run_id: str
    total_samples: int
    cluster_count: int
    noise_count: int
    largest_cluster_size: int
    smallest_cluster_size: int
    avg_cluster_size: float
    algorithm: str
    profile_id: str
    model_id: str
    implementation_version: str
    embedding_dimension: int
    cache_hit: bool = Field(description="本次执行是否需要新的向量推理")
    elapsed_ms: int = Field(description="cluster-engine 报告的算法耗时（毫秒）")
    gateway_ms: int = Field(description="网关总耗时（含可视化加工，毫秒）")
    warnings: list[str] = Field(default_factory=list)

    # —— semantic-v1 扩展 ——
    invalid_count: int = 0
    duplicate_count: int = 0
    unique_count: int = 0
    coverage: float | None = None
    noise_ratio: float | None = None
    selected_k: int | None = Field(default=None, description="Auto-K 选出的 K（后处理前）")
    final_k: int | None = Field(default=None, description="后处理后的最终簇数")
    auto_k_status: str | None = None
    quality_score: float | None = None
    weighted_quality_score: float | None = None
    overall_quality: float | None = None
    quality_version: str | None = None
    confidence_version: str | None = None
    naming_version: str | None = None
    normalization_version: str | None = None
    calibration_version: str | None = None
    calibration_status: str | None = None
    effective_model_id: str | None = None
    device: str | None = None
    seed: int | None = None
    embedding_cache_hits: int | None = None
    embedding_cache_misses: int | None = None
    embedding_cache_hit_ratio: float | None = None
    cache_only: bool = False
    fallback_reason: str | None = None
    visualization_sample_count: int | None = None
    visualization_total_count: int | None = None
    stage_timings: dict[str, float] | None = None
    diagnostics: dict[str, Any] | None = None
    auto_k: dict[str, Any] | None = None
    reduction: dict[str, Any] | None = None
    postprocess: dict[str, Any] | None = None
    # —— spear-v1 扩展：阶段 1（语义净化）的真实状态与前后对照 ——
    purification: PurificationReportInfo | None = None
    # —— 外部指标：数据集带类别标签时才有；键为 ari/vm/fms/ami/hs/cs/score/nClusters/noiseRatio ——
    metrics: dict[str, Any] | None = Field(
        default=None, description="本次主运行的 6 项外部指标（需要数据集带类别标签）"
    )
    control_metrics: dict[str, Any] | None = Field(
        default=None, description="「净化关」对照运行的指标；未勾选对照时为 null"
    )
    metrics_comparison: dict[str, float] | None = Field(
        default=None, description="对照差值（主运行 − 净化关对照）与 ARI 相对增益"
    )
    ground_truth: dict[str, Any] | None = Field(
        default=None, description="真值来源：命中的字段名与带标签条数"
    )


class ClusteringData(ApiModel):
    """聚类结果主体。"""

    summary: ClusterSummary
    clusters: list[ClusterGroup] = Field(default_factory=list)
    items: list[ClusterResultItem] = Field(default_factory=list)
    visualization: list[VisualizationPoint] = Field(default_factory=list)


# ------------------------------------------------------------------ 异步作业


class JobOptions(RunOptions):
    """作业选项：在同步选项之上加"数据集引用"与"幂等键"。

    `items` 与 `datasetId` 二选一：样本多的时候按引用提交，请求体里只有一个 ID，
    不必把上百 MB 的样本再传一遍。
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    dataset_id: str | None = Field(default=None, description="引用已上传的数据集，免去重复传样本")
    idempotency_key: str | None = Field(
        default=None,
        description="幂等键：同一份请求重复提交会返回同一个作业",
        max_length=128,
    )


class ClusteringJobRequest(ApiModel):
    """作业提交请求。允许 `items` 为空——此时必须给出 `options.datasetId`。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    items: list[DatasetItem] = Field(default_factory=list, description="样本列表")
    options: JobOptions = Field(default_factory=JobOptions)


class JobError(ApiModel):
    """作业失败原因（与 HTTP 错误结构同形，便于前端复用一套展示）。"""

    code: str
    message: str


class ClusteringJobInfo(ApiModel):
    """作业状态快照。轮询接口返回的就是它。"""

    job_id: str
    status: str = Field(description="queued / running / succeeded / failed / cancelled")
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    profile_id: str | None = None
    algorithm: str | None = None
    implementation_version: str | None = None
    dataset_id: str | None = None
    item_count: int = 0
    cancel_requested: bool = False
    hard_cancelled: bool = Field(
        default=False,
        description="取消是否真的落到了执行进程上；False 表示只有协作式取消",
    )
    run_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict, description="明细分片元信息")
    warnings: list[str] = Field(default_factory=list)
    error: JobError | None = None
    terminal: bool = Field(default=False, description="是否已进入终态，前端据此停止轮询")


class JobListData(ApiModel):
    """作业列表。"""

    jobs: list[ClusteringJobInfo] = Field(default_factory=list)


class ClusteringJobResultData(ApiModel):
    """作业结果的摘要部分：统计、簇、抽样坐标与明细元信息。

    刻意**不含 `items`**：明细一律走分页接口。这样"看统计"和"翻明细"
    是两条独立的请求，10 万条结果也不会让首屏卡在下载上。
    """

    job_id: str
    run_id: str
    summary: ClusterSummary
    clusters: list[ClusterGroup] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)
    visualization: list[VisualizationPoint] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ClusterFilterOption(ApiModel):
    """簇筛选项：簇号 + 条数（前端据此渲染下拉，并显示每个簇有多大）。"""

    cluster_id: int
    size: int


class JobFilterOptions(ApiModel):
    """明细分页可用的筛选取值（只读索引得到，与样本量无关）。"""

    job_id: str
    cluster_ids: list[ClusterFilterOption] = Field(default_factory=list)
    item_count: int = 0


class JobItemsData(ApiModel):
    """明细分页。`total` 是筛选后的总数，不是本页条数。"""

    job_id: str
    run_id: str
    total: int
    offset: int
    limit: int
    returned: int
    has_more: bool = False
    cluster_id: int | None = None
    scanned: int = Field(default=0, description="本次筛选实际扫描的条数")
    truncated: bool = Field(
        default=False,
        description="按文本筛选时是否因达到扫描上限而截断；True 表示结果不完整",
    )
    items: list[ClusterResultItem] = Field(default_factory=list)


class DatasetReferenceData(ApiModel):
    """数据集引用的元信息（上传接口返回，供作业提交引用）。"""

    dataset_id: str
    source_name: str
    total: int
    created_at: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DatasetListData(ApiModel):
    """数据集引用列表。"""

    datasets: list[DatasetReferenceData] = Field(default_factory=list)


# ------------------------------------------------------------------ 偏差数据库


class KnowledgeBaseInfo(ApiModel):
    """一个知识库的元信息（列表与详情共用）。

    字段全部来自构建时落盘的 ``manifest.json`` 或目录本身，不做推断：
    拿不到就留空/为 0，由前端展示"未知"，不编造。
    """

    knowledge_base_id: str
    display_name: str
    status: str
    verification: str = "unknown"
    processing: str = "unknown"
    entry_count: int = 0
    dimension: int = 0
    metric: str = ""
    model_fingerprint: str = ""
    model_id: str | None = None
    source_name: str | None = None
    source_sha256: str = ""
    created_at: str | None = None
    size_bytes: int = 0
    has_entries: bool = Field(default=False, description="是否保留了索引文本快照")
    has_corpus: bool = Field(default=False, description="是否保留了上传语料原文快照")


class KnowledgeBaseEntry(ApiModel):
    """知识库里的一条文本（``index`` 为库内序号，从 0 开始）。"""

    index: int
    text: str


class KnowledgeBaseDetailData(KnowledgeBaseInfo):
    """知识库详情：元信息 + 一页条目内容。"""

    entries_source: str = Field(
        default="unavailable",
        description="snapshot / corpus / source / unavailable，说明条目内容从哪里读到的",
    )
    entry_total: int = 0
    offset: int = 0
    limit: int = 50
    entries: list[KnowledgeBaseEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeBaseListData(ApiModel):
    """知识库列表。"""

    knowledge_bases: list[KnowledgeBaseInfo] = Field(default_factory=list)


class KnowledgeBaseBuildJob(ApiModel):
    """知识库生成作业的状态快照。

    ``processed / total`` 是**当前阶段**的进度（净化或编码），不是全局百分比：
    阶段切换时 ``stage`` 变化、``processed`` 归零重计——这比编一个加权百分比
    更诚实，前端也更容易解释。
    """

    job_id: str
    status: str = Field(description="queued / running / succeeded / failed")
    stage: str = Field(default="queued", description="queued / purify / embed / index / done")
    processed: int = 0
    total: int = 0
    message: str | None = None
    knowledge_base_id: str | None = None
    display_name: str | None = None
    mode: str | None = Field(default=None, description="purified / raw")
    model_id: str | None = None
    source_name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    error: JobError | None = None
    warnings: list[str] = Field(default_factory=list)
    terminal: bool = Field(default=False, description="是否已进入终态，前端据此停止轮询")


class KnowledgeBaseBuildJobListData(ApiModel):
    """知识库生成作业列表（新→旧）。"""

    jobs: list[KnowledgeBaseBuildJob] = Field(default_factory=list)


class ErrorInfo(ApiModel):
    """统一错误结构。"""

    code: str
    message: str
    request_id: str | None = None
    detail: str | None = None


class Envelope(ApiModel, Generic[T]):
    """统一响应包裹：`success / data / error` 三件套。"""

    success: bool
    data: T | None = None
    error: ErrorInfo | None = None


class ProfilesData(ApiModel):
    """可用 profile 列表。"""

    profiles: list[ProfileInfo] = Field(default_factory=list)


class AlgorithmsData(ApiModel):
    """可用算法列表。"""

    algorithms: list[AlgorithmInfo] = Field(default_factory=list)


class SampleData(ApiModel):
    """内置示例数据。"""

    source_name: str
    items: list[DatasetItem] = Field(default_factory=list)


class BaselineMetrics(ApiModel):
    """离线基准里一个配置的指标（缺失即 null，不猜）。"""

    ari: float | None = None
    vm: float | None = None
    fms: float | None = None
    ami: float | None = None
    hs: float | None = None
    cs: float | None = None
    n_clusters: int | None = None
    noise_ratio: float | None = None
    score: float | None = None


class BaselineData(ApiModel):
    """离线基准结果：现场实时计算失败时的兜底展示。

    与实时结果的区别必须写在 `source` / `verified_at` / `full_run` 上——
    展示归档数字而不说明出处，等于把演示变成误导。
    """

    source: str
    verified_at: str | None = None
    full_run: bool = True
    full_run_item_count: int | None = Field(
        default=None,
        description="归档所指的全量口径样本数；用于识别整份测试集的历史运行（如 20198 条）",
    )
    rows: dict[str, BaselineMetrics] = Field(default_factory=dict)
    baseline_key: str | None = Field(
        default=None, description="差值比较的基线行键；前端据此从 rows 里取对比双方"
    )
    target_key: str | None = Field(default=None, description="差值比较的目标行键")
    comparison: dict[str, float] | None = None
    table: str | None = None
    notes: list[str] = Field(default_factory=list)


#: 聚类接口的响应类型别名，方便在路由里直接引用。
ClusteringEnvelope = Envelope[ClusteringData]
ProfilesEnvelope = Envelope[ProfilesData]
AlgorithmsEnvelope = Envelope[AlgorithmsData]
SampleEnvelope = Envelope[SampleData]
BaselineEnvelope = Envelope[BaselineData]
DatasetEnvelope = Envelope[DatasetPreview]

#: 异步作业相关响应（作业状态 / 结果摘要 / 明细分页 / 筛选取值 / 数据集列表）。
ClusteringJobEnvelope = Envelope[ClusteringJobInfo]
JobListEnvelope = Envelope[JobListData]
ClusteringJobResultEnvelope = Envelope[ClusteringJobResultData]
JobItemsEnvelope = Envelope[JobItemsData]
JobFilterOptionsEnvelope = Envelope[JobFilterOptions]
DatasetListEnvelope = Envelope[DatasetListData]

#: 偏差数据库相关响应（列表 / 详情 / 生成作业）。
KnowledgeBaseListEnvelope = Envelope[KnowledgeBaseListData]
KnowledgeBaseDetailEnvelope = Envelope[KnowledgeBaseDetailData]
KnowledgeBaseBuildJobEnvelope = Envelope[KnowledgeBaseBuildJob]
KnowledgeBaseBuildJobListEnvelope = Envelope[KnowledgeBaseBuildJobListData]


class EngineStatus(ApiModel):
    """cluster-engine 的加载状态。"""

    config_file: str | None = Field(default=None, description="配置文件名（不暴露绝对路径）")
    loaded: bool
    profiles_total: int = 0
    profiles_available: int = 0
    algorithms_available: list[str] = Field(default_factory=list)
    model_id: str | None = None
    model_provider: str | None = None
    model_dimension: int | None = None
    message: str | None = None


class HealthData(ApiModel):
    """健康检查结果。"""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    engine: EngineStatus


class HealthEnvelope(Envelope[HealthData]):
    """健康检查响应。

    既是统一包裹（`success` / `data` / `error`），又在顶层镜像一份 `status`，
    这样探针类工具可以直接读 `body.status`，前端则统一走 `body.data`。
    """

    status: Literal["ok", "degraded"]
