/**
 * 聚类分析的共享类型定义。
 *
 * 与后端 Python 网关的 Pydantic 契约（`backend/python/app/schemas/clustering.py`）
 * 逐字段对齐：Python 侧是唯一事实来源，这里只做镜像，不额外发明字段。
 *
 * 统一响应包裹（`success / data / error`）由 `web/api/clustering.ts` 拆封，
 * 页面拿到的都是业务数据本体，不需要关心 `success` 字段。
 */

/** 后端统一错误结构。 */
export interface ClusterApiErrorInfo {
  /** 机器可读错误码，例如 `INVALID_INPUT`、`ENGINE_UNAVAILABLE`。 */
  code: string;
  /** 面向用户的中文错误描述。 */
  message: string;
  /** 请求追踪 ID，便于在后端日志中定位。 */
  request_id?: string | null;
  /** 可选的补充细节。 */
  detail?: string | null;
}

/** 一条待聚类的样本。 */
export interface ClusteringDatasetItem {
  /** 样本唯一 ID。 */
  id: string;
  /** 参与聚类的文本。 */
  text: string;
  /** 随样本展示的业务字段（隐患级别、作业区域等）。 */
  metadata: Record<string, unknown>;
}

/** 二维可视化降维方式。 */
export type ClusterReduceMethod = 'pca' | 'none';

/** 聚类执行选项。算法由后端的 profile 决定，前端不虚构算法超参。 */
export interface ClusteringRunOptions {
  /** 指定 profile；留空则由后端按算法偏好自动挑选。 */
  profileId?: string | null;
  /** 按算法名自动挑选可用 profile。 */
  algorithm?: string | null;
  /** 是否计算二维可视化坐标。 */
  visualize?: boolean;
  /** 二维降维方式。 */
  reduceMethod?: ClusterReduceMethod;
  /**
   * spear-v1 专用：覆盖 profile 的净化开关。
   * `true/false` 用于现场对照实验，`null`（不传）表示完全按 profile 执行。
   */
  purify?: boolean | null;
}

/** 聚类请求体。 */
export interface ClusteringRunRequest {
  items: ClusteringDatasetItem[];
  options: ClusteringRunOptions;
}

/**
 * Strategy 能力描述。
 *
 * `semantic-v1` 与 `legacy-v1` 的准入规则不同（能否单条、能否 cache-only），
 * 前端据此决定是否禁用「至少 2 条」这类校验，不需要自己按算法名猜。
 */
export interface ClusteringCapability {
  implementationVersion: string;
  /** 是否允许只有 1 条样本（semantic-v1 允许）。 */
  supportsSingleItem: boolean;
  /** 是否支持「向量全部命中缓存、无需加载模型」的执行。 */
  supportsCacheOnly: boolean;
  /** 阈值校准版本；legacy 为 null。 */
  calibrationVersion?: string | null;
  /** 该实现版本是否带输入层语义净化（spear-v1）。 */
  purification?: boolean;
}

/** 净化依赖的可用性快照（软依赖：缺失时降级而不是让 profile 不可用）。 */
export interface ClusteringPurificationStatus {
  enabled: boolean;
  requestedBackend: string;
  /** 实际生效的后端：`qwen` / `rule` / `cache` / `identity`。 */
  effectiveBackend: string;
  degraded: boolean;
  reason?: string | null;
  guarded: boolean;
}

/** 一条「净化前 → 净化后」对照。 */
export interface ClusteringPurificationSample {
  id: string;
  raw: string;
  condensed: string;
}

/** 本次执行里阶段 1（语义净化）的真实状态。 */
export interface ClusteringPurificationReport {
  enabled: boolean;
  backend: string;
  requestedBackend: string;
  degraded: boolean;
  reason?: string | null;
  guarded: boolean;
  guardHits: number;
  total: number;
  elapsedMs: number;
  samples: ClusteringPurificationSample[];
}

/** 一个可用 profile 的描述。 */
export interface ClusteringProfileInfo {
  profileId: string;
  algorithm: string;
  modelId: string;
  knowledgeBaseId?: string | null;
  features: Record<string, unknown>;
  algorithmParams: Record<string, unknown>;
  implementationVersion: string;
  maxSamples: number;
  available: boolean;
  /** 不可用原因；`available` 为 true 时通常为空。 */
  unavailableReason?: string | null;
  warnings: string[];
  /** 语义 profile 引用的校准配置 ID（阈值不写在 profile 里）。 */
  calibrationId?: string | null;
  /** 该 profile 所属实现版本的能力。 */
  capability?: ClusteringCapability | null;
  /** spear-v1 的净化依赖状态；其它版本为 null。 */
  purification?: ClusteringPurificationStatus | null;
}

/** 一种聚类算法的可用性。 */
export interface ClusteringAlgorithmInfo {
  algorithm: string;
  available: boolean;
  /** 实现后端，例如 `sklearn`、`hdbscan`、`inhouse`。 */
  backend: string;
  implementationVersion: string;
  maxSamples: number;
  warnings: string[];
  /** 该算法所属实现版本的能力（legacy 与 semantic 的准入规则不同）。 */
  capability?: ClusteringCapability | null;
}

/** 数据文件解析结果。 */
export interface ClusteringDatasetPreview {
  sourceName: string;
  total: number;
  warnings: string[];
  items: ClusteringDatasetItem[];
  /**
   * 数据集引用 ID。提交异步作业时可以只传它，省掉重传整份样本。
   * 老版本后端不返回该字段，因此是可选的。
   */
  datasetId?: string | null;
}

/** 元数据取值计数。 */
export interface ClusteringValueCount {
  value: string;
  count: number;
}

/** 簇的代表样本（离质心最近的若干条）。 */
export interface ClusteringRepresentativeSample {
  id: string;
  text: string;
  /** 与簇质心的余弦相似度映射到 0~1。 */
  confidence?: number | null;
  /** 到簇质心的欧氏距离。 */
  distance?: number | null;
}

/** 未归类桶里的一条示例文本（噪声/无效，不构成语义簇）。 */
export interface ClusteringNoiseExample {
  id: string;
  text: string;
  /** 被判为噪声/无效的原因码。 */
  reason?: string | null;
}

/**
 * 一个簇的聚合描述。
 *
 * `clusterName` 等语义扩展字段只在 `semantic-v1` 路径出现；
 * `legacy-v1` 路径下为 undefined，此时按 `label` 展示即可。
 * 两套字段同时存在是刻意的：旧字段保持兼容，新字段承载新口径。
 */
export interface ClusteringClusterGroup {
  /** `-1` 表示未归类（噪声 + 无效）。 */
  clusterId: number;
  label: string;
  size: number;
  keywords: string[];
  /** 簇内平均余弦相似度。 */
  cohesion?: number | null;
  representativeSamples: ClusteringRepresentativeSample[];
  metadataDistribution: Record<string, ClusteringValueCount[]>;

  // —— semantic-v1 扩展 ——
  /** 证据式命名（与 `label` 同值，保留旧字段镜像）。 */
  clusterName?: string | null;
  /** 命名来源：`representative_phrase` / `keyword_pair` / `fallback_text` / `fallback_id` / `system_bucket`。 */
  nameSource?: string | null;
  /** 命名所依据的原文（命名必须能在真实文本里找到出处）。 */
  nameEvidence?: string | null;
  namingVersion?: string | null;
  /** 去重后的不同文本条数；`size` 是含重复行的原始条数。 */
  uniqueSize?: number | null;
  /** 占全部输入样本的百分比（0~100）。 */
  percentage?: number | null;
  /** 与最强竞争簇的分离支持分（-1~1）。 */
  separation?: number | null;
  /** 簇级质量分 0~1；样本太少时为 null。 */
  qualityScore?: number | null;
  qualityStatus?: string | null;
  qualityVersion?: string | null;
  confidenceVersion?: string | null;
  /** 距离口径，semantic-v1 恒为 `cosine`。 */
  distanceMetric?: string | null;
  /** 代表文本原文（与 `representativeSamples` 对应，便于直接展示）。 */
  representativeTexts?: string[];
  /** 是否是「未归类」系统桶（无质心、无质量分，不是语义簇）。 */
  isNoiseBucket?: boolean;
  /** 「未归类」桶的少量示例。 */
  noiseExamples?: ClusteringNoiseExample[];
}

/**
 * 单条样本的聚类归属。
 *
 * `assignmentStatus` 是比 `clusterId` 更细的成员判定：
 * `core` / `borderline` / `small_coherent` / `duplicate_only` / `noise` / `invalid`。
 */
export interface ClusteringResultItem {
  id: string;
  text: string;
  clusterId: number;
  clusterLabel: string;
  /** 与簇质心的余弦相似度映射到 0~1。 */
  confidence?: number | null;
  /** 到簇质心的距离（semantic-v1 为余弦距离，见 `distanceMetric`）。 */
  distance?: number | null;
  keywords: string[];
  metadata: Record<string, unknown>;

  // —— semantic-v1 扩展 ——
  assignmentStatus?: string | null;
  /** 被判为噪声/无效的原因码。 */
  noiseReason?: string | null;
  confidenceVersion?: string | null;
  distanceMetric?: string | null;

  // —— spear-v1 扩展 ——
  /** 同一行的浓缩文本（引擎只返回前若干条，其余为 null）。 */
  purifiedText?: string | null;
}

/** 二维散点坐标（仅用于可视化，不参与聚类计算）。 */
export interface ClusteringVisualizationPoint {
  id: string;
  x: number;
  y: number;
  clusterId: number;
}

/** 分阶段耗时（毫秒）。键名由引擎定义，前端只做展示。 */
export type ClusteringStageTimings = Record<string, number>;

/**
 * 聚类统计。
 *
 * 前 20 个字段是 legacy / semantic 两条路径共有的「兼容字段」；
 * 之后的 `semantic-v1` 扩展字段在 legacy 路径下为 undefined/null，
 * 页面必须容忍缺失（不要假设永远有值）。
 */
export interface ClusteringSummary {
  runId: string;
  totalSamples: number;
  clusterCount: number;
  noiseCount: number;
  largestClusterSize: number;
  smallestClusterSize: number;
  avgClusterSize: number;
  algorithm: string;
  profileId: string;
  modelId: string;
  implementationVersion: string;
  embeddingDimension: number;
  /** 本次执行是否需要新的向量推理（false 表示全部命中缓存/无需模型）。 */
  cacheHit: boolean;
  /** cluster-engine 报告的算法耗时（毫秒）。 */
  elapsedMs: number;
  /** 网关总耗时（含可视化加工，毫秒）。 */
  gatewayMs: number;
  warnings: string[];

  // —— semantic-v1 扩展（legacy 路径为 null / 0 / false）——
  /** 规范化后被判为无效（空白 / 纯标点）的条数。 */
  invalidCount?: number;
  /** 被精确去重合并掉的重复条数。 */
  duplicateCount?: number;
  /** 去重后的不同文本条数（Auto-K 与拟合的真实规模）。 */
  uniqueCount?: number;
  /** 归入有效簇的样本占比 0~1。 */
  coverage?: number | null;
  noiseRatio?: number | null;
  /** Auto-K 选出的 K（后处理前）。 */
  selectedK?: number | null;
  /** 后处理后的最终簇数。 */
  finalK?: number | null;
  /** Auto-K 状态：`ok` / `single_cluster` / `pair` / `singleton` / `no_coherent_topics`。 */
  autoKStatus?: string | null;
  /** 簇级质量分的宏平均 0~1。 */
  qualityScore?: number | null;
  /** 按簇大小加权的质量分。 */
  weightedQualityScore?: number | null;
  /** 质量分 × 覆盖率（把「未归类」也算进去的综合分）。 */
  overallQuality?: number | null;
  qualityVersion?: string | null;
  confidenceVersion?: string | null;
  namingVersion?: string | null;
  normalizationVersion?: string | null;
  /** 阈值校准版本与状态（`experimental` 表示尚未经人工校准验证）。 */
  calibrationVersion?: string | null;
  calibrationStatus?: string | null;
  /** 实际使用的向量模型；结构退化（只有 1 条唯一文本）时为 null。 */
  effectiveModelId?: string | null;
  device?: string | null;
  seed?: number | null;
  embeddingCacheHits?: number | null;
  embeddingCacheMisses?: number | null;
  embeddingCacheHitRatio?: number | null;
  /** true 表示未加载模型即完成（全部向量命中缓存或结构退化）。 */
  cacheOnly?: boolean;
  /** 有界回退原因，例如 `gpu_oom_cpu_fallback`。 */
  fallbackReason?: string | null;
  /** 二维散点实际返回的点数 / 参与聚类的唯一文本数。 */
  visualizationSampleCount?: number | null;
  visualizationTotalCount?: number | null;
  stageTimings?: ClusteringStageTimings | null;
  /** 在线诊断（质量、噪声、抽样轮廓等）。 */
  diagnostics?: Record<string, unknown> | null;
  /** Auto-K 全过程诊断：候选分数、被拒原因、选择理由、稳定性挑战。 */
  autoK?: Record<string, unknown> | null;
  /** 降维状态：是否启用、为何跳过/回退、拟合样本量。 */
  reduction?: Record<string, unknown> | null;
  /** 后处理日志：拆分 / 合并 / 小簇保护 / 规范编号。 */
  postprocess?: Record<string, unknown> | null;

  // —— spear-v1 扩展 ——
  /** 阶段 1（语义净化）的真实状态与前后对照；legacy/semantic 路径为 null。 */
  purification?: ClusteringPurificationReport | null;
}

/** 聚类结果主体。 */
export interface ClusteringData {
  summary: ClusteringSummary;
  clusters: ClusteringClusterGroup[];
  items: ClusteringResultItem[];
  visualization: ClusteringVisualizationPoint[];
}

/** 内置示例数据。 */
export interface ClusteringSampleData {
  sourceName: string;
  items: ClusteringDatasetItem[];
}

/** cluster-engine 的加载状态。 */
export interface ClusteringEngineStatus {
  /** 配置文件名（不下发绝对路径）。 */
  configFile?: string | null;
  loaded: boolean;
  profilesTotal: number;
  profilesAvailable: number;
  algorithmsAvailable: string[];
  modelId?: string | null;
  modelProvider?: string | null;
  modelDimension?: number | null;
  /** 未加载时的说明信息。 */
  message?: string | null;
}

/** 健康检查结果。 */
export interface ClusteringHealthData {
  status: 'ok' | 'degraded';
  service: string;
  version: string;
  engine: ClusteringEngineStatus;
}

/* ------------------------------------------------------------------ 离线基准 */

/** 一组已归档的指标（离线基准里的一个配置）。 */
export interface ClusteringBaselineMetrics {
  ari?: number | null;
  vm?: number | null;
  fms?: number | null;
  ami?: number | null;
  hs?: number | null;
  cs?: number | null;
  nClusters?: number | null;
  noiseRatio?: number | null;
  score?: number | null;
}

/**
 * 离线基准结果：现场实时计算失败或时间不够时，用它保证演示不中断。
 *
 * 刻意带 `source` 与 `verifiedAt`：展示已归档数字时必须写明出处，
 * 不能与"本次实时计算的结果"混为一谈。
 */
export interface ClusteringOfflineBaseline {
  /** 这批数字的来源说明（数据集口径、机器、论文出处）。 */
  source: string;
  /** 记录时间（ISO 字符串）。 */
  verifiedAt?: string | null;
  /** 是否是可复现的全量口径（false 表示抽样，指标不具可比性）。 */
  fullRun: boolean;
  /** 配置名（如 `nr0` / `spear_purified_retrieval`）→ 指标。 */
  rows: Record<string, ClusteringBaselineMetrics>;
  /** 相对基线的增益（由后端算好，前端不重复造口径）。 */
  comparison?: Record<string, number> | null;
  /** 可直接展示的文本表格。 */
  table?: string | null;
  notes?: string[];
}

/* ------------------------------------------------------------------ 异步作业 */

/**
 * 作业状态。`queued → running → succeeded | failed | cancelled`，终态不再变化。
 *
 * 「取消」是独立状态而不是 failed 的一种：失败要人查原因，取消是调用方的意图。
 */
export type ClusteringJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/** 作业失败原因（与 HTTP 错误结构同形，前端可复用一套展示）。 */
export interface ClusteringJobError {
  code: string;
  message: string;
}

/** 明细分片的元信息。 */
export interface ClusteringJobDetail {
  runId?: string;
  /** 明细总条数（与作业的样本数一致）。 */
  itemCount?: number;
  shardCount?: number;
  /** 每个分片的条数——前端不必知道内部布局，仅用于诊断展示。 */
  shardSize?: number;
}

/** 作业状态快照（轮询接口返回的就是它）。 */
export interface ClusteringJobInfo {
  jobId: string;
  status: ClusteringJobStatus;
  createdAt?: string | null;
  updatedAt?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  profileId?: string | null;
  algorithm?: string | null;
  implementationVersion?: string | null;
  datasetId?: string | null;
  itemCount: number;
  cancelRequested: boolean;
  /** 取消是否真的落到了执行进程上；false 表示只有协作式取消。 */
  hardCancelled: boolean;
  runId?: string | null;
  detail: ClusteringJobDetail;
  warnings: string[];
  error?: ClusteringJobError | null;
  /** 是否已进入终态，前端据此停止轮询。 */
  terminal: boolean;
}

/**
 * 作业结果摘要。
 *
 * 刻意**不含 `items`**：明细一律走分页接口。这样"看统计"和"翻明细"是两条
 * 独立请求，10 万条结果也不会让首屏卡在下载上。
 */
export interface ClusteringJobResultData {
  jobId: string;
  runId: string;
  summary: ClusteringSummary;
  clusters: ClusteringClusterGroup[];
  aggregate: Record<string, unknown>;
  visualization: ClusteringVisualizationPoint[];
  detail: ClusteringJobDetail;
  warnings: string[];
}

/** 簇筛选项：簇号 + 条数。 */
export interface ClusteringClusterFilterOption {
  clusterId: number;
  size: number;
}

/** 明细分页可用的筛选取值（只读索引得到，与样本量无关）。 */
export interface ClusteringJobFilterOptions {
  jobId: string;
  clusterIds: ClusteringClusterFilterOption[];
  itemCount: number;
}

/** 明细分页。`total` 是筛选后的总数，不是本页条数。 */
export interface ClusteringJobItemsPage {
  jobId: string;
  runId: string;
  total: number;
  offset: number;
  limit: number;
  returned: number;
  hasMore: boolean;
  clusterId?: number | null;
  /** 本次筛选实际扫描的条数。 */
  scanned: number;
  /** 按文本筛选时是否因达到扫描上限而截断；true 表示结果不完整。 */
  truncated: boolean;
  items: ClusteringResultItem[];
}

/** 作业提交选项：在同步选项之上加"数据集引用"与"幂等键"。 */
export interface ClusteringJobSubmitOptions extends ClusteringRunOptions {
  datasetId?: string | null;
  idempotencyKey?: string | null;
}

/** 数据集引用（上传接口返回，供作业提交引用）。 */
export interface ClusteringDatasetReference {
  datasetId: string;
  sourceName: string;
  total: number;
  createdAt?: string | null;
  warnings: string[];
}
