/**
 * 聚类分析接口客户端。
 *
 * 约定：
 * - 一律走**同源相对路径** `/api/clustering/*`，由 Vite / Node 代理转发到本机
 *   Python 聚类服务，前端任何位置都不硬编码 Python 的地址与端口；
 * - 后端统一返回 `{ success, data, error }`，本模块负责拆封：成功取 `data`，
 *   失败统一抛 `ApiError`（带 `code`），页面只需一套错误分支；
 * - 服务未启动、代理不可达等网络层错误也转成 `ApiError`，避免出现"按钮点了没反应"。
 */

import type {
  ClusteringAlgorithmInfo,
  ClusteringData,
  ClusteringDatasetItem,
  ClusteringDatasetPreview,
  ClusteringDatasetReference,
  ClusteringHealthData,
  ClusteringJobFilterOptions,
  ClusteringJobInfo,
  ClusteringJobItemsPage,
  ClusteringJobResultData,
  ClusteringJobStatus,
  ClusteringJobSubmitOptions,
  ClusteringKnowledgeBaseBuildOptions,
  ClusteringKnowledgeBaseDetail,
  ClusteringKnowledgeBaseInfo,
  ClusteringKnowledgeBaseJob,
  ClusteringOfflineBaseline,
  ClusteringProfileInfo,
  ClusteringRunOptions,
  ClusteringSampleData,
} from '../../shared/clustering';
import { ApiError } from './client';

/** 网关基址：与后端 `clustering_router` 的挂载前缀保持一致。 */
const BASE = '/api/clustering';

interface Envelope<T> {
  success: boolean;
  data: T | null;
  error: { code: string; message: string; requestId?: string | null; detail?: string | null } | null;
}

/** 统一请求：拆封包裹 + 归一化错误。 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch (cause) {
    throw new ApiError(
      '无法连接聚类服务，请确认后端已启动（npm run dev:py）。',
      'NETWORK_ERROR',
      undefined,
    );
  }

  const raw = await response.text();
  let envelope: Envelope<T> | null = null;
  if (raw) {
    try {
      envelope = JSON.parse(raw) as Envelope<T>;
    } catch {
      // 代理层返回的 HTML 错误页（如 502）走下面的兜底分支
      envelope = null;
    }
  }

  if (response.ok && envelope?.success && envelope.data != null) {
    return envelope.data;
  }

  const detail = envelope?.error;
  const fallback = response.ok
    ? '聚类服务返回了无法识别的响应。'
    : response.status === 502 || response.status === 503 || response.status === 504
      ? '聚类服务暂时不可用，请稍后重试。'
      : `请求失败（HTTP ${response.status}）。`;
  throw new ApiError(detail?.message ?? fallback, detail?.code ?? 'HTTP_ERROR', response.status);
}

/** 仅在需要强制重新校验时才带上 `refresh`，保持缓存友好的干净 URL。 */
const refreshParam = (refresh: boolean) => (refresh ? '?refresh=true' : '');

/** 健康检查：服务与聚类引擎的就绪状态。 */
export const fetchClusteringHealth = (refresh = false) =>
  request<ClusteringHealthData>(`/health${refreshParam(refresh)}`);

/** 列出可用 profile（含可用性与不可用原因）。 */
export const fetchClusteringProfiles = (refresh = false) =>
  request<{ profiles: ClusteringProfileInfo[] }>(`/profiles${refreshParam(refresh)}`).then(
    (data) => data.profiles,
  );

/** 列出引擎暴露的聚类算法及其依赖可用性。 */
export const fetchClusteringAlgorithms = () =>
  request<{ algorithms: ClusteringAlgorithmInfo[] }>('/algorithms').then(
    (data) => data.algorithms,
  );

/** 读取内置示例数据（真实核电工程隐患抽样）。 */
export const fetchClusteringSample = () => request<ClusteringSampleData>('/sample');

/**
 * 读取基准结果（各分析配置在全量测试集上的参考指标）。
 *
 * 页面进入后自动调用一次；随响应返回的出处与口径（`source` / `verifiedAt` /
 * `fullRun`）由界面原样展示，与本次运行分开呈现。
 */
export const fetchClusteringBaseline = () => request<ClusteringOfflineBaseline>('/baseline');

/** 上传并解析数据文件（CSV / XLSX / JSON / TXT）。 */
export function uploadClusteringDataset(file: File) {
  const body = new FormData();
  body.append('file', file);
  return request<ClusteringDatasetPreview>('/datasets', { method: 'POST', body });
}

/** 执行一次聚类。 */
export function runClustering(items: ClusteringDatasetItem[], options: ClusteringRunOptions = {}) {
  return request<ClusteringData>('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items, options }),
  });
}

/* ------------------------------------------------------------------ 异步作业 */

/**
 * 提交异步聚类作业。
 *
 * 与 `runClustering` 的分工：样本多的时候必须走这条。它只做准入与入队就返回，
 * HTTP 请求不会被挂在一次几十分钟的计算上；后续用 `fetchClusteringJob` 轮询，
 * 用 `fetchClusteringJobItems` 分页取明细（不必下载全量），用 `cancelClusteringJob` 停掉。
 *
 * `items` 与 `options.datasetId` 二选一：按引用提交时请求体里只有一个 ID。
 */
export function submitClusteringJob(
  items: ClusteringDatasetItem[],
  options: ClusteringJobSubmitOptions = {},
) {
  const { idempotencyKey, ...rest } = options;
  // 幂等键同时放头部与 body：代理层可能丢掉自定义头，而 body 一定到得了服务端。
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (idempotencyKey) headers['Idempotency-Key'] = idempotencyKey;
  return request<ClusteringJobInfo>('/jobs', {
    method: 'POST',
    headers,
    body: JSON.stringify({ items, options: rest }),
  });
}

/** 查询作业状态（轮询用；`terminal=true` 时应停止轮询）。 */
export const fetchClusteringJob = (jobId: string) =>
  request<ClusteringJobInfo>(`/jobs/${encodeURIComponent(jobId)}`);

/** 列出作业（新→旧）。 */
export const fetchClusteringJobs = (limit = 20, status?: ClusteringJobStatus) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set('status', status);
  return request<{ jobs: ClusteringJobInfo[] }>(`/jobs?${params.toString()}`).then(
    (data) => data.jobs,
  );
};

/** 取消作业：排队中的直接出队，执行中的杀掉执行子进程。 */
export const cancelClusteringJob = (jobId: string) =>
  request<ClusteringJobInfo>(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });

/** 读取作业结果摘要（统计 / 簇 / 抽样坐标，不含明细）。 */
export const fetchClusteringJobResult = (jobId: string) =>
  request<ClusteringJobResultData>(`/jobs/${encodeURIComponent(jobId)}/result`);

/** 读取明细筛选取值（可见的簇及各自条数）。 */
export const fetchClusteringJobFilters = (jobId: string) =>
  request<ClusteringJobFilterOptions>(`/jobs/${encodeURIComponent(jobId)}/filters`);

/** 服务端分页读取明细。 */
export function fetchClusteringJobItems(
  jobId: string,
  query: {
    offset?: number;
    limit?: number;
    clusterId?: number | null;
    assignmentStatus?: string | null;
    keyword?: string | null;
  } = {},
) {
  const params = new URLSearchParams();
  if (query.offset !== undefined) params.set('offset', String(query.offset));
  if (query.limit !== undefined) params.set('limit', String(query.limit));
  if (query.clusterId !== undefined && query.clusterId !== null) {
    params.set('clusterId', String(query.clusterId));
  }
  if (query.assignmentStatus) params.set('assignmentStatus', query.assignmentStatus);
  if (query.keyword) params.set('keyword', query.keyword);
  const suffix = params.toString();
  return request<ClusteringJobItemsPage>(
    `/jobs/${encodeURIComponent(jobId)}/items${suffix ? `?${suffix}` : ''}`,
  );
}

/** 列出已保存的数据集引用。 */
export const fetchClusteringDatasets = (limit = 20) =>
  request<{ datasets: ClusteringDatasetReference[] }>(`/datasets?limit=${limit}`).then(
    (data) => data.datasets,
  );

/* ------------------------------------------------------------------ 偏差数据库 */

/** 列出已有知识库（元信息，不含条目内容）。 */
export const fetchKnowledgeBases = () =>
  request<{ knowledgeBases: ClusteringKnowledgeBaseInfo[] }>('/knowledge-bases').then(
    (data) => data.knowledgeBases,
  );

/** 读取知识库详情与一页条目。 */
export const fetchKnowledgeBaseDetail = (knowledgeBaseId: string, offset = 0, limit = 50) => {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  return request<ClusteringKnowledgeBaseDetail>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}?${params.toString()}`,
  );
};

/** 上传语料并提交一次知识库生成任务（异步，返回 jobId）。 */
export function buildKnowledgeBase(file: File, options: ClusteringKnowledgeBaseBuildOptions) {
  const body = new FormData();
  body.append('file', file);
  body.append('name', options.name);
  body.append('mode', options.mode);
  if (options.ident) body.append('ident', options.ident);
  if (options.modelId) body.append('model_id', options.modelId);
  return request<ClusteringKnowledgeBaseJob>('/knowledge-bases', { method: 'POST', body });
}

/** 查询知识库生成进度；`terminal=true` 时停止轮询。 */
export const fetchKnowledgeBaseJob = (jobId: string) =>
  request<ClusteringKnowledgeBaseJob>(`/knowledge-bases/jobs/${encodeURIComponent(jobId)}`);

/** 列出知识库生成任务（新→旧）。 */
export const fetchKnowledgeBaseJobs = (limit = 20) =>
  request<{ jobs: ClusteringKnowledgeBaseJob[] }>(`/knowledge-bases/jobs?limit=${limit}`).then(
    (data) => data.jobs,
  );
