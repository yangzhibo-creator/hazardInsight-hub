/**
 * 「历史测试记录」的纯逻辑。
 *
 * 历史来源是**异步作业历史**（`GET /api/clustering/jobs`）：后端本来就是按
 * 新→旧返回所有提交过的聚类作业，离线基准归档只是单份论文复现数据、没有历史，
 * 两者口径不同，不能混在一起展示，因此这里只处理作业记录。
 *
 * 页面需要的格式化、排序、展开态判定都集中在这里，组件只负责取数与渲染，
 * 这样这些判断可以脱离 DOM 直接测。
 */

import type {
  ClusteringClusterGroup,
  ClusteringJobInfo,
  ClusteringJobResultData,
  ClusteringJobStatus,
} from '../../shared/clustering';

/** 历史列表一次取多少条。后端上限 100，取 50 足够覆盖演示现场的历史。 */
export const HISTORY_LIST_LIMIT = 50;

/** 展开后明细分页每页条数；后端上限 500。 */
export const HISTORY_ITEMS_PAGE_SIZE = 20;

const STATUS_LABELS: Record<ClusteringJobStatus, string> = {
  queued: '排队中',
  running: '执行中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

/** 作业是否尚未进入终态（终态不再变化，只有非终态才需要"刷新看进度"）。 */
export function hasActiveJobs(jobs: ClusteringJobInfo[]): boolean {
  return jobs.some((job) => !job.terminal);
}

/** 未完成作业的数量，用于提示文案。 */
export function countActiveJobs(jobs: ClusteringJobInfo[]): number {
  return jobs.filter((job) => !job.terminal).length;
}

/** ISO 时间字符串 → 毫秒时间戳；缺失或无法解析时返回 null（而不是 NaN）。 */
function parseTime(value?: string | null): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * 新→旧排序。
 *
 * 后端已经排过一次，但页面可能把「本次刚提交的作业」合并进来，排序必须由前端
 * 自己保证一致；`createdAt` 缺失/相同时用 jobId 兜底，避免顺序抖动。
 */
export function sortJobsNewestFirst(jobs: ClusteringJobInfo[]): ClusteringJobInfo[] {
  return [...jobs].sort((a, b) => {
    const left = parseTime(a.createdAt) ?? 0;
    const right = parseTime(b.createdAt) ?? 0;
    if (left !== right) return right - left;
    return a.jobId.localeCompare(b.jobId);
  });
}

/** 状态短标签（用于列表徽标；详细原因由 describeJobOutcome 给出）。 */
export function historyStatusLabel(job: ClusteringJobInfo): string {
  return STATUS_LABELS[job.status];
}

/** 状态徽标的配色类：成功绿、失败红、取消黄，其余保持中性。 */
export function historyStatusClass(job: ClusteringJobInfo): string {
  if (job.status === 'succeeded') return 'is-ok';
  if (job.status === 'failed') return 'is-bad';
  if (job.status === 'cancelled') return 'is-warn';
  return '';
}

/** 记录时间。无法解析时原样回显，绝不显示 "Invalid Date"。 */
export function formatJobTime(value?: string | null): string {
  const parsed = parseTime(value);
  if (parsed === null) return value ? value : '—';
  return new Date(parsed).toLocaleString('zh-CN', { hour12: false });
}

/**
 * 耗时口径：优先 `startedAt → finishedAt`，缺失时退化到 `createdAt → updatedAt`。
 * 非终态显示「进行中」而不是拿"现在"算出一个会跳动的数字。
 */
export function formatJobDuration(job: ClusteringJobInfo): string {
  const start = parseTime(job.startedAt) ?? parseTime(job.createdAt);
  if (start === null) return '—';
  const end = parseTime(job.finishedAt) ?? parseTime(job.updatedAt);
  if (end === null) return job.terminal ? '—' : '进行中';
  const ms = Math.max(0, end - start);
  if (ms < 1000) return `${ms} 毫秒`;
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`;
  const hours = Math.floor(minutes / 60);
  const restMinutes = minutes % 60;
  return restMinutes ? `${hours} 小时 ${restMinutes} 分` : `${hours} 小时`;
}

/** 展开后能看到的形态：成功读结果、失败给原因、取消说明、未完成提示刷新。 */
export type JobHistoryExpandKind = 'result' | 'running' | 'failed' | 'cancelled';

export function historyExpandKind(job: ClusteringJobInfo): JobHistoryExpandKind {
  if (job.status === 'succeeded') return 'result';
  if (job.status === 'failed') return 'failed';
  if (job.status === 'cancelled') return 'cancelled';
  return 'running';
}

/**
 * 点击同一行时收起、点击另一行时切换过去（同一时刻只展开一条，避免整页被撑开）。
 */
export function toggleExpandedJob(current: string | null, jobId: string): string | null {
  return current === jobId ? null : jobId;
}

/** 展开区里的人话结论：失败/取消/执行中的原因与下一步动作。 */
export function describeJobOutcome(job: ClusteringJobInfo): string {
  switch (historyExpandKind(job)) {
    case 'failed':
      return job.error
        ? `作业执行失败：${job.error.message}（错误码 ${job.error.code}）`
        : '作业执行失败，服务端没有返回具体原因。';
    case 'cancelled':
      return job.hardCancelled
        ? '作业已取消，执行进程已终止，不会再有结果。'
        : '作业已取消（在排队阶段被移出队列），没有产生结果。';
    case 'running':
      return job.status === 'queued'
        ? `作业还在排队（共 ${job.itemCount} 条样本），开始执行后这里会显示进度。`
        : `作业正在执行（共 ${job.itemCount} 条样本）。结果要在作业完成后才能读取，可点「刷新」查看最新状态。`;
    default:
      return '';
  }
}

/**
 * 结果摘要卡片里的统计项。
 *
 * 只列出能解释"这次跑出了什么"的字段；语义路径专有的质量分不存在时整项省略，
 * 不显示 0 或 '—' 这类会被误读的占位。
 */
export function summarizeJobResult(result: ClusteringJobResultData): Array<[string, string]> {
  const summary = result.summary;
  const rows: Array<[string, string]> = [
    ['样本总数', String(summary.totalSamples)],
    ['簇数量', String(summary.clusterCount)],
    ['噪声点', String(summary.noiseCount)],
    ['最大簇', String(summary.largestClusterSize)],
    ['算法', summary.algorithm],
    ['Profile', summary.profileId],
    ['实现版本', summary.implementationVersion],
    ['引擎耗时', `${summary.elapsedMs} ms`],
    ['网关耗时', `${summary.gatewayMs} ms`],
  ];
  if (summary.qualityScore !== undefined && summary.qualityScore !== null) {
    rows.push(['簇质量分', summary.qualityScore.toFixed(3)]);
  }
  return rows;
}

/** 簇列表概览：按大小降序取前 N 个（噪声簇也参与排序，它通常最大）。 */
export function clusterOverview(
  clusters: ClusteringClusterGroup[],
  limit = 8,
): ClusteringClusterGroup[] {
  return [...clusters]
    .sort((a, b) => b.size - a.size || a.clusterId - b.clusterId)
    .slice(0, Math.max(0, limit));
}

/** 明细总页数；total 非法或为 0 时也至少 1 页，便于渲染 "第 1 / 1 页"。 */
export function jobItemsPageCount(total: number, pageSize = HISTORY_ITEMS_PAGE_SIZE): number {
  if (!Number.isFinite(total) || total <= 0 || pageSize <= 0) return 1;
  return Math.max(1, Math.ceil(total / pageSize));
}

/** 页码收敛到合法区间，避免筛选变化后停在越界页显示空白。 */
export function clampPage(page: number, pageCount: number): number {
  const max = Math.max(0, pageCount - 1);
  if (!Number.isFinite(page)) return 0;
  return Math.min(Math.max(0, Math.floor(page)), max);
}

/** 列表里的作业标识：Run ID 是新→旧之外最可靠的定位手段，缺失时退回 jobId。 */
export function jobRunLabel(job: ClusteringJobInfo): string {
  return job.runId ?? job.jobId;
}
