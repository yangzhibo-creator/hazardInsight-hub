/**
 * 异步聚类作业面板（大规模数据的入口）。
 *
 * 与「开始聚类」按钮的关系：小样本用同步接口即点即看；样本到万级/十万级时，
 * 同步接口会把整份明细塞进一次 HTTP 响应，页面只能干等，刷新还等于丢任务。
 * 这个面板走作业路径：
 *
 * * 提交只入队，立刻拿到 `jobId`，页面照常可用；
 * * 进度靠轮询，可以随时**取消**（后端杀执行子进程，资源真的被释放）；
 * * 明细走**服务端分页**——列表一次只取一页，"总数"是后端筛选后的真实总数，
 *   因此不会出现"看起来只筛出了 50 条"这种把当前页当全量的假象。
 *
 * 关键词筛选由后端执行；若命中扫描上限，后端会回 `truncated=true`，
 * 这里必须如实提示"结果不完整"——不能把截断当成筛选结果。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  ClusteringDatasetItem,
  ClusteringJobInfo,
  ClusteringJobItemsPage,
  ClusteringJobSubmitOptions,
  ClusteringResultItem,
} from '../../shared/clustering';
import { fetchClusteringJobFilters, fetchClusteringJobItems } from '../api/clustering';
import { IconPlay, IconRefresh } from './icons';
import { clusterColor } from '../lib/clusterColor';
import type { ClusteringJobApi } from '../lib/useClusteringJob';

/** 每页条数。后端还有 500 的硬上限，这里取一个对表格友好的值。 */
const PAGE_SIZE = 50;

const STATUS_LABELS: Record<ClusteringJobInfo['status'], string> = {
  queued: '排队中',
  running: '执行中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

/** 作业是否可以被取消（终态的作业取消没有意义）。 */
export function isCancellable(job: ClusteringJobInfo | null): boolean {
  return Boolean(job && !job.terminal);
}

/** 一次性说清"这个作业现在是什么状态"——避免页面各处各写一套判断。 */
export function describeJobStatus(job: ClusteringJobInfo | null): string {
  if (!job) return '尚未提交作业';
  if (job.status === 'running') return `执行中（共 ${job.itemCount} 条样本）`;
  if (job.status === 'queued') return `排队中（共 ${job.itemCount} 条样本）`;
  if (job.status === 'succeeded') return `已完成（共 ${job.itemCount} 条样本）`;
  if (job.status === 'cancelled') {
    return job.hardCancelled ? '已取消（执行进程已终止）' : '已取消';
  }
  return job.error ? `失败：${job.error.message}` : '失败';
}

interface Props {
  /**
   * 作业状态机（由页面持有）。
   *
   * 之所以从外面传进来：提交按钮要和「开始聚类」并排放在同一行运行选项里，
   * 而不是藏在页面下方的面板里——选项在哪，提交就在哪，不该让人满页找。
   * 面板只负责"作业跑起来之后"的进度、取消与分页明细。
   */
  api: ClusteringJobApi;
  /** 当前数据集的样本。样本多时只提交 `datasetId`，不重传整份列表。 */
  items: ClusteringDatasetItem[];
  /** 已上传数据集返回的引用 ID；有它就不必把样本再传一遍。 */
  datasetId?: string | null;
  options: ClusteringJobSubmitOptions;
  /** 引擎未就绪或正在上传数据时禁用提交。 */
  disabled?: boolean;
  /**
   * 是否在面板内渲染「提交后台作业」按钮。
   * 页面把提交按钮放进运行选项行时传 false；此时没有作业可展示的面板自行隐藏。
   */
  showSubmit?: boolean;
  /**
   * 本次作业的参考数据库（可读名称）。
   * 与聚类选项同一份来源：作业提交后在服务端跑的就是这个库，
   * 面板必须把它写出来，否则几十分钟后没人记得这份结果是哪个库算的。
   */
  knowledgeBaseLabel?: string;
}

export function ClusteringJobPanel({
  api: jobApi,
  items,
  datasetId,
  options,
  disabled = false,
  showSubmit = true,
  knowledgeBaseLabel = '',
}: Props) {
  const { job, result, error } = jobApi;

  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [keyword, setKeyword] = useState('');
  const [page, setPage] = useState(0);
  const [pageData, setPageData] = useState<ClusteringJobItemsPage | null>(null);
  const [pageError, setPageError] = useState('');
  const [clusterOptions, setClusterOptions] = useState<{ clusterId: number; size: number }[]>([]);

  const jobId = job?.status === 'succeeded' ? job.jobId : null;

  // 作业成功后拉一次筛选取值（只读后端索引，与样本量无关）
  useEffect(() => {
    if (!jobId) {
      setClusterOptions([]);
      return;
    }
    let cancelled = false;
    void fetchClusteringJobFilters(jobId)
      .then((data) => {
        if (!cancelled) setClusterOptions(data.clusterIds);
      })
      .catch(() => {
        // 筛选取值拿不到不该让整个结果区不可用：列表仍可按默认顺序翻页
        if (!cancelled) setClusterOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // 服务端分页：页、簇筛选、关键词任一变化都重新取数（不是本地切当前页）
  useEffect(() => {
    if (!jobId) {
      setPageData(null);
      return;
    }
    let cancelled = false;
    setPageError('');
    void fetchClusteringJobItems(jobId, {
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      clusterId: clusterFilter,
      keyword: keyword.trim() || null,
    })
      .then((data) => {
        if (!cancelled) setPageData(data);
      })
      .catch((cause) => {
        if (!cancelled) setPageError((cause as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, page, clusterFilter, keyword]);

  const start = useCallback(() => {
    void jobApi.submit(items, datasetId ? { ...options, datasetId } : options);
  }, [jobApi, items, datasetId, options]);

  /** 改筛选条件必须同时回到第一页，否则会停在越界页码显示空白。 */
  const changeClusterFilter = useCallback((next: number | null) => {
    setPage(0);
    setClusterFilter(next);
  }, []);

  const changeKeyword = useCallback((next: string) => {
    setPage(0);
    setKeyword(next);
  }, []);

  const total = pageData?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const visibleItems: ClusteringResultItem[] = useMemo(() => pageData?.items ?? [], [pageData]);

  /**
   * 提交按钮在页面运行选项行里时，面板只在"有作业可看"或"提交失败要报错"时出现。
   * 否则页面下方会长期挂着一张空卡片，把真正的结果推得更远。
   */
  if (!showSubmit && !job && !error) return null;

  return (
    <section className="card card-pad clustering-job-panel" aria-label="异步聚类作业">
      <div className="clustering-toolbar">
        <div className="clustering-dataset">
          <strong>后台作业（大数据量推荐）</strong>
          <div className="small muted">{describeJobStatus(job)}</div>
        </div>
        {showSubmit && (
          <button
            className="btn btn-primary"
            disabled={disabled || items.length === 0 || isCancellable(job)}
            onClick={start}
          >
            <IconPlay size={15} />提交后台作业
          </button>
        )}
        {isCancellable(job) && (
          <button className="btn btn-outline" onClick={() => void jobApi.cancel()}>
            取消作业
          </button>
        )}
        {job && (
          <button className="btn btn-outline" onClick={() => jobApi.reset()}>
            <IconRefresh size={14} />清空
          </button>
        )}
      </div>

      <p className="small muted">
        提交后立刻返回作业 ID，可以随时取消（后端会终止执行进程并释放资源），
        明细按页读取，不需要把整份结果下载到浏览器。算法 / profile / 净化 / 参考数据库沿用上方运行选项。
      </p>
      {knowledgeBaseLabel && (
        <p className="small muted">
          参考数据库：<strong>{knowledgeBaseLabel}</strong>
        </p>
      )}

      {error && <p className="clustering-notice is-error">{error}</p>}
      {job?.error && <p className="clustering-notice is-error">{job.error.message}</p>}
      {job && !job.terminal && (
        <p className="small muted">
          作业 {job.jobId} · 状态「{STATUS_LABELS[job.status]}」
          {job.cancelRequested ? ' · 已请求取消' : ''}。页面可以刷新，作业在服务端继续执行。
        </p>
      )}

      {job?.status === 'succeeded' && result && (
        <>
          <div className="clustering-stat-grid" aria-label="作业结果统计">
            {[
              ['样本总数', String(result.summary.totalSamples)],
              ['簇数量', String(result.summary.clusterCount)],
              ['噪声点', String(result.summary.noiseCount)],
              ['明细条数', String(result.detail.itemCount ?? result.summary.totalSamples)],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          {clusterOptions.length > 0 && (
            <div className="clustering-toolbar">
              <label>
                所属簇
                <select
                  className="select"
                  value={clusterFilter === null ? '' : String(clusterFilter)}
                  onChange={(event) =>
                    changeClusterFilter(event.target.value === '' ? null : Number(event.target.value))
                  }
                >
                  <option value="">全部</option>
                  {clusterOptions.map((option) => (
                    <option key={option.clusterId} value={option.clusterId}>
                      {option.clusterId < 0 ? '未归类' : `簇 ${option.clusterId}`}（{option.size} 条）
                    </option>
                  ))}
                </select>
              </label>
              <label>
                关键词
                <input
                  className="clustering-job-keyword"
                  type="search"
                  value={keyword}
                  placeholder="按文本筛选（服务端执行）"
                  onChange={(event) => changeKeyword(event.target.value)}
                />
              </label>
              <span className="small muted">共 {total} 条（服务端筛选后的总数）</span>
            </div>
          )}

          {pageData?.truncated && (
            <p className="clustering-notice">
              关键词筛选只扫描了前 {pageData.scanned} 条，结果<strong>不完整</strong>。请先缩小簇范围再搜索。
            </p>
          )}
          {pageError && <p className="clustering-notice is-error">{pageError}</p>}

          <ul className="clustering-job-items">
            {visibleItems.map((item) => (
              <li key={item.id}>
                <span
                  className="clustering-job-dot"
                  style={{ background: clusterColor(item.clusterId) }}
                  aria-hidden="true"
                />
                <span className="clustering-job-cluster">
                  {item.clusterId < 0 ? '未归类' : item.clusterLabel}
                </span>
                <span className="clustering-job-text">{item.text}</span>
              </li>
            ))}
          </ul>

          <div className="clustering-pager">
            <button
              className="btn btn-outline btn-sm"
              disabled={currentPage <= 0}
              onClick={() => setPage((value) => Math.max(0, value - 1))}
            >
              上一页
            </button>
            <span className="small muted">
              第 {currentPage + 1} / {pageCount} 页
            </span>
            <button
              className="btn btn-outline btn-sm"
              disabled={currentPage >= pageCount - 1}
              onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}
            >
              下一页
            </button>
          </div>
        </>
      )}
    </section>
  );
}
