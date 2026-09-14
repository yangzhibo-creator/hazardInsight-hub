/**
 * 历史测试记录（点击就地展开）。
 *
 * 数据来源是**服务端保存的聚类作业记录**（`GET /api/clustering/jobs`）：
 * 每次聚类都会在服务端留下一条记录，这是唯一真正带时间线、能列出"过去跑过哪些分析"的来源。
 *
 * 交互上刻意**不用抽屉、不跳页**：点一行就在列表内展开该作业的结果摘要、簇概览与
 * 分页明细，再点一次收起。作业状态是历史的一部分，失败 / 取消 / 仍在执行都必须
 * 就地给出可读原因，而不是打开后发现空白。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  ClusteringJobInfo,
  ClusteringJobItemsPage,
  ClusteringJobResultData,
  ClusteringOfflineBaseline,
  ClusteringSummary,
} from '../../shared/clustering';
import {
  fetchClusteringBaseline,
  fetchClusteringJobItems,
  fetchClusteringJobResult,
  fetchClusteringJobs,
} from '../api/clustering';
import { clusterColor, clusterTint } from '../lib/clusterColor';
import {
  METRIC_DEFINITIONS,
  archivedBenchmarkColumns,
  archivedColumnLabel,
  comparisonRows,
  describeGroundTruth,
  formatDelta,
  formatMetricValue,
  hasComputedMetrics,
  hasControlMetrics,
  metricRows,
  shouldUseArchivedBenchmark,
} from '../lib/clusteringMetrics';
import { logDebugWarnings, splitClusteringWarnings } from '../lib/clusteringWarnings';
import {
  HISTORY_ITEMS_PAGE_SIZE,
  HISTORY_LIST_LIMIT,
  clampPage,
  clusterOverview,
  countActiveJobs,
  describeJobOutcome,
  formatJobDuration,
  formatJobTime,
  historyExpandKind,
  historyStatusClass,
  historyStatusLabel,
  jobItemsPageCount,
  jobRunLabel,
  sortJobsNewestFirst,
  summarizeJobResult,
  toggleExpandedJob,
} from '../lib/clusteringJobHistory';
import { IconChevronDown, IconChevronRight, IconHistory, IconRefresh } from './icons';

export function ClusteringJobHistory() {
  const [jobs, setJobs] = useState<ClusteringJobInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  /** 同一时刻只展开一条：历史列表可能很长，多条同时展开会让页面失去焦点。 */
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const loadJobs = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      setJobs(sortJobsNewestFirst(await fetchClusteringJobs(HISTORY_LIST_LIMIT)));
    } catch (cause) {
      // 取数失败要给出可读错误，而不是让卡片停在"加载中"或显示空白
      setError((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);

  const toggle = useCallback((jobId: string) => {
    setExpandedId((current) => toggleExpandedJob(current, jobId));
  }, []);

  const activeCount = countActiveJobs(jobs);

  return (
    <section className="card card-pad" aria-label="历史测试记录">
      <div className="clustering-section-head">
        <h2><IconHistory size={15} /> 历史测试记录</h2>
        <button className="btn btn-outline btn-sm" disabled={busy} onClick={() => void loadJobs()}>
          <IconRefresh size={14} />{busy ? '加载中…' : '刷新'}
        </button>
      </div>
      <p className="small muted">
        这里列出服务端保存的历史聚类记录（新→旧，最多 {HISTORY_LIST_LIMIT} 条），点击任意一条
        <strong>就地展开</strong>结果摘要与明细，再点一次收起。每条都是真实跑过的分析记录。
      </p>

      {error && <p className="clustering-notice is-error">历史记录加载失败：{error}</p>}
      {activeCount > 0 && (
        <p className="clustering-notice">
          有 {activeCount} 个作业尚未完成；它们可能仍在服务端执行，点「刷新」查看最新进度。
        </p>
      )}
      {!busy && !error && jobs.length === 0 && (
        <p className="small muted">
          还没有历史记录。完成一次聚类后，这里就会出现可回看的记录。
        </p>
      )}

      {jobs.length > 0 && (
        <div className="clustering-history-scroll">
          <div className="clustering-history clustering-history-head" aria-hidden="true">
            <span />
            <span>时间</span>
            <span>状态</span>
            <span>Profile · 算法</span>
            <span>样本</span>
            <span>耗时</span>
            <span>Run ID</span>
            <span>备注</span>
          </div>
          <ul className="clustering-history-list">
            {jobs.map((job) => {
              const open = job.jobId === expandedId;
              // 只统计"界面真的会显示"的提示：全被分流到控制台时不该冒出"N 条提示"
              const visibleWarnings = splitClusteringWarnings(job.warnings).visible;
              return (
                <li key={job.jobId} className={`clustering-history-item ${open ? 'is-open' : ''}`}>
                  <button
                    type="button"
                    className="clustering-history clustering-history-row"
                    aria-expanded={open}
                    onClick={() => toggle(job.jobId)}
                  >
                    <span className="clustering-history-caret" aria-hidden="true">
                      {open ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
                    </span>
                    <span>{formatJobTime(job.createdAt)}</span>
                    <span>
                      <span className={`clustering-status ${historyStatusClass(job)}`}>
                        {historyStatusLabel(job)}
                      </span>
                    </span>
                    <span className="clustering-history-profile">
                      {job.profileId ?? '—'}
                      {job.algorithm ? `（${job.algorithm}）` : ''}
                    </span>
                    <span>{job.itemCount} 条</span>
                    <span>{formatJobDuration(job)}</span>
                    <span className="mono clustering-history-run" title={job.jobId}>{jobRunLabel(job)}</span>
                    <span className="clustering-history-flag">
                      {job.error
                        ? `有错误：${job.error.message}`
                        : visibleWarnings.length
                          ? `${visibleWarnings.length} 条提示`
                          : ''}
                    </span>
                  </button>
                  {open && <JobHistoryDetail job={job} onRefresh={loadJobs} refreshing={busy} />}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}

/** 单条历史记录的展开区：按状态决定是读结果、给原因，还是提示刷新。 */
function JobHistoryDetail({
  job,
  onRefresh,
  refreshing,
}: {
  job: ClusteringJobInfo;
  onRefresh: () => Promise<void>;
  refreshing: boolean;
}) {
  const kind = historyExpandKind(job);
  const [result, setResult] = useState<ClusteringJobResultData | null>(null);
  const [resultBusy, setResultBusy] = useState(false);
  const [resultError, setResultError] = useState('');
  /** 结果读取失败后的重试计数：只改它就能重新触发下面两个取数 effect。 */
  const [retry, setRetry] = useState(0);
  const [page, setPage] = useState(0);
  const [items, setItems] = useState<ClusteringJobItemsPage | null>(null);
  const [itemsBusy, setItemsBusy] = useState(false);
  const [itemsError, setItemsError] = useState('');

  // 失败 / 取消 / 执行中的作业没有结果可读，直接打接口只会拿到注定失败的响应，
  // 因此这里只在成功作业上取摘要。
  useEffect(() => {
    if (kind !== 'result') return;
    let cancelled = false;
    setResultBusy(true);
    setResultError('');
    void fetchClusteringJobResult(job.jobId)
      .then((data) => {
        if (!cancelled) setResult(data);
      })
      .catch((cause) => {
        if (!cancelled) setResultError((cause as Error).message);
      })
      .finally(() => {
        if (!cancelled) setResultBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [job.jobId, kind, retry]);

  // 明细走服务端分页：展开时取第一页，翻页只换偏移量，不把全量拉进浏览器。
  useEffect(() => {
    if (kind !== 'result') return;
    let cancelled = false;
    setItemsBusy(true);
    setItemsError('');
    void fetchClusteringJobItems(job.jobId, {
      offset: page * HISTORY_ITEMS_PAGE_SIZE,
      limit: HISTORY_ITEMS_PAGE_SIZE,
    })
      .then((data) => {
        if (!cancelled) setItems(data);
      })
      .catch((cause) => {
        if (!cancelled) setItemsError((cause as Error).message);
      })
      .finally(() => {
        if (!cancelled) setItemsBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [job.jobId, kind, page, retry]);

  /**
   * 告警分流：作业级与结果级的原始英文码统一过滤后只写控制台（调试窗口），
   * 界面只留能读懂的中文提示。分两个 memo 是为了保持原有的显示来源不变
   * （作业级提示只在非成功态出现，结果级提示只在成功态出现）。
   */
  const jobWarnings = useMemo(() => splitClusteringWarnings(job.warnings), [job]);
  const resultWarnings = useMemo(
    () => splitClusteringWarnings(result ? [...result.summary.warnings, ...result.warnings] : []),
    [result],
  );
  useEffect(() => {
    logDebugWarnings(`作业 ${job.jobId}`, [...jobWarnings.debug, ...resultWarnings.debug]);
  }, [job.jobId, jobWarnings, resultWarnings]);

  if (kind !== 'result') {
    return (
      <div className="clustering-history-detail">
        <p className={`clustering-notice ${kind === 'failed' ? 'is-error' : ''}`}>
          {describeJobOutcome(job)}
        </p>
        {kind === 'running' && (
          <button className="btn btn-outline btn-sm" disabled={refreshing} onClick={() => void onRefresh()}>
            <IconRefresh size={14} />{refreshing ? '刷新中…' : '刷新状态'}
          </button>
        )}
        {jobWarnings.visible.map((warning) => (
          <p key={warning} className="clustering-notice">{warning}</p>
        ))}
      </div>
    );
  }

  const total = items?.total ?? result?.detail.itemCount ?? job.itemCount;
  const pageCount = jobItemsPageCount(total);
  const currentPage = clampPage(page, pageCount);
  const clusterRows = result ? clusterOverview(result.clusters) : [];

  return (
    <div className="clustering-history-detail">
      {resultBusy && <p className="small muted">正在读取结果摘要…</p>}
      {resultError && (
        <>
          <p className="clustering-notice is-error">结果摘要读取失败：{resultError}</p>
          <button className="btn btn-outline btn-sm" onClick={() => setRetry((value) => value + 1)}>
            <IconRefresh size={14} />重试
          </button>
        </>
      )}

      {result && (
        <>
          <div className="clustering-stat-grid clustering-history-stats" aria-label="历史作业结果统计">
            {summarizeJobResult(result).map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          {resultWarnings.visible.map((warning) => (
            <p key={warning} className="clustering-notice">{warning}</p>
          ))}

          {/* 基准指标：算出来的走实测，没有指标的整份测试集运行回退到参考结果 */}
          <JobBenchmark job={job} summary={result.summary} />

          <div className="clustering-history-clusters">
            <span className="small muted">簇概览</span>
            {clusterRows.length === 0 ? (
              <span className="small muted">本次结果没有返回任何簇。</span>
            ) : (
              clusterRows.map((cluster) => (
                <span
                  key={cluster.clusterId}
                  className="cluster-pill"
                  style={{
                    background: clusterTint(cluster.clusterId, 0.14),
                    color: clusterColor(cluster.clusterId),
                  }}
                >
                  <span className="cluster-legend-dot" style={{ background: clusterColor(cluster.clusterId) }} />
                  {cluster.clusterId < 0 ? '未归类' : cluster.label}（{cluster.size}）
                </span>
              ))
            )}
            {result.clusters.length > clusterRows.length && (
              <span className="small muted">还有 {result.clusters.length - clusterRows.length} 个簇未列出</span>
            )}
          </div>

          {itemsError && <p className="clustering-notice is-error">明细读取失败：{itemsError}</p>}
          {items?.truncated && (
            <p className="clustering-notice">明细结果被服务端截断，仅展示已扫描到的部分。</p>
          )}

          {items && items.items.length === 0 ? (
            <p className="small muted">本次作业没有可展示的明细条目。</p>
          ) : (
            <ul className="clustering-job-items">
              {(items?.items ?? []).map((item) => (
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
          )}

          <div className="clustering-pager">
            <button
              className="btn btn-outline btn-sm"
              disabled={itemsBusy || currentPage <= 0}
              onClick={() => setPage(currentPage - 1)}
            >
              上一页
            </button>
            <span className="small muted">
              第 {currentPage + 1} / {pageCount} 页 · 共 {total} 条
              {itemsBusy ? ' · 加载中…' : ''}
            </span>
            <button
              className="btn btn-outline btn-sm"
              disabled={itemsBusy || currentPage >= pageCount - 1}
              onClick={() => setPage(currentPage + 1)}
            >
              下一页
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * 基准指标区。
 *
 * 三种来源必须泾渭分明：
 * 1. 本次算出（`summary.metrics`）—— 主运行 +（可选）净化关对照 + 差值；
 * 2. 本次没算、但作业是整份测试集口径 —— 回退到基准结果，并**明说不是本次实测**；
 * 3. 本次没算、也不是全量 —— 说明原因（数据集缺类别标签），不硬凑数字。
 */
function JobBenchmark({ job, summary }: { job: ClusteringJobInfo; summary: ClusteringSummary }) {
  const computed = hasComputedMetrics(summary);
  const [baseline, setBaseline] = useState<ClusteringOfflineBaseline | null>(null);
  const [baselineError, setBaselineError] = useState('');
  /** 基准结果读取失败后的重试计数。 */
  const [retry, setRetry] = useState(0);

  // 只有没算出指标时才需要基准结果：算出来的实测值永远优先
  useEffect(() => {
    if (computed) return;
    let cancelled = false;
    setBaselineError('');
    void fetchClusteringBaseline()
      .then((data) => {
        if (!cancelled) setBaseline(data);
      })
      .catch((cause) => {
        if (!cancelled) setBaselineError((cause as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [computed, retry]);

  if (computed) {
    const main = metricRows(summary.metrics);
    const control = new Map(metricRows(summary.controlMetrics).map((row) => [row.key, row]));
    const deltas = new Map(comparisonRows(summary.metricsComparison).map((row) => [row.key, row]));
    const withControl = hasControlMetrics(summary);
    return (
      <div className="clustering-benchmark">
        <div className="clustering-toolbar">
          <h3 className="clustering-table-title">基准指标（外部指标）</h3>
          <span className="small muted">{describeGroundTruth(summary.groundTruth)}</span>
        </div>
        <div className="clustering-table-scroll">
          <table className="table clustering-table">
            <thead>
              <tr>
                <th>指标</th>
                <th>主运行{summary.purification?.enabled ? '（净化开）' : ''}</th>
                {withControl && <th>对照（净化关）</th>}
                {withControl && <th>差值（主 − 对照）</th>}
              </tr>
            </thead>
            <tbody>
              {main.map((row) => (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td>{row.display}</td>
                  {withControl && <td>{control.get(row.key)?.display ?? '—'}</td>}
                  {withControl && <td>{deltas.get(row.key)?.display ?? '—'}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {withControl && (
          <p className="small muted">
            主运行与对照仅差「净化开关」，其余选项完全一致；差值方向为「主运行 − 对照」。
          </p>
        )}
        {summary.metrics?.score === -1 && (
          <p className="clustering-notice">
            这次只得到一个有效簇，成对指标无法计算（后端返回哨兵值 -1），请调大样本量或更换分析配置。
          </p>
        )}
      </div>
    );
  }

  if (baseline && shouldUseArchivedBenchmark(job, baseline, summary)) {
    const columns = archivedBenchmarkColumns(baseline);
    return (
      <div className="clustering-benchmark">
        <div className="clustering-toolbar">
          <h3 className="clustering-table-title">基准指标（外部指标）</h3>
          <span className="small muted">本作业未计算指标，下为基准结果对照</span>
        </div>
        <div className="clustering-table-scroll">
          <table className="table clustering-table">
            <thead>
              <tr>
                <th>指标</th>
                {columns.map((column) => (
                  <th key={column.key}>{archivedColumnLabel(column.key)}</th>
                ))}
                <th>目标 − 基线</th>
              </tr>
            </thead>
            <tbody>
              {METRIC_DEFINITIONS.map(({ key, label }) => (
                <tr key={String(key)}>
                  <td>{label}</td>
                  {columns.map((column) => (
                    <td key={column.key}>{formatMetricValue(column.row[key] ?? undefined)}</td>
                  ))}
                  <td>{formatDelta(String(key), baseline.comparison?.[String(key)])}</td>
                </tr>
              ))}
              <tr>
                <td>聚类数</td>
                {columns.map((column) => (
                  <td key={column.key}>{column.row.nClusters ?? '—'}</td>
                ))}
                <td>{formatDelta('nClusters', baseline.comparison?.nClusters)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="clustering-benchmark">
      <div className="clustering-toolbar">
        <h3 className="clustering-table-title">基准指标（外部指标）</h3>
      </div>
      {baselineError && (
        <p className="clustering-notice">
          基准结果暂不可用。
          <button className="btn btn-outline btn-sm" onClick={() => setRetry((value) => value + 1)}>
            <IconRefresh size={14} />重试
          </button>
        </p>
      )}
      <p className="small muted">
        本次作业没有计算出 ARI / VM / FMS / AMI / HS / CS：数据集里没有可用于比对真值的类别标签列
        （如 <code>category</code> / <code>label</code>）。上传带标签的数据后重新提交，并在运行选项里勾选
        「同时跑未净化对照」，即可在这里看到 6 项指标与差值。
      </p>
    </div>
  );
}
