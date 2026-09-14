/**
 * 聚类展示（演示/汇报视角）。
 *
 * 目标：一键跑通真实聚类，并把结果以「散点图 + 簇卡片 + 明细表」三层结构呈现。
 * 计算全部由 Python 聚类服务完成（算法来自 cluster-engine），本页只负责
 * 取数、渲染与交互，不在前端复现任何聚类逻辑。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  ClusteringClusterGroup,
  ClusteringData,
  ClusteringDatasetItem,
  ClusteringOfflineBaseline,
  ClusteringResultItem,
} from '../../shared/clustering';
import {
  fetchClusteringBaseline,
  fetchClusteringSample,
  runClustering,
  uploadClusteringDataset,
} from '../api/clustering';
import { ClusterFilterBar } from '../components/ClusterFilterBar';
import { ClusterItemDetail } from '../components/ClusterItemDetail';
import { ClusterScatter } from '../components/ClusterScatter';
import { ClusteringJobPanel } from '../components/ClusteringJobPanel';
import { FilterHeader, KeywordCells, KeywordSearch, OptionList, SortHeader } from '../components/TableControls';
import { IconLayers, IconPlay, IconRefresh, IconUpload } from '../components/icons';
import { clusterColor, clusterTint } from '../lib/clusterColor';
import { useClusterEngine } from '../lib/useClusterEngine';
import { useTableQuery } from '../lib/useTableQuery';
import './Clustering.css';

const PAGE_SIZE = 20;

interface LoadedDataset {
  name: string;
  items: ClusteringDatasetItem[];
  warnings: string[];
  /** 后端保存的数据集引用；有它时提交作业只需传这个 ID，不必重传样本。 */
  datasetId?: string | null;
}

export function ClusteringOverview() {
  const engine = useClusterEngine();
  const [dataset, setDataset] = useState<LoadedDataset | null>(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [algorithm, setAlgorithm] = useState('');
  const [profileId, setProfileId] = useState('');
  const [visualize, setVisualize] = useState(true);
  const [result, setResult] = useState<ClusteringData | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState('');
  /**
   * 净化开关：`profile` 表示完全按 profile 执行，`on`/`off` 是现场对照实验。
   * 只在所选 profile 的实现版本带净化能力时才有意义（见 selectedProfile）。
   */
  const [purifyMode, setPurifyMode] = useState<'profile' | 'on' | 'off'>('profile');
  const [baseline, setBaseline] = useState<ClusteringOfflineBaseline | null>(null);
  const [baselineBusy, setBaselineBusy] = useState(false);
  const [baselineError, setBaselineError] = useState('');
  const [activeClusterId, setActiveClusterId] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  /**
   * 明细表的筛选/排序/搜索。「所属簇筛选条」的选中态也由本 hook 持有，
   * 保证它与表头筛选走同一条流水线（原先页面另存一份 clusterFilter）。
   */
  const table = useTableQuery(result?.items, result?.clusters);
  const { setRowClusterFilter } = table;

  // 首次进入直接带上内置示例数据，保证页面「打开即可点开始」。
  const loadSample = useCallback(async () => {
    setDatasetBusy(true);
    setDatasetError('');
    try {
      const sample = await fetchClusteringSample();
      setDataset({ name: sample.sourceName, items: sample.items, warnings: [] });
      setResult(null);
      setActiveClusterId(null);
      setSelectedId(null);
      setRowClusterFilter([]);
      setPage(1);
    } catch (cause) {
      setDatasetError((cause as Error).message);
    } finally {
      setDatasetBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadSample();
  }, [loadSample]);

  const availableAlgorithms = useMemo(
    () => engine.algorithms.filter((item) => item.available),
    [engine.algorithms],
  );

  /** 当前算法下的可用 profile；未选算法时列出全部可用 profile。 */
  const selectableProfiles = useMemo(
    () => engine.profiles.filter((profile) => profile.available && (!algorithm || profile.algorithm === algorithm)),
    [engine.profiles, algorithm],
  );

  // 切换算法时，若已选 profile 不属于该算法则清空，交给后端自动挑选。
  useEffect(() => {
    if (profileId && !selectableProfiles.some((profile) => profile.profileId === profileId)) {
      setProfileId('');
    }
  }, [profileId, selectableProfiles]);

  const upload = useCallback(async (file: File) => {
    setDatasetBusy(true);
    setDatasetError('');
    try {
      const preview = await uploadClusteringDataset(file);
      setDataset({
        name: preview.sourceName,
        items: preview.items,
        warnings: preview.warnings,
        datasetId: preview.datasetId ?? null,
      });
      setResult(null);
      setActiveClusterId(null);
      setSelectedId(null);
      setRowClusterFilter([]);
      setPage(1);
    } catch (cause) {
      setDatasetError((cause as Error).message);
    } finally {
      setDatasetBusy(false);
    }
  }, []);

  /**
   * 本次执行允许的最小样本数。
   *
   * `semantic-v1` 允许单条（返回 `autoKStatus=singleton` 并如实标注低支持），
   * `legacy-v1` 要求 ≥2。门槛从所选 profile 下发的 `capability` 读取，前端不按
   * 算法名硬编码——否则新增策略时这里会先一步把用户拦住，而后端其实是支持的。
   * 未显式选择 profile 时按 2 处理：后端自动挑选的仍是 legacy 算法。
   */
  const selectedProfile = useMemo(
    () => selectableProfiles.find((profile) => profile.profileId === profileId),
    [profileId, selectableProfiles],
  );

  const minItems = useMemo(
    () => (selectedProfile?.capability?.supportsSingleItem ? 1 : 2),
    [selectedProfile],
  );

  /**
   * 所选 profile 是否带输入层语义净化（spear-v1）。
   * 能力从后端下发的 `capability.purification` 读取，前端不按 profile 名硬编码。
   */
  const purificationSupported = Boolean(selectedProfile?.capability?.purification);

  const purifyOverride = purifyMode === 'profile' ? undefined : purifyMode === 'on';

  const canRun = Boolean(dataset && dataset.items.length >= minItems) && !running && !datasetBusy;

  /** 加载离线基准结果：实时计算失败或时间不够时的兜底入口。 */
  const loadBaseline = useCallback(async () => {
    setBaselineBusy(true);
    setBaselineError('');
    try {
      setBaseline(await fetchClusteringBaseline());
    } catch (cause) {
      setBaselineError((cause as Error).message);
    } finally {
      setBaselineBusy(false);
    }
  }, []);

  const start = useCallback(async () => {
    if (!dataset || dataset.items.length < minItems) {
      setRunError(`至少需要 ${minItems} 条样本才能聚类。`);
      return;
    }
    setRunning(true);
    setRunError('');
    try {
      const next = await runClustering(dataset.items, {
        algorithm: algorithm || undefined,
        profileId: profileId || undefined,
        visualize,
        reduceMethod: visualize ? 'pca' : 'none',
        purify: purifyOverride,
      });
      setResult(next);
      setActiveClusterId(null);
      setSelectedId(null);
      setRowClusterFilter([]);
      setPage(1);
    } catch (cause) {
      setRunError((cause as Error).message);
    } finally {
      setRunning(false);
    }
  }, [dataset, algorithm, profileId, visualize, purifyOverride, minItems, setRowClusterFilter]);

  // 明细表过滤（含簇筛选/关键词搜索/表头筛选）与排序，全部由 useTableQuery 统一提供。
  const filteredItems = table.rows;

  const pageCount = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const visibleItems = filteredItems.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const selected = result?.items.find((item) => item.id === selectedId) ?? null;
  const highlighted = activeClusterId === null
    ? null
    : result?.clusters.find((cluster) => cluster.clusterId === activeClusterId) ?? null;

  const summary = result?.summary;

  /** 明细表副标题：按筛选与高亮状态给出人话描述。 */
  const scopeText = (() => {
    if (table.rowClusterFilter.length) {
      const labels = (result?.clusters ?? [])
        .filter((cluster) => table.rowClusterFilter.includes(cluster.clusterId))
        .map((cluster) => cluster.label);
      return `共 ${filteredItems.length} 条（已筛选：${labels.join('、') || `${table.rowClusterFilter.length} 个簇`}）`;
    }
    if (table.hasColumnFilter) return `共 ${filteredItems.length} 条（已按表头筛选/搜索）`;
    if (highlighted) return `共 ${filteredItems.length} 条（已高亮：${highlighted.label}）`;
    return `共 ${filteredItems.length} 条`;
  })();

  return (
    <div className="page clustering-page">
      <div className="page-head">
        <h1>聚类分析 · 聚类展示</h1>
        <div className="sub">
          对核电工程隐患文本执行无监督聚类，查看簇结构、代表样本与关键词。算法由本地 Python 聚类服务提供。
        </div>
      </div>

      {!engine.loading && !engine.ready && (
        <div className="card card-pad clustering-banner" role="alert">
          <strong>聚类服务未就绪</strong>
          <p className="small">
            {engine.error
              ? engine.error
              : engine.health?.engine.message ?? '聚类引擎尚未加载完成。'}
          </p>
          <p className="small muted">
            请在项目根目录执行 <code>npm run dev:py</code> 启动 Python 聚类服务；首次启动需要校验本地向量模型，约需数十秒。
          </p>
          <button className="btn btn-outline btn-sm" onClick={() => void engine.refresh()}>
            <IconRefresh size={14} />重新检查
          </button>
        </div>
      )}

      {(datasetError || runError) && (
        <div className="card card-pad clustering-banner is-error" role="alert">
          <strong>执行失败</strong>
          <p className="small">{datasetError || runError}</p>
        </div>
      )}

      <section className="card card-pad">
        <div className="clustering-toolbar">
          <div className="clustering-dataset">
            <strong>{dataset?.name ?? (datasetBusy ? '正在加载示例数据…' : '尚未选择数据')}</strong>
            {dataset && <div className="small muted">共 {dataset.items.length} 条样本</div>}
          </div>
          <button className="btn btn-outline" disabled={datasetBusy || running} onClick={() => void loadSample()}>
            使用示例数据
          </button>
          <label className={`btn btn-outline clustering-upload ${datasetBusy || running ? 'is-disabled' : ''}`}>
            <IconUpload size={15} />上传文件
            <input
              aria-label="上传聚类数据文件"
              type="file"
              accept=".csv,.json,.xlsx,.txt"
              disabled={datasetBusy || running}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = '';
                if (file) void upload(file);
              }}
            />
          </label>
          <button className="btn btn-primary" disabled={!canRun || !engine.ready} onClick={() => void start()}>
            <IconPlay size={15} />{running ? '聚类进行中…' : '开始聚类'}
          </button>
        </div>

        <div className="clustering-toolbar clustering-options">
          <label>
            算法
            <select
              className="select"
              value={algorithm}
              disabled={running || !engine.ready}
              onChange={(event) => setAlgorithm(event.target.value)}
            >
              <option value="">自动选择（推荐）</option>
              {availableAlgorithms.map((item) => (
                <option key={item.algorithm} value={item.algorithm}>{item.algorithm}</option>
              ))}
            </select>
          </label>
          <label>
            配置档 Profile
            <select
              className="select"
              value={profileId}
              disabled={running || !engine.ready}
              onChange={(event) => setProfileId(event.target.value)}
            >
              <option value="">自动选择</option>
              {selectableProfiles.map((profile) => (
                <option key={profile.profileId} value={profile.profileId}>
                  {profile.profileId}（{profile.algorithm}
                  {Number(profile.features.n_results ?? 0) > 0 ? ' · 检索增强' : ''}
                  {profile.capability?.supportsSingleItem ? ' · 支持单条' : ''}）
                </option>
              ))}
            </select>
          </label>
          <label className="clustering-check">
            <input type="checkbox" checked={visualize} disabled={running} onChange={(event) => setVisualize(event.target.checked)} />
            计算二维散点坐标（PCA，仅用于展示）
          </label>
          {/*
            净化开关只在所选 profile 属于 spear-v1 时出现。`profile` 是默认值：
            现场正常演示按 profile 口径执行，只有做对照实验才手动覆盖。
          */}
          {purificationSupported && (
            <label>
              语义净化
              <select
                className="select"
                value={purifyMode}
                disabled={running}
                onChange={(event) => setPurifyMode(event.target.value as 'profile' | 'on' | 'off')}
              >
                <option value="profile">按 profile（推荐）</option>
                <option value="on">强制开启</option>
                <option value="off">关闭（对照组）</option>
              </select>
            </label>
          )}
          <span className="small muted">
            {engine.health
              ? `引擎已加载 ${engine.health.engine.profilesTotal} 个 profile，其中 ${engine.health.engine.profilesAvailable} 个可执行`
              : engine.loading ? '正在检查聚类服务…' : '聚类服务不可用'}
          </span>
        </div>

        {dataset?.warnings.map((warning) => (
          <p key={warning} className="clustering-notice">{warning}</p>
        ))}
        <p className="small muted">
          支持 CSV / XLSX / JSON / TXT，自动识别文本列；最多 30 万条、200 MB。文本会做规范化清洗（全角转半角、去控制字符、合并空白）。
        </p>
        <p className="small muted">
          现场耗时预期：向量嵌入约 10 秒（全量 20,198 条）· 语义净化约 0.3–0.5 秒/条（开启且首条需等模型装载约 20 秒）·
          无净化全量端到端约 214 秒。样本量越大越建议改用下方异步作业面板。
        </p>
      </section>

      {/*
        离线基准结果：现场演示的兜底入口。
        归档数字与实时结果必须区分开，因此这里把 source / verifiedAt / fullRun
        原样展示，不做美化，也不与本次运行的统计混排。
      */}
      <section className="card card-pad" aria-label="离线基准结果">
        <div className="clustering-section-head">
          <h2>离线基准结果</h2>
          <button className="btn btn-outline btn-sm" disabled={baselineBusy} onClick={() => void loadBaseline()}>
            <IconRefresh size={14} />{baselineBusy ? '加载中…' : '加载离线基准结果'}
          </button>
        </div>
        <p className="small muted">
          实时计算失败或时间不足时的兜底：直接展示已归档的论文复现指标。归档结果是全量 20,198 条口径，
          与本次实时运行是两回事——出处与口径会显示在下方，不做混合。
        </p>
        {baselineError && <p className="clustering-notice">{baselineError}</p>}
        {baseline && (
          <>
            <p className="small muted">
              出处：{baseline.source}
              {baseline.verifiedAt ? ` · 记录于 ${baseline.verifiedAt}` : ''} ·{' '}
              {baseline.fullRun ? '全量口径' : '抽样口径（指标不具可比性）'}
            </p>
            <div className="clustering-table-scroll">
              <table className="table clustering-table">
                <thead>
                  <tr>
                    <th>配置</th>
                    <th>ARI</th>
                    <th>VM</th>
                    <th>聚类数</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(baseline.rows).map(([name, metrics]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{metrics.ari ?? '—'}</td>
                      <td>{metrics.vm ?? '—'}</td>
                      <td>{metrics.nClusters ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {baseline.comparison && baseline.comparison.ari !== undefined && (
              <p className="small">
                相对基线增益：ARI{' '}
                <strong>
                  {baseline.comparison.ari >= 0 ? '+' : ''}
                  {baseline.comparison.ari}
                </strong>
                {baseline.comparison.ariRelative !== undefined && (
                  <>（{(baseline.comparison.ariRelative * 100).toFixed(1)}%）</>
                )}
                {baseline.comparison.vm !== undefined && <> · VM {baseline.comparison.vm >= 0 ? '+' : ''}{baseline.comparison.vm}</>}
                {baseline.comparison.nClusters !== undefined && <> · 聚类数 {baseline.comparison.nClusters >= 0 ? '+' : ''}{baseline.comparison.nClusters}</>}
              </p>
            )}
            {(baseline.notes ?? []).map((note) => (
              <p key={note} className="small muted">{note}</p>
            ))}
          </>
        )}
      </section>

      {/*
        大数据量入口。同步接口会把整份明细放进一次 HTTP 响应并在请求上等满整个
        计算过程，样本到万级就不合适了；作业面板走提交/轮询/取消 + 服务端分页，
        因此这里与上面的「开始聚类」并存而不是替换它。
      */}
      <ClusteringJobPanel
        items={dataset?.items ?? []}
        datasetId={dataset?.datasetId ?? null}
        options={{
          algorithm: algorithm || undefined,
          profileId: profileId || undefined,
          visualize,
          reduceMethod: visualize ? 'pca' : 'none',
          purify: purifyOverride,
        }}
        disabled={!engine.ready || datasetBusy}
      />

      {summary && (
        <>
          <section className="card clustering-stat-grid" aria-label="聚类统计">
            {[
              ['样本总数', String(summary.totalSamples)],
              ['簇数量', String(summary.clusterCount)],
              ['噪声点', String(summary.noiseCount)],
              ['最大簇', String(summary.largestClusterSize)],
              ['最小簇', String(summary.smallestClusterSize)],
              ['平均簇大小', summary.avgClusterSize.toFixed(2)],
              // 语义路径专有口径：去重后的真实规模与覆盖率。
              // legacy 路径不返回这些字段，用 filter 整条剔除，保持原样。
              ...(summary.uniqueCount ? [['去重后唯一文本', String(summary.uniqueCount)] as [string, string]] : []),
              ...(summary.invalidCount ? [['无效文本', String(summary.invalidCount)] as [string, string]] : []),
              ...(summary.duplicateCount ? [['重复合并', String(summary.duplicateCount)] as [string, string]] : []),
              ...(summary.coverage !== undefined && summary.coverage !== null
                ? [['归入簇占比', `${(summary.coverage * 100).toFixed(1)}%`] as [string, string]]
                : []),
              ...(summary.qualityScore !== undefined && summary.qualityScore !== null
                ? [['簇质量分（宏平均）', summary.qualityScore.toFixed(3)] as [string, string]]
                : []),
              ...(summary.overallQuality !== undefined && summary.overallQuality !== null
                ? [['综合质量分', summary.overallQuality.toFixed(3)] as [string, string]]
                : []),
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </section>

          <section className="card card-pad clustering-run-meta">
            <p className="small muted">
              运行 ID：{summary.runId} · 算法：<strong>{summary.algorithm}</strong> · Profile：{summary.profileId} ·
              实现版本：{summary.implementationVersion} ·
              向量模型：{summary.modelId}（{summary.embeddingDimension} 维）·
              引擎耗时 {summary.elapsedMs} ms · 网关总耗时 {summary.gatewayMs} ms
              {summary.cacheHit ? ' · 命中向量缓存' : ''}
            </p>
            {/* Auto-K 与向量缓存的真实口径：语义路径下才有的字段 */}
            {(summary.selectedK !== undefined && summary.selectedK !== null) || summary.embeddingCacheMisses !== undefined ? (
              <p className="small muted">
                {summary.selectedK !== undefined && summary.selectedK !== null && (
                  <>
                    Auto-K：候选选出 K={summary.selectedK} → 后处理 {summary.finalK} 个簇（{summary.autoKStatus ?? '—'}）·
                  </>
                )}
                {summary.embeddingCacheMisses !== undefined && summary.embeddingCacheMisses !== null && (
                  <>
                    向量缓存：命中 {summary.embeddingCacheHits ?? 0} / 未命中 {summary.embeddingCacheMisses}
                    {summary.cacheOnly ? '（未加载模型，全部复用缓存）' : ''}
                    {summary.device ? ` · 设备 ${summary.device}` : ''}
                  </>
                )}
              </p>
            ) : null}
            {summary.calibrationVersion && (
              <p className="clustering-notice">
                阈值校准：{summary.calibrationVersion}
                {summary.calibrationStatus === 'validated' ? '（已校准）' : '（尚未经人工校准验证，结果仅供探索）'}。
                阈值与权重不在 profile 里，如需调整请改校准配置后重新运行。
              </p>
            )}
            {summary.warnings.map((warning) => (
              <p key={warning} className="clustering-notice">{warning}</p>
            ))}
            {(summary.clusterCount <= 1 || summary.noiseCount === summary.totalSamples) && (
              <p className="clustering-notice">
                {summary.noiseCount === summary.totalSamples
                  ? summary.implementationVersion === 'semantic-v1'
                    ? '本次所有样本都未被归入任何簇：说明没有任何候选划分同时满足「成员语义支持」与「小组自洽」门槛。'
                    : '本次所有样本都被判定为噪声，说明密度类算法在当前数据上的阈值过严。'
                  : '本次只得到 1 个簇，说明当前算法的默认参数把这批文本并成了一类。'}
                {summary.implementationVersion === 'semantic-v1'
                  ? '可以换用其他 profile 对比，或检查文本是否本身差异过大（主题数远多于样本数时必然难以成簇）。'
                  : '不同算法对同一批数据的粒度差异很大，建议在「算法」下拉里换一个再试（例如 leader / birch / canopy），或到「聚类测试」页做多算法横向对比。'}
              </p>
            )}
          </section>

          {/*
            净化前后对照：演示里最有说服力的一屏。
            这里展示的是引擎真实产出的净化结果（不是前端模拟），因此同时把
            后端、是否降级、护栏拦截条数一并显示——"用的什么净化"本身就是结论的一部分。
          */}
          {summary.purification && (
            <section className="card card-pad" aria-label="语义净化">
              <div className="clustering-section-head">
                <h2>阶段 1 · 语义净化</h2>
                <span className="small muted">
                  净化只做剥离与压缩，不判定缺陷类别；否定词等极性表述由确定性护栏保证不丢失。
                </span>
              </div>
              <p className="small muted">
                后端：<strong>{summary.purification.backend}</strong>
                {summary.purification.requestedBackend !== summary.purification.backend
                  ? `（请求 ${summary.purification.requestedBackend}）`
                  : ''}{' '}
                · 护栏：
                {summary.purification.guarded
                  ? `开启，拦截 ${summary.purification.guardHits}/${summary.purification.total} 条`
                  : '关闭'}{' '}
                · 耗时 {summary.purification.elapsedMs} ms
              </p>
              {summary.purification.degraded && (
                <p className="clustering-notice">
                  净化已降级为规则兜底（原因：{summary.purification.reason ?? '未知'}）。
                  降级不影响结果可复现，但净化质量低于论文口径，演示时应如实说明。
                </p>
              )}
              {summary.purification.samples.length ? (
                <div className="clustering-table-scroll">
                  <table className="table clustering-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>原始文本</th>
                        <th>浓缩文本</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.purification.samples.map((sample) => (
                        <tr key={sample.id}>
                          <td>{sample.id}</td>
                          <td>{sample.raw}</td>
                          <td>{sample.condensed}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="small muted">本次运行没有返回净化对照样本（关闭净化时属正常）。</p>
              )}
            </section>
          )}

          <section className="card card-pad">
            <div className="clustering-section-head">
              <h2>二维分布</h2>
              <span className="small muted">点击图例或散点可高亮某个簇；明细表的筛选条按所属簇过滤样本。</span>
            </div>
            {summary.implementationVersion === 'spear-v1' && (result?.visualization?.length ?? 0) === 0 && (
              <p className="clustering-notice">
                spear-v1 的聚类空间来自净化后的文本，网关不复现该空间，因此本路径不生成二维坐标；
                簇结构与逐条归属不受影响。需要散点图请用 legacy / semantic profile。
              </p>
            )}
            <ClusterScatter
              points={result?.visualization ?? []}
              clusters={result?.clusters ?? []}
              items={result?.items ?? []}
              activeClusterId={activeClusterId}
              onPickCluster={(clusterId) => {
                setActiveClusterId(clusterId);
                setPage(1);
              }}
              onPickPoint={(pointId) => setSelectedId(pointId)}
            />
          </section>

          <section className="clustering-cluster-grid">
            {(result?.clusters ?? []).map((cluster) => (
              <ClusterCard
                key={cluster.clusterId}
                cluster={cluster}
                active={activeClusterId === cluster.clusterId}
                onToggle={() => {
                  setActiveClusterId(activeClusterId === cluster.clusterId ? null : cluster.clusterId);
                  setPage(1);
                }}
                onPickItem={(id) => setSelectedId(id)}
              />
            ))}
          </section>

          <section className="card">
            <div className="clustering-toolbar card-pad">
              <h2 className="clustering-table-title">样本归属明细</h2>
              <span className="small muted">{scopeText}</span>
              {highlighted && (
                <button className="btn btn-outline btn-sm" onClick={() => { setActiveClusterId(null); setPage(1); }}>
                  取消高亮
                </button>
              )}
            </div>
            <ClusterFilterBar
              clusters={result?.clusters ?? []}
              selected={table.rowClusterFilter}
              onChange={(next) => { setRowClusterFilter(next); setPage(1); }}
              matchCount={filteredItems.length}
              totalCount={result?.items.length ?? 0}
            />
            {/* 关键词列的表内搜索：自由检索，与表头下拉筛选互补 */}
            <div className="kw-search-bar">
              <KeywordSearch
                value={table.keywordSearch}
                onChange={(next) => { table.setKeywordSearch(next); setPage(1); }}
                suggestions={table.keywordOptions.slice(0, 6).map((option) => option.value)}
              />
              <span className="small muted">
                筛选后 {filteredItems.length} / {result?.items.length ?? 0} 条
              </span>
              {table.hasColumnFilter && (
                <button
                  type="button"
                  className="btn btn-outline btn-sm"
                  onClick={() => { table.clearColumnFilters(); setPage(1); }}
                >
                  清除筛选
                </button>
              )}
            </div>
            <div className="clustering-table-scroll">
              <table className="table clustering-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>文本</th>
                    <FilterHeader label="所属簇" selectedCount={table.clusterColumnFilter.length}>
                      {(close) => (
                        <OptionList
                          options={table.clusterOptions.map((option) => ({
                            key: String(option.clusterId),
                            label: option.label,
                            count: option.size,
                            color: clusterColor(option.clusterId),
                          }))}
                          selected={table.clusterColumnFilter.map(String)}
                          onToggle={(key) => { table.toggleClusterColumnFilter(Number(key)); setPage(1); }}
                          onClear={() => {
                            for (const id of [...table.clusterColumnFilter]) table.toggleClusterColumnFilter(id);
                            close();
                          }}
                          emptyText="本次运行没有簇。"
                        />
                      )}
                    </FilterHeader>
                    <SortHeader
                      label="置信度"
                      sortKey="confidence"
                      direction={table.sortDirectionOf('confidence')}
                      onToggle={(key) => { table.toggleSort(key); setPage(1); }}
                      hint="按置信度排序（降序 = 置信度高在前）"
                    />
                    <SortHeader
                      label="到质心距离"
                      sortKey="distance"
                      direction={table.sortDirectionOf('distance')}
                      onToggle={(key) => { table.toggleSort(key); setPage(1); }}
                      hint="按到质心距离排序（升序 = 离质心近在前）"
                    />
                    <FilterHeader label="关键词" selectedCount={table.keywordColumnFilter.length}>
                      {(close) => (
                        <OptionList
                          options={table.keywordOptions.map((option) => ({
                            key: option.value,
                            label: option.value,
                            count: option.count,
                          }))}
                          selected={table.keywordColumnFilter}
                          onToggle={(keyword) => { table.toggleKeywordColumnFilter(keyword); setPage(1); }}
                          onClear={() => {
                            for (const keyword of [...table.keywordColumnFilter]) table.toggleKeywordColumnFilter(keyword);
                            close();
                          }}
                          emptyText="没有匹配的关键词。"
                          searchPlaceholder="过滤关键词…"
                        />
                      )}
                    </FilterHeader>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleItems.map((item) => (
                    <tr key={item.id} onClick={() => setSelectedId(item.id)}>
                      <td className="clustering-id">{item.id}</td>
                      <td><div className="clustering-text" title={item.text}>{item.text}</div></td>
                      <td>
                        <span className="cluster-pill" style={{ background: clusterTint(item.clusterId, 0.14), color: clusterColor(item.clusterId) }}>
                          <span className="cluster-legend-dot" style={{ background: clusterColor(item.clusterId) }} />
                          {item.clusterLabel}
                        </span>
                      </td>
                      <td>{item.confidence === null || item.confidence === undefined ? '—' : item.confidence.toFixed(3)}</td>
                      <td>{item.distance === null || item.distance === undefined ? '—' : item.distance.toFixed(3)}</td>
                      <td className="clustering-keywords">
                        <KeywordCells keywords={item.keywords} search={table.keywordSearch} />
                      </td>
                      <td>
                        <button className="btn btn-outline btn-sm" onClick={(event) => { event.stopPropagation(); setSelectedId(item.id); }}>
                          详情
                        </button>
                      </td>
                    </tr>
                  ))}
                  {!visibleItems.length && (
                    <tr><td colSpan={7} className="clustering-empty">没有符合当前过滤条件的样本。</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="clustering-pagination">
              <button className="btn btn-outline btn-sm" disabled={currentPage <= 1} onClick={() => setPage(currentPage - 1)}>上一页</button>
              <span>{currentPage} / {pageCount}</span>
              <button className="btn btn-outline btn-sm" disabled={currentPage >= pageCount} onClick={() => setPage(currentPage + 1)}>下一页</button>
            </div>
          </section>
        </>
      )}

      {!summary && dataset && engine.ready && (
        <section className="card clustering-placeholder">
          <IconLayers size={26} />
          <p>数据已就绪（{dataset.items.length} 条样本），点击「开始聚类」查看簇结构与二维分布。</p>
        </section>
      )}

      {selected && summary && (
        <ClusterItemDetail item={selected} summary={summary} onClose={() => setSelectedId(null)} />
      )}
    </div>
  );
}

/** 单个簇的摘要卡片：关键词 + 元数据分布 + 代表样本。 */
function ClusterCard({
  cluster,
  active,
  onToggle,
  onPickItem,
}: {
  cluster: ClusteringClusterGroup;
  active: boolean;
  onToggle: () => void;
  onPickItem: (id: string) => void;
}) {
  const isNoise = cluster.clusterId < 0;
  const title = cluster.clusterName ?? cluster.label;
  const distributions = Object.entries(cluster.metadataDistribution).slice(0, 3);
  return (
    <article
      className={`card card-pad cluster-card ${active ? 'is-active' : ''}`}
      style={{ borderColor: active ? clusterColor(cluster.clusterId) : undefined }}
      role="button"
      tabIndex={0}
      aria-pressed={active}
      aria-label={`${active ? '取消高亮' : '高亮'} ${title}（${cluster.size} 条）`}
      onClick={onToggle}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onToggle();
        }
      }}
    >
      <header className="cluster-card-head">
        <span className="cluster-legend-dot" style={{ background: clusterColor(cluster.clusterId) }} />
        <h3>{title}</h3>
        <span className="cluster-card-size">
          {cluster.size} 条
          {cluster.percentage !== undefined && cluster.percentage !== null && ` · ${cluster.percentage.toFixed(1)}%`}
        </span>
      </header>
      {isNoise && (
        <p className="small muted">
          该组为未归类样本（噪声 / 无效文本），不是一个语义簇：没有质心，也没有质量分。
        </p>
      )}
      {/* 语义路径的质量分与去重口径；legacy 路径下这些字段为 undefined，整段不渲染 */}
      {cluster.qualityScore !== undefined && cluster.qualityScore !== null && (
        <p className="small muted">
          簇质量分：{cluster.qualityScore.toFixed(3)}
          {cluster.qualityStatus ? `（${cluster.qualityStatus}）` : ''}
          {cluster.uniqueSize !== undefined && cluster.uniqueSize !== null && ` · 去重后 ${cluster.uniqueSize} 条不同文本`}
        </p>
      )}
      {cluster.cohesion !== null && cluster.cohesion !== undefined && (
        <p className="small muted">簇内平均余弦相似度：{cluster.cohesion.toFixed(3)}</p>
      )}
      {cluster.nameSource && cluster.nameEvidence && (
        <p className="small muted" title={cluster.nameEvidence}>
          命名依据：原文片段（{cluster.nameSource}）
        </p>
      )}
      {cluster.keywords.length > 0 && (
        <div className="cluster-card-keywords">
          {cluster.keywords.map((word) => <span key={word} className="chip cluster-keyword">{word}</span>)}
        </div>
      )}
      {distributions.length > 0 && (
        <div className="cluster-card-meta">
          {distributions.map(([key, counts]) => (
            <p key={key} className="small">
              <span className="muted">{key}：</span>
              {counts.slice(0, 3).map((entry) => `${entry.value}(${entry.count})`).join(' · ')}
            </p>
          ))}
        </div>
      )}
      {cluster.representativeSamples.length > 0 && (
        <div className="cluster-card-samples">
          <span className="small muted">代表样本</span>
          {cluster.representativeSamples.slice(0, 3).map((sample) => (
            <button
              key={sample.id}
              className="cluster-sample"
              onClick={(event) => { event.stopPropagation(); onPickItem(sample.id); }}
              title={sample.text}
            >
              {sample.text}
            </button>
          ))}
        </div>
      )}
    </article>
  );
}
