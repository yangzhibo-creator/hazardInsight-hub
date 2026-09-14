/**
 * 偏差数据库：检索增强聚类所用参考库的查看、生成与追溯。
 *
 * 页面收敛成三块，各司其职（此前它们混在一起，看不出"该先做什么"）：
 * ① 概览条 —— 现在有几个库、共多少条、哪些 profile 默认用哪个库；
 * ② 生成新库 —— 选语料 → 定名称与方式 → 提交，**按钮就在表单最后一步**，
 *    进度条也长在同一张卡片下方，不必满页找"我刚提交的任务呢"；
 * ③ 已有库 —— 可搜索的清单，标明 profile 默认库，点开抽屉看条目内容。
 *
 * 异步生成的作业在服务端排队。页面刷新后按 `/knowledge-bases/jobs` 自动接回
 * 未完成的任务——jobId 不该只活在浏览器内存里，否则"刷新即失联"。
 *
 * 刻意不提供删除：知识库是现场演示资产，误删一次就得重跑几小时净化。
 * 需要清理走运维命令，页面上不做一键危险操作。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  ClusteringKnowledgeBaseDetail,
  ClusteringKnowledgeBaseInfo,
  ClusteringKnowledgeBaseJob,
} from '../../shared/clustering';
import {
  buildKnowledgeBase,
  fetchKnowledgeBaseDetail,
  fetchKnowledgeBaseJob,
  fetchKnowledgeBaseJobs,
  fetchKnowledgeBases,
} from '../api/clustering';
import {
  IconAlert,
  IconCheckCircle,
  IconDatabase,
  IconFileText,
  IconRefresh,
  IconSearch,
  IconUpload,
  IconX,
} from '../components/icons';
import { knowledgeBaseName } from '../components/KnowledgeBasePicker';
import { useClusterEngine } from '../lib/useClusterEngine';
// 抽屉里复用聚类页的 `.clustering-id` 等排版样式，避免再抄一份等宽小字
import './Clustering.css';
import './KnowledgeBase.css';

const MODE_LABEL: Record<string, string> = {
  purified: '大模型净化',
  raw: '原文直接入库',
};

const STAGE_LABEL: Record<string, string> = {
  queued: '排队中',
  purify: '语义净化',
  embed: '向量化',
  index: '写入索引',
  done: '已完成',
  failed: '已失败',
};

const SOURCE_LABEL: Record<string, string> = {
  snapshot: '构建时保留的索引文本快照',
  corpus: '上传语料原文快照',
  source: '按来源指纹匹配到的历史语料',
  unavailable: '无文本快照，仅元信息',
};

const MODE_OPTIONS = [
  {
    value: 'purified',
    label: '大模型净化后入库',
    hint: '逐条剥离噪声、统一表述，检索质量更高；两万条约 1–3 小时',
  },
  {
    value: 'raw',
    label: '原文直接入库',
    hint: '不调用大模型，通常几分钟完成；适合语料本身已清洗干净',
  },
] as const;

/** 语料文件允许的扩展名；与后端 `validate_upload_extension` 的口径保持一致。 */
const ACCEPTED_EXTENSIONS = ['.txt', '.csv', '.xlsx', '.json'];

function formatBytes(value: number): string {
  if (!value) return '—';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatTime(value?: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

/** 文件扩展名（含点，小写）；没有扩展名时返回空串。 */
function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

/** 状态点/文字的语义色：完成=成功，失败=危险，其余=进行中。 */
function statusTone(status: string): 'ok' | 'bad' | 'busy' {
  if (status === 'succeeded' || status === 'completed') return 'ok';
  if (status === 'failed') return 'bad';
  return 'busy';
}

export function KnowledgeBase() {
  const engine = useClusterEngine();

  const [items, setItems] = useState<ClusteringKnowledgeBaseInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');

  const [detail, setDetail] = useState<ClusteringKnowledgeBaseDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [name, setName] = useState('');
  const [mode, setMode] = useState<'purified' | 'raw'>('purified');
  const [ident, setIdent] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  const [job, setJob] = useState<ClusteringKnowledgeBaseJob | null>(null);
  const [jobs, setJobs] = useState<ClusteringKnowledgeBaseJob[]>([]);

  const fileInput = useRef<HTMLInputElement>(null);
  /**
   * 用户主动收起过的任务。
   * 没有它的话，"刷新后自动接回未完成任务"会在收起的下一秒把它重新弹出来——
   * 自动恢复与用户意图打架，界面会变得不可控。
   */
  const dismissedJob = useRef<string | null>(null);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError('');
    try {
      setItems(await fetchKnowledgeBases());
    } catch (cause) {
      setItems([]);
      setError((cause as Error).message);
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await fetchKnowledgeBaseJobs(8));
    } catch {
      // 任务历史是只读的追溯信息：拿不到不该让整页不可用
      setJobs([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshJobs();
  }, [refresh, refreshJobs]);

  // 刷新后自动接回未完成的生成任务：作业在服务端继续跑，页面不该"刷新即失联"。
  // 但用户主动收起过的任务不在此列（见 dismissedJob）。
  useEffect(() => {
    if (job) return;
    const active = jobs.find((item) => !item.terminal);
    if (active && active.jobId !== dismissedJob.current) setJob(active);
  }, [jobs, job]);

  // 生成作业轮询：终态即停；完成后刷新清单与任务历史，让新库立刻可选。
  useEffect(() => {
    if (!job || job.terminal) return;
    const timer = window.setInterval(() => {
      void fetchKnowledgeBaseJob(job.jobId)
        .then((next) => {
          setJob(next);
          if (next.terminal) {
            void refresh(true);
            void refreshJobs();
          }
        })
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job, refresh, refreshJobs]);

  /** profile 默认库 → 使用它的 profile 列表（一个库可能被多个 profile 引用）。 */
  const defaultKnowledgeBases = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const profile of engine.profiles) {
      const id = profile.knowledgeBaseId;
      if (!id) continue;
      map.set(id, [...(map.get(id) ?? []), profile.profileId]);
    }
    return map;
  }, [engine.profiles]);

  const filteredItems = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return items;
    return items.filter((item) =>
      [item.displayName, item.knowledgeBaseId, item.sourceName ?? '']
        .some((field) => field.toLowerCase().includes(keyword)),
    );
  }, [items, query]);

  const totalEntries = useMemo(
    () => items.reduce((sum, item) => sum + (item.entryCount || 0), 0),
    [items],
  );

  const defaultSummary = useMemo(() => {
    if (!engine.ready) return engine.loading ? '检查中…' : '引擎未就绪';
    if (defaultKnowledgeBases.size === 0) return '未配置';
    return [...defaultKnowledgeBases.keys()]
      .map((id) => knowledgeBaseName(items, id))
      .join('、');
  }, [engine.ready, engine.loading, defaultKnowledgeBases, items]);

  const acceptFile = useCallback((next: File) => {
    const ext = extensionOf(next.name);
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setFileError(`不支持的文件类型「${ext || '未知'}」；请上传 ${ACCEPTED_EXTENSIONS.join(' / ')}。`);
      return;
    }
    setFileError('');
    setFile(next);
    // 名称默认取文件名（去掉扩展名）：省一步输入，也避免忘记命名。
    setName((current) => current.trim() || next.name.slice(0, next.name.length - ext.length));
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    setFileError('');
    if (fileInput.current) fileInput.current.value = '';
  }, []);

  const submit = useCallback(async () => {
    if (!file) {
      setFileError('请先选择语料文件。');
      return;
    }
    setSubmitting(true);
    setFormError('');
    try {
      const created = await buildKnowledgeBase(file, {
        name: name.trim() || file.name,
        mode,
        ident: ident.trim() || null,
      });
      dismissedJob.current = null;
      setJob(created);
      clearFile();
      setName('');
      setIdent('');
      void refreshJobs();
    } catch (cause) {
      setFormError((cause as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [file, name, mode, ident, clearFile, refreshJobs]);

  const openDetail = useCallback(async (knowledgeBaseId: string) => {
    setDetailLoading(true);
    try {
      setDetail(await fetchKnowledgeBaseDetail(knowledgeBaseId, 0, 50));
    } catch (cause) {
      // 详情读不到也要给出结构完整的抽屉：把失败原因如实写在 warnings 里，
      // 而不是让整个抽屉空白或直接关掉。
      setDetail({
        knowledgeBaseId,
        displayName: knowledgeBaseId,
        status: 'unavailable',
        verification: 'unknown',
        processing: 'unknown',
        entryCount: 0,
        dimension: 0,
        metric: '',
        modelFingerprint: '',
        sourceSha256: '',
        sizeBytes: 0,
        hasEntries: false,
        hasCorpus: false,
        entriesSource: 'unavailable',
        entryTotal: 0,
        offset: 0,
        limit: 50,
        entries: [],
        warnings: [(cause as Error).message],
      });
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const loadMoreEntries = useCallback(async () => {
    if (!detail) return;
    try {
      const next = await fetchKnowledgeBaseDetail(detail.knowledgeBaseId, detail.entries.length, 50);
      setDetail({ ...next, entries: [...detail.entries, ...next.entries] });
    } catch (cause) {
      setDetail({ ...detail, warnings: [...detail.warnings, (cause as Error).message] });
    }
  }, [detail]);

  const progress = useMemo(() => {
    if (!job || !job.total) return 0;
    return Math.min(100, Math.round((job.processed / job.total) * 100));
  }, [job]);

  return (
    <div className="page kb-page">
      <div className="page-head">
        <h1>偏差数据库</h1>
        <div className="sub">
          检索增强聚类（SPEAR）使用的语义知识库：上传语料生成新库、查看库内条目。
          具体用哪个库，在「聚类展示 / 聚类测试」页的「参考数据库」下拉里选择。
        </div>
      </div>

      {/* ---------- 概览：先回答"现在有什么" ---------- */}
      <section className="card kb-overview" aria-label="偏差数据库概览">
        <div className="kb-overview-item">
          <span>已有知识库</span>
          <strong>{loading ? '—' : items.length.toLocaleString('zh-CN')}</strong>
        </div>
        <div className="kb-overview-item">
          <span>条目合计</span>
          <strong>{loading ? '—' : totalEntries.toLocaleString('zh-CN')}</strong>
        </div>
        <div className="kb-overview-item">
          <span>生成中</span>
          <strong>{jobs.filter((item) => !item.terminal).length}</strong>
        </div>
        <div className="kb-overview-item is-wide">
          <span>profile 默认参考库</span>
          <strong title={defaultSummary}>{defaultSummary}</strong>
        </div>
      </section>

      {/* ---------- 生成新库：表单与提交在同一张卡片里 ---------- */}
      <section className="card kb-builder" aria-label="生成偏差数据库">
        <header className="kb-card-head">
          <div>
            <h2>生成新库</h2>
            <p className="small muted">上传一份语料，净化 / 向量化后写入检索索引。</p>
          </div>
          <span className="kb-card-tag">异步任务</span>
        </header>

        <div className="kb-steps">
          <div className="kb-step">
            <span className="kb-step-index">1</span>
            <div className="kb-step-body">
              <div className="kb-step-title">选择语料文件</div>
              <div
                className={`kb-drop ${dragging ? 'is-drag' : ''} ${file ? 'has-file' : ''}`}
                role="button"
                tabIndex={0}
                aria-label="选择或拖拽语料文件"
                onClick={() => fileInput.current?.click()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    fileInput.current?.click();
                  }
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  const dropped = event.dataTransfer.files?.[0];
                  if (dropped) acceptFile(dropped);
                }}
              >
                <input
                  ref={fileInput}
                  type="file"
                  accept={ACCEPTED_EXTENSIONS.join(',')}
                  hidden
                  onChange={(event) => {
                    const picked = event.target.files?.[0];
                    if (picked) acceptFile(picked);
                  }}
                />
                {file ? <IconFileText size={20} /> : <IconUpload size={20} />}
                <div className="kb-drop-text">
                  <strong>{file ? file.name : '点击选择或拖拽语料文件到这里'}</strong>
                  <span className="small muted">
                    {file
                      ? `${formatBytes(file.size)} · 点击可重新选择`
                      : `TXT（每行一条）/ CSV / XLSX / JSON`}
                  </span>
                </div>
                {file && (
                  <button
                    type="button"
                    className="kb-drop-clear"
                    aria-label="移除已选文件"
                    onClick={(event) => {
                      event.stopPropagation();
                      clearFile();
                    }}
                  >
                    <IconX size={14} />
                  </button>
                )}
              </div>
              {fileError && (
                <p className="kb-error">
                  <IconAlert size={13} /> {fileError}
                </p>
              )}
            </div>
          </div>

          <div className="kb-step">
            <span className="kb-step-index">2</span>
            <div className="kb-step-body">
              <div className="kb-step-title">名称与生成方式</div>
              <label className="kb-field">
                <span>知识库名称</span>
                <input
                  className="input"
                  placeholder="例如：2026Q1 巡检偏差语料"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>

              <div className="kb-mode-grid" role="radiogroup" aria-label="生成方式">
                {MODE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={mode === option.value}
                    className={`kb-mode ${mode === option.value ? 'is-active' : ''}`}
                    disabled={submitting}
                    onClick={() => setMode(option.value)}
                  >
                    <strong>{option.label}</strong>
                    <small>{option.hint}</small>
                  </button>
                ))}
              </div>

              <details className="kb-advanced">
                <summary>高级选项：自定义标识</summary>
                <label className="kb-field">
                  <span>知识库标识（可选）</span>
                  <input
                    className="input mono"
                    placeholder="字母/数字/.-_，留空自动生成"
                    value={ident}
                    onChange={(event) => setIdent(event.target.value)}
                  />
                </label>
                <p className="small muted">
                  标识会变成目录名与接口参数；同名冲突时后端会自动追加序号。
                </p>
              </details>
            </div>
          </div>

          <div className="kb-step">
            <span className="kb-step-index">3</span>
            <div className="kb-step-body">
              <div className="kb-step-title">提交生成</div>
              <div className="kb-submit">
                <button
                  className="btn btn-primary"
                  disabled={submitting || !file}
                  onClick={() => void submit()}
                >
                  <IconUpload size={15} />
                  {submitting ? '提交中…' : '上传并生成'}
                </button>
                <p className="kb-submit-hint">
                  生成是异步任务：
                  <strong>
                    {mode === 'purified'
                      ? '两万条语料逐条净化约需 1–3 小时'
                      : '原文入库通常几分钟内完成'}
                  </strong>
                  。提交后可离开本页，刷新回来会自动接上进度；生成期间请不要同时执行聚类。
                </p>
              </div>
              {formError && (
                <p className="kb-error">
                  <IconAlert size={13} /> {formError}
                </p>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ---------- 当前/最近的生成任务 ---------- */}
      {job && (
        <section
          className={`card kb-job is-${statusTone(job.status)}`}
          aria-label="生成任务进度"
        >
          <div className="kb-job-head">
            <span className="kb-job-state">
              {job.status === 'succeeded' ? (
                <IconCheckCircle size={16} />
              ) : job.status === 'failed' ? (
                <IconAlert size={16} />
              ) : (
                <span className="spinner" />
              )}
            </span>
            <div className="kb-job-title">
              <strong>{job.displayName || job.knowledgeBaseId || '生成任务'}</strong>
              <span className="mono small muted">
                {job.jobId}
                {job.knowledgeBaseId ? ` · ${job.knowledgeBaseId}` : ''}
              </span>
            </div>
            <span className="chip">{STAGE_LABEL[job.stage] ?? job.stage}</span>
            <button
              className="kb-job-close"
              aria-label="收起进度"
              onClick={() => {
                dismissedJob.current = job.jobId;
                setJob(null);
              }}
            >
              <IconX size={14} />
            </button>
          </div>

          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${job.status === 'succeeded' ? 100 : progress}%` }}
            />
          </div>

          <p className="small muted kb-job-meta">
            {job.message ?? '已提交，等待服务端开始处理。'}
            {(job.stage === 'purify' || job.stage === 'embed' || job.stage === 'index') &&
              ` · 当前阶段 ${job.processed}/${job.total}`}
            {job.status === 'failed' && job.error ? ` · 失败原因：${job.error.message}` : ''}
          </p>
          {job.warnings.length > 0 && (
            <div className="kb-job-warnings">
              {job.warnings.map((warning) => (
                <p key={warning} className="small muted">· {warning}</p>
              ))}
            </div>
          )}
        </section>
      )}

      {jobs.length > 0 && (
        <section className="card kb-jobs" aria-label="最近生成任务">
          <header className="kb-card-head">
            <div>
              <h2>最近生成任务</h2>
              <p className="small muted">服务端保留的任务记录；「查看进度」可接回未完成的任务。</p>
            </div>
            <button className="btn btn-outline btn-sm" onClick={() => void refreshJobs()}>
              <IconRefresh size={13} />刷新
            </button>
          </header>
          <ul className="kb-job-list">
            {jobs.map((item) => (
              <li key={item.jobId} className={job?.jobId === item.jobId ? 'is-active' : ''}>
                <span className={`kb-dot is-${statusTone(item.status)}`} aria-hidden="true" />
                <span className="kb-job-list-name">{item.displayName || item.knowledgeBaseId || item.jobId}</span>
                <span className="chip">{STAGE_LABEL[item.stage] ?? item.stage}</span>
                <span className="small muted kb-job-list-time">{formatTime(item.createdAt)}</span>
                <button
                  className="btn btn-outline btn-sm"
                  disabled={job?.jobId === item.jobId}
                  onClick={() => {
                    dismissedJob.current = null;
                    setJob(item);
                  }}
                >
                  {job?.jobId === item.jobId ? '已显示' : '查看进度'}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ---------- 已有库 ---------- */}
      <section className="card kb-list">
        <header className="kb-card-head">
          <div>
            <h2>已有知识库（{items.length}）</h2>
            <p className="small muted">
              带「profile 默认」标记的库，是检索增强 profile 未显式选择时使用的库。
            </p>
          </div>
          <div className="kb-list-actions">
            <label className="kb-search">
              <IconSearch size={14} />
              <input
                type="search"
                placeholder="按名称 / 标识 / 来源搜索"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              {query && (
                <button type="button" aria-label="清除搜索" onClick={() => setQuery('')}>
                  <IconX size={12} />
                </button>
              )}
            </label>
            <button className="btn btn-outline btn-sm" onClick={() => void refresh()} disabled={loading}>
              <IconRefresh size={13} />
              {loading ? '加载中…' : '刷新'}
            </button>
          </div>
        </header>

        {error && (
          <p className="kb-error kb-error-pad">
            <IconAlert size={13} /> {error}
          </p>
        )}

        <div className="kb-table-scroll">
          <table className="table kb-table">
            <thead>
              <tr>
                <th>名称 / 标识</th>
                <th>状态</th>
                <th>条目</th>
                <th>生成方式</th>
                <th>维度</th>
                <th>创建时间</th>
                <th>大小</th>
                <th aria-label="操作" />
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((item) => {
                const profiles = defaultKnowledgeBases.get(item.knowledgeBaseId) ?? [];
                return (
                  <tr
                    key={item.knowledgeBaseId}
                    className="kb-row"
                    onClick={() => void openDetail(item.knowledgeBaseId)}
                  >
                    <td>
                      <div className="kb-name">
                        <strong>{item.displayName}</strong>
                        {profiles.length > 0 && (
                          <span className="kb-badge is-default" title={`被 ${profiles.join('、')} 引用`}>
                            profile 默认
                          </span>
                        )}
                        <span className={`kb-badge ${item.hasEntries ? '' : 'is-warn'}`}>
                          {item.hasEntries ? '可查看条目' : '仅元信息'}
                        </span>
                      </div>
                      <span className="mono small muted">{item.knowledgeBaseId}</span>
                      {item.sourceName && <div className="small muted">来源：{item.sourceName}</div>}
                    </td>
                    <td>
                      <span className={`kb-status is-${statusTone(item.status)}`}>{item.status}</span>
                    </td>
                    <td className="kb-num">{item.entryCount.toLocaleString('zh-CN')}</td>
                    <td>{MODE_LABEL[item.processing] ?? item.processing}</td>
                    <td className="kb-num">{item.dimension || '—'}</td>
                    <td className="small">{formatTime(item.createdAt)}</td>
                    <td className="small">{formatBytes(item.sizeBytes)}</td>
                    <td>
                      <button
                        className="btn btn-primary-soft btn-sm"
                        onClick={(event) => {
                          event.stopPropagation();
                          void openDetail(item.knowledgeBaseId);
                        }}
                      >
                        查看
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!loading && filteredItems.length === 0 && (
                <tr>
                  <td colSpan={8} className="kb-empty">
                    <IconDatabase size={20} />
                    {items.length === 0
                      ? '还没有可用的知识库：用上面的表单上传一份语料生成，或先用运维命令构建。'
                      : `没有匹配「${query}」的知识库。`}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {detail && (
        <KnowledgeBaseDrawer
          detail={detail}
          loading={detailLoading}
          onClose={() => setDetail(null)}
          onLoadMore={() => void loadMoreEntries()}
        />
      )}
    </div>
  );
}

function KnowledgeBaseDrawer({
  detail,
  loading,
  onClose,
  onLoadMore,
}: {
  detail: ClusteringKnowledgeBaseDetail;
  loading: boolean;
  onClose: () => void;
  onLoadMore: () => void;
}) {
  return (
    <>
      <div className="drawer-mask" onClick={onClose} />
      <div className="drawer kb-drawer" role="dialog" aria-modal="true" aria-label="知识库详情">
        <div className="drawer-head">
          <div className="kb-drawer-title">
            <h3>{detail.displayName}</h3>
            <span className="mono small muted">{detail.knowledgeBaseId}</span>
          </div>
          <button className="btn btn-outline btn-sm" onClick={onClose}>
            <IconX size={14} />
            关闭
          </button>
        </div>
        <div className="drawer-body">
          <div className="kb-detail-grid">
            <div><span>状态</span><strong>{detail.status} / {detail.verification}</strong></div>
            <div><span>条目数</span><strong>{detail.entryCount.toLocaleString('zh-CN')}</strong></div>
            <div><span>生成方式</span><strong>{MODE_LABEL[detail.processing] ?? detail.processing}</strong></div>
            <div><span>向量模型</span><strong>{detail.modelId || '—'}</strong></div>
            <div><span>维度 / 度量</span><strong>{detail.dimension || '—'} · {detail.metric || '—'}</strong></div>
            <div><span>大小</span><strong>{formatBytes(detail.sizeBytes)}</strong></div>
            <div><span>创建时间</span><strong>{formatTime(detail.createdAt)}</strong></div>
            <div><span>来源</span><strong>{detail.sourceName || '—'}</strong></div>
          </div>

          <div className="kb-detail-block">
            <div className="kb-detail-label">指纹</div>
            <div className="mono small">
              模型 {detail.modelFingerprint ? `${detail.modelFingerprint.slice(0, 24)}…` : '—'}
            </div>
            <div className="mono small">
              来源 {detail.sourceSha256 ? `${detail.sourceSha256.slice(0, 24)}…` : '—'}
            </div>
          </div>

          {detail.warnings.map((warning) => (
            <p key={warning} className="kb-warn">
              <IconAlert size={13} /> {warning}
            </p>
          ))}

          <div className="kb-detail-head">
            <div className="kb-detail-label">
              条目内容（{detail.entryTotal.toLocaleString('zh-CN')}）
            </div>
            <span className="small muted">
              来源：{SOURCE_LABEL[detail.entriesSource] ?? detail.entriesSource}
            </span>
          </div>
          {loading && <div className="muted small">加载中…</div>}
          <div className="kb-entries">
            {detail.entries.map((entry) => (
              <div key={entry.index} className="kb-entry">
                <span className="clustering-id">#{entry.index}</span>
                <span className="kb-entry-text">{entry.text}</span>
              </div>
            ))}
            {!loading && detail.entries.length === 0 && (
              <div className="muted small">该知识库没有可展示的条目文本（见上方说明）。</div>
            )}
          </div>
          {detail.entries.length < detail.entryTotal && (
            <button className="btn btn-outline btn-sm kb-load-more" onClick={onLoadMore}>
              加载更多（已显示 {detail.entries.length}/{detail.entryTotal}）
            </button>
          )}
        </div>
      </div>
    </>
  );
}
