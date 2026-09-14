import assert from 'node:assert/strict';
import test from 'node:test';
import type {
  ClusteringClusterGroup,
  ClusteringDatasetItem,
  ClusteringJobInfo,
  ClusteringJobResultData,
  ClusteringKnowledgeBaseInfo,
  ClusteringOfflineBaseline,
  ClusteringProfileInfo,
  ClusteringResultItem,
} from '../../shared/clustering.js';
import {
  buildKnowledgeBase,
  cancelClusteringJob,
  fetchClusteringBaseline,
  fetchClusteringHealth,
  fetchClusteringJob,
  fetchClusteringJobItems,
  fetchClusteringJobResult,
  fetchClusteringJobs,
  fetchClusteringProfiles,
  fetchKnowledgeBaseDetail,
  fetchKnowledgeBases,
  runClustering,
  submitClusteringJob,
  uploadClusteringDataset,
} from '../../web/api/clustering.js';
import { describeJobStatus, isCancellable } from '../../web/components/ClusteringJobPanel.js';
import { knowledgeBaseName, supportsRetrieval } from '../../web/components/KnowledgeBasePicker.js';
import { MAX_VISIBLE, orderClusters } from '../../web/components/ClusterFilterBar.js';
import { NOISE_COLOR, clusterColor, clusterTint } from '../../web/lib/clusterColor.js';
import {
  clampPage,
  clusterOverview,
  countActiveJobs,
  describeJobOutcome,
  formatJobDuration,
  formatJobTime,
  hasActiveJobs,
  historyExpandKind,
  historyStatusClass,
  historyStatusLabel,
  jobItemsPageCount,
  jobRunLabel,
  sortJobsNewestFirst,
  summarizeJobResult,
  toggleExpandedJob,
} from '../../web/lib/clusteringJobHistory.js';
import {
  archivedBenchmarkColumns,
  archivedColumnLabel,
  comparisonRows,
  describeGroundTruth,
  detectLabelField,
  formatDelta,
  formatMetricValue,
  hasComputedMetrics,
  hasControlMetrics,
  metricRows,
  shouldUseArchivedBenchmark,
} from '../../web/lib/clusteringMetrics.js';
import {
  DEBUG_ONLY_CLUSTERING_WARNINGS,
  logDebugWarnings,
  splitClusteringWarnings,
} from '../../web/lib/clusteringWarnings.js';
import { applyTableQuery, collectKeywordOptions } from '../../web/lib/useTableQuery.js';

/**
 * 替换全局 fetch，记录调用参数并返回预置响应。
 *
 * 支持传工厂函数：`Response` 的 body 只能读一次，同一个实例无法服务两次请求，
 * 因此需要多次调用的用例必须每次生成新的 Response。
 */
function stubFetch(responder: Response | Error | (() => Response | Error)) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    const response = typeof responder === 'function' ? responder() : responder;
    if (response instanceof Error) throw response;
    return response;
  }) as typeof fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

test('successful envelope is unwrapped to data', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: { status: 'ok', engine: { loaded: true } }, error: null }),
  );
  try {
    const health = await fetchClusteringHealth(true);
    assert.equal(health.status, 'ok');
    assert.equal(health.engine.loaded, true);
  } finally {
    stub.restore();
  }
});

test('requests are same-origin and never hardcode the python host', async () => {
  const stub = stubFetch(() => jsonResponse({ success: true, data: { status: 'ok', engine: {} }, error: null }));
  try {
    await fetchClusteringHealth(true);
    await fetchClusteringProfiles();

    assert.equal(stub.calls[0].url, '/api/clustering/health?refresh=true');
    assert.equal(stub.calls[1].url, '/api/clustering/profiles');
    for (const call of stub.calls) {
      assert.ok(!/^https?:\/\//.test(call.url), `不应硬编码绝对地址：${call.url}`);
    }
  } finally {
    stub.restore();
  }
});

test('error envelope surfaces backend code and message', async () => {
  const stub = stubFetch(
    jsonResponse(
      {
        success: false,
        data: null,
        error: { code: 'PROFILE_UNAVAILABLE', message: '算法「hdbscan」当前没有可用的 profile。', requestId: 'r-1' },
      },
      409,
    ),
  );
  try {
    await assert.rejects(fetchClusteringProfiles(), (error: Error & { code?: string; status?: number }) => {
      assert.equal(error.code, 'PROFILE_UNAVAILABLE');
      assert.equal(error.status, 409);
      assert.match(error.message, /没有可用的 profile/);
      return true;
    });
  } finally {
    stub.restore();
  }
});

test('non-json gateway error is turned into a readable message', async () => {
  const stub = stubFetch(new Response('<html>Bad Gateway</html>', { status: 502 }));
  try {
    await assert.rejects(fetchClusteringProfiles(), (error: Error & { code?: string; status?: number }) => {
      assert.equal(error.code, 'HTTP_ERROR');
      assert.equal(error.status, 502);
      assert.match(error.message, /暂时不可用/);
      return true;
    });
  } finally {
    stub.restore();
  }
});

test('network failure is reported instead of failing silently', async () => {
  const stub = stubFetch(new Error('connect ECONNREFUSED'));
  try {
    await assert.rejects(fetchClusteringHealth(), (error: Error & { code?: string }) => {
      assert.equal(error.code, 'NETWORK_ERROR');
      assert.match(error.message, /npm run dev:py/);
      return true;
    });
  } finally {
    stub.restore();
  }
});

test('run posts camelCase options in the body', async () => {
  const stub = stubFetch(jsonResponse({ success: true, data: { summary: {}, clusters: [], items: [], visualization: [] }, error: null }));
  try {
    await runClustering(
      [
        { id: 'a', text: '未设置警戒围栏', metadata: {} },
        { id: 'b', text: '缺少连墙件', metadata: {} },
      ],
      { algorithm: 'dbscan', profileId: 'p-1', visualize: false, reduceMethod: 'none' },
    );

    const call = stub.calls[0];
    assert.equal(call.url, '/api/clustering/run');
    assert.equal(call.init?.method, 'POST');
    assert.equal((call.init?.headers as Record<string, string>)['Content-Type'], 'application/json');

    const body = JSON.parse(String(call.init?.body));
    assert.deepEqual(body.options, {
      algorithm: 'dbscan',
      profileId: 'p-1',
      visualize: false,
      reduceMethod: 'none',
    });
    assert.equal(body.items[0].text, '未设置警戒围栏');
  } finally {
    stub.restore();
  }
});

test('run forwards the selected knowledge base as a camelCase option', async () => {
  const stub = stubFetch(jsonResponse({ success: true, data: { summary: {}, clusters: [], items: [], visualization: [] }, error: null }));
  try {
    await runClustering([{ id: 'a', text: '支架安装偏差', metadata: {} }], {
      profileId: 'spear_purified_retrieval',
      knowledgeBaseId: 'kb-2026q1',
    });
    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.equal(body.options.knowledgeBaseId, 'kb-2026q1');
  } finally {
    stub.restore();
  }
});

test('knowledge base list unwraps the envelope', async () => {
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: { knowledgeBases: [{ knowledgeBaseId: 'kb-1', displayName: '库一', entryCount: 3 }] },
      error: null,
    }),
  );
  try {
    const items = await fetchKnowledgeBases();
    assert.equal(stub.calls[0].url, '/api/clustering/knowledge-bases');
    assert.equal(items[0].knowledgeBaseId, 'kb-1');
  } finally {
    stub.restore();
  }
});

test('knowledge base detail url-encodes the id and pages entries', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: { knowledgeBaseId: 'kb/1', entries: [], warnings: [] }, error: null }),
  );
  try {
    await fetchKnowledgeBaseDetail('kb/1', 20, 50);
    assert.equal(stub.calls[0].url, '/api/clustering/knowledge-bases/kb%2F1?offset=20&limit=50');
  } finally {
    stub.restore();
  }
});

test('knowledge base build posts multipart fields the backend expects', async () => {
  const stub = stubFetch(jsonResponse({ success: true, data: { jobId: 'kbjob_1', status: 'running', terminal: false }, error: null }));
  try {
    await buildKnowledgeBase(new File(['支架安装偏差'], 'corpus.txt', { type: 'text/plain' }), {
      name: '巡检语料',
      mode: 'purified',
      ident: 'kb-custom',
    });
    const call = stub.calls[0];
    assert.equal(call.url, '/api/clustering/knowledge-bases');
    assert.equal(call.init?.method, 'POST');
    const body = call.init?.body as FormData;
    assert.ok(body instanceof FormData);
    assert.ok(body.get('file'));
    assert.equal(body.get('name'), '巡检语料');
    assert.equal(body.get('mode'), 'purified');
    assert.equal(body.get('ident'), 'kb-custom');
  } finally {
    stub.restore();
  }
});

test('upload sends the file as multipart form data', async () => {
  const stub = stubFetch(jsonResponse({ success: true, data: { sourceName: 'a.csv', total: 2, warnings: [], items: [] }, error: null }));
  try {
    await uploadClusteringDataset(new File(['隐患描述'], 'a.csv', { type: 'text/csv' }));

    const call = stub.calls[0];
    assert.equal(call.url, '/api/clustering/datasets');
    assert.equal(call.init?.method, 'POST');
    assert.ok(call.init?.body instanceof FormData);
    assert.ok((call.init?.body as FormData).get('file'));
  } finally {
    stub.restore();
  }
});

// ---------------------------------------------------------------- 异步作业

test('job submission keeps the idempotency key in the header and out of the body options', async () => {
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: { jobId: 'job_1', status: 'queued', itemCount: 2, detail: {}, warnings: [] },
      error: null,
    }),
  );
  try {
    await submitClusteringJob([{ id: 'a', text: '未设置警戒围栏', metadata: {} }], {
      profileId: 'p-1',
      datasetId: 'ds_1',
      idempotencyKey: 'k-1',
    });

    const call = stub.calls[0];
    assert.equal(call.url, '/api/clustering/jobs');
    assert.equal(call.init?.method, 'POST');
    const headers = call.init?.headers as Record<string, string>;
    // 代理层可能丢掉自定义头，因此 body 里也必须能承载幂等键；
    // 但提交路径把键放在头部，body 的 options 里不应再出现同名字段。
    assert.equal(headers['Idempotency-Key'], 'k-1');

    const body = JSON.parse(String(call.init?.body));
    assert.equal(body.options.datasetId, 'ds_1');
    assert.equal(body.options.profileId, 'p-1');
    assert.equal(body.options.idempotencyKey, undefined);
  } finally {
    stub.restore();
  }
});

test('job submission forwards the selected reference database like the sync run does', async () => {
  // 异步作业与同步 /run 必须同口径：选中的偏差数据库不能只在同步路径生效。
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: { jobId: 'job_2', status: 'queued', itemCount: 2, detail: {}, warnings: [] },
      error: null,
    }),
  );
  try {
    await submitClusteringJob([{ id: 'a', text: '支架安装偏差', metadata: {} }], {
      profileId: 'spear_purified_retrieval',
      datasetId: 'ds_1',
      knowledgeBaseId: 'kb-2026q1',
    });

    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.equal(body.options.knowledgeBaseId, 'kb-2026q1');
    assert.equal(body.options.profileId, 'spear_purified_retrieval');
  } finally {
    stub.restore();
  }
});

test('reference database picker only claims to be active on retrieval profiles', () => {
  const base: ClusteringProfileInfo = {
    profileId: 'p-ret',
    algorithm: 'dbscan',
    modelId: 'fixture',
    features: { n_results: 8 },
    algorithmParams: {},
    implementationVersion: 'semantic-v1',
    maxSamples: 1000,
    available: true,
    warnings: [],
  };
  assert.equal(supportsRetrieval(base), true);
  assert.equal(supportsRetrieval({ ...base, profileId: 'p-plain', features: { n_results: 0 } }), false);
  assert.equal(supportsRetrieval({ ...base, features: {} }), false);
  assert.equal(supportsRetrieval(null), false);
});

test('knowledge base name falls back to the identifier instead of hiding it', () => {
  const entry: ClusteringKnowledgeBaseInfo = {
    knowledgeBaseId: 'kb-1',
    displayName: '库一',
    status: 'completed',
    verification: 'verified',
    processing: 'raw-upload-v1',
    entryCount: 3,
    dimension: 1024,
    metric: 'l2',
    modelFingerprint: 'fp',
    sourceSha256: 'sha',
    sizeBytes: 1024,
    hasEntries: true,
    hasCorpus: true,
  };
  const list = [entry];
  assert.equal(knowledgeBaseName(list, 'kb-1'), '库一');
  // 找不到时返回标识本身：调用方据此知道"默认库不在清单里"，而不是显示空白
  assert.equal(knowledgeBaseName(list, 'kb-missing'), 'kb-missing');
  assert.equal(knowledgeBaseName(list, null), '');
});

test('job item paging sends filters as query params and never fetches everything', async () => {  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: {
        jobId: 'job_1',
        runId: 'run_1',
        total: 400,
        offset: 100,
        limit: 50,
        returned: 0,
        hasMore: true,
        scanned: 400,
        truncated: false,
        items: [],
      },
      error: null,
    }),
  );
  try {
    const page = await fetchClusteringJobItems('job_1', {
      offset: 100,
      limit: 50,
      clusterId: -1,
      keyword: '连墙件',
    });

    assert.equal(
      stub.calls[0].url,
      '/api/clustering/jobs/job_1/items?offset=100&limit=50&clusterId=-1&keyword=%E8%BF%9E%E5%A2%99%E4%BB%B6',
    );
    // total 是后端筛选后的总数，前端据此渲染页数——不是"当前页条数"
    assert.equal(page.total, 400);
    assert.equal(page.hasMore, true);
  } finally {
    stub.restore();
  }
});

test('job item ids are url-encoded so a bad id cannot escape the route', async () => {
  const stub = stubFetch(jsonResponse({ success: true, data: { jobId: 'x', items: [] }, error: null }));
  try {
    await fetchClusteringJob('job_1/../secrets');
    assert.equal(stub.calls[0].url, '/api/clustering/jobs/job_1%2F..%2Fsecrets');
  } finally {
    stub.restore();
  }
});

test('cancel posts to the job cancel endpoint', async () => {
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: { jobId: 'job_1', status: 'cancelled', itemCount: 0, detail: {}, warnings: [] },
      error: null,
    }),
  );
  try {
    const snapshot = await cancelClusteringJob('job_1');
    assert.equal(stub.calls[0].url, '/api/clustering/jobs/job_1/cancel');
    assert.equal(stub.calls[0].init?.method, 'POST');
    assert.equal(snapshot.status, 'cancelled');
  } finally {
    stub.restore();
  }
});

test('job status wording distinguishes cancellation from failure', () => {
  const base = { jobId: 'job_1', itemCount: 7, cancelRequested: false, hardCancelled: false, detail: {}, warnings: [] };
  assert.equal(describeJobStatus(null), '尚未提交作业');
  assert.match(describeJobStatus({ ...base, status: 'running', terminal: false }), /执行中（共 7 条样本）/);
  // 取消是调用方的意图，不是失败：措辞必须区分开，否则用户以为是自己搞坏了
  assert.equal(describeJobStatus({ ...base, status: 'cancelled', hardCancelled: true, terminal: true }), '已取消（执行进程已终止）');
  assert.match(
    describeJobStatus({ ...base, status: 'failed', terminal: true, error: { code: 'X', message: '模型不可用' } }),
    /失败：模型不可用/,
  );
});

test('只有未进入终态的作业才允许取消', () => {
  const base = { jobId: 'job_1', itemCount: 1, cancelRequested: false, hardCancelled: false, detail: {}, warnings: [] };
  assert.equal(isCancellable(null), false);
  assert.equal(isCancellable({ ...base, status: 'queued', terminal: false }), true);
  assert.equal(isCancellable({ ...base, status: 'running', terminal: false }), true);
  assert.equal(isCancellable({ ...base, status: 'succeeded', terminal: true }), false);
  assert.equal(isCancellable({ ...base, status: 'cancelled', terminal: true }), false);
});

test('cluster colours are stable per cluster id and distinct across clusters', () => {
  assert.equal(clusterColor(0), clusterColor(0));
  assert.notEqual(clusterColor(0), clusterColor(1));
  assert.notEqual(clusterColor(0), clusterColor(2));
});

test('noise keeps its own neutral colour', () => {
  assert.equal(clusterColor(-1), NOISE_COLOR);
  assert.notEqual(clusterColor(-1), clusterColor(0));
});

test('cluster tint derives an rgba value from the cluster colour', () => {
  assert.equal(clusterTint(0, 0.5), 'rgba(0, 113, 227, 0.5)');
  assert.match(clusterTint(-1), /^rgba\(161, 161, 166/);
});

/* ---------- 明细表的筛选 / 排序 / 搜索 ---------- */

/** 构造一行归属结果，只填测试关心的字段。 */
function row(
  id: string,
  clusterId: number,
  confidence: number | null,
  distance: number | null,
  keywords: string[],
): ClusteringResultItem {
  return {
    id,
    text: `文本 ${id}`,
    clusterId,
    clusterLabel: `类别 ${clusterId}`,
    confidence,
    distance,
    keywords,
    metadata: {},
  };
}

const ROWS: ClusteringResultItem[] = [
  row('a', 0, 0.91, 0.42, ['护栏', '作业平台']),
  row('b', 1, 0.55, 0.88, ['脚手架', '连墙件']),
  row('c', 0, 0.73, 0.15, ['护栏', '警示标识']),
  row('d', 2, null, null, ['配电箱']),
];

const NO_QUERY = {
  keywordSearch: '',
  clusterColumnFilter: [] as number[],
  keywordColumnFilter: [] as string[],
  sort: null,
};

test('no query returns rows unchanged and preserves order', () => {
  const out = applyTableQuery(ROWS, NO_QUERY);
  assert.deepEqual(out.map((item) => item.id), ['a', 'b', 'c', 'd']);
});

test('row cluster filter keeps only the chosen clusters (union)', () => {
  const out = applyTableQuery(ROWS, NO_QUERY, [0, 2]);
  assert.deepEqual(out.map((item) => item.id), ['a', 'c', 'd']);
});

test('keyword search matches substrings case-insensitively', () => {
  const out = applyTableQuery(ROWS, { ...NO_QUERY, keywordSearch: '护栏' });
  assert.deepEqual(out.map((item) => item.id), ['a', 'c']);

  const partial = applyTableQuery(ROWS, { ...NO_QUERY, keywordSearch: '脚手' });
  assert.deepEqual(partial.map((item) => item.id), ['b']);
});

test('keyword search trims input and ignores blank queries', () => {
  const padded = applyTableQuery(ROWS, { ...NO_QUERY, keywordSearch: '  护栏  ' });
  assert.deepEqual(padded.map((item) => item.id), ['a', 'c']);

  const blank = applyTableQuery(ROWS, { ...NO_QUERY, keywordSearch: '   ' });
  assert.equal(blank.length, ROWS.length);
});

test('column filters combine as AND across columns', () => {
  const out = applyTableQuery(ROWS, {
    ...NO_QUERY,
    clusterColumnFilter: [0],
    keywordColumnFilter: ['警示标识'],
  });
  assert.deepEqual(out.map((item) => item.id), ['c']);
});

test('confidence sorts high-to-low on desc, with nulls sunk to the end', () => {
  const out = applyTableQuery(ROWS, { ...NO_QUERY, sort: { key: 'confidence', direction: 'desc' } });
  assert.deepEqual(out.map((item) => item.id), ['a', 'c', 'b', 'd']);
});

test('distance sorts low-to-high on asc, with nulls sunk to the end', () => {
  const out = applyTableQuery(ROWS, { ...NO_QUERY, sort: { key: 'distance', direction: 'asc' } });
  assert.deepEqual(out.map((item) => item.id), ['c', 'a', 'b', 'd']);
});

test('sorting never mutates the original array', () => {
  const original = ROWS.map((item) => item.id);
  applyTableQuery(ROWS, { ...NO_QUERY, sort: { key: 'confidence', direction: 'asc' } });
  assert.deepEqual(ROWS.map((item) => item.id), original);
});

test('keyword options are ranked by frequency then by name', () => {
  const options = collectKeywordOptions(ROWS);
  assert.equal(options[0].value, '护栏');
  assert.equal(options[0].count, 2);
  const singles = options.filter((option) => option.count === 1).map((option) => option.value);
  assert.deepEqual(singles, [...singles].sort((a, b) => a.localeCompare(b, 'zh-Hans-CN')));
});

/* ---------- 簇筛选条的排序与截断 ---------- */

/** 构造一个簇分组，只填排序/渲染关心的字段。 */
function group(clusterId: number, size: number): ClusteringClusterGroup {
  return {
    clusterId,
    label: `类别 ${clusterId}`,
    size,
    keywords: [],
    representativeSamples: [],
    metadataDistribution: {},
  };
}

test('clusters are ordered by size descending, so the biggest comes first', () => {
  const ordered = orderClusters([group(0, 3), group(1, 40), group(2, 12)]);
  assert.deepEqual(ordered.map((item) => item.clusterId), [1, 2, 0]);
});

test('noise stays first regardless of its size', () => {
  const ordered = orderClusters([group(5, 99), group(-1, 1), group(0, 50)]);
  assert.equal(ordered[0].clusterId, -1);
  assert.deepEqual(ordered.slice(1).map((item) => item.clusterId), [5, 0]);
});

test('equal sizes fall back to cluster id ascending for a stable order', () => {
  const ordered = orderClusters([group(7, 5), group(2, 5), group(4, 5)]);
  assert.deepEqual(ordered.map((item) => item.clusterId), [2, 4, 7]);
});

test('ordering does not mutate the input array', () => {
  const input = [group(0, 3), group(1, 40)];
  orderClusters(input);
  assert.deepEqual(input.map((item) => item.clusterId), [0, 1]);
});

test('filter bar renders at most MAX_VISIBLE clusters', () => {
  const many = Array.from({ length: MAX_VISIBLE + 17 }, (_, index) => group(index, index + 1));
  const ordered = orderClusters(many);
  // 降序后，前 MAX_VISIBLE 个应当是 size 最大的那批
  const shown = ordered.slice(0, MAX_VISIBLE);
  assert.equal(shown.length, MAX_VISIBLE);
  assert.equal(shown[0].clusterId, MAX_VISIBLE + 16);
  assert.equal(shown[MAX_VISIBLE - 1].clusterId, 17);
  // 被截断的都是小簇
  const hidden = ordered.slice(MAX_VISIBLE);
  assert.equal(hidden.length, 17);
  assert.ok(hidden.every((item) => item.size < shown[shown.length - 1].size));
});

test('a cluster selected beyond the cap is kept visible so it can be unselected', () => {
  const many = Array.from({ length: MAX_VISIBLE + 5 }, (_, index) => group(index, index + 1));
  const ordered = orderClusters(many);
  const smallest = ordered[ordered.length - 1];
  // 模拟组件里的可见性计算
  const chosen = new Set([smallest.clusterId]);
  const visible = ordered.filter((cluster, index) => index < MAX_VISIBLE || chosen.has(cluster.clusterId));
  assert.ok(visible.some((cluster) => cluster.clusterId === smallest.clusterId));
  assert.equal(visible.length, MAX_VISIBLE + 1);
});

// ---------------------------------------------------------------- SPEAR / 离线基准

test('purify override is sent as a boolean when the caller asks for a control run', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: { summary: {}, clusters: [], items: [], visualization: [] }, error: null }),
  );
  try {
    await runClustering([{ id: 'a', text: '未设置警戒围栏', metadata: {} }], {
      profileId: 'spear_purified',
      purify: false,
    });
    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.equal(body.options.purify, false);
  } finally {
    stub.restore();
  }
});

test('omitting the purify override leaves it out of the body entirely', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: { summary: {}, clusters: [], items: [], visualization: [] }, error: null }),
  );
  try {
    await runClustering([{ id: 'a', text: '未设置警戒围栏', metadata: {} }], { profileId: 'spear_purified' });
    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.ok(!('purify' in body.options));
  } finally {
    stub.restore();
  }
});

test('offline baseline uses the relative endpoint and unwraps the envelope', async () => {
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: {
        source: 'unit-test',
        fullRun: true,
        rows: { nr0: { ari: 0.1528 }, spear_purified_retrieval: { ari: 0.2705 } },
        comparison: { ari: 0.1177, ariRelative: 0.7703 },
      },
      error: null,
    }),
  );
  try {
    const baseline = await fetchClusteringBaseline();
    assert.equal(stub.calls[0].url, '/api/clustering/baseline');
    assert.equal(baseline.rows.nr0.ari, 0.1528);
    assert.equal(baseline.comparison?.ariRelative, 0.7703);
    // 离线演示要求零外链：任何绝对 URL 都会在离网环境下白屏
    assert.ok(!/^https?:/.test(stub.calls[0].url));
  } finally {
    stub.restore();
  }
});

/* ---------- 历史测试记录（作业历史） ---------- */

/** 构造一条作业记录，只填测试关心的字段。 */
function job(overrides: Partial<ClusteringJobInfo> = {}): ClusteringJobInfo {
  return {
    jobId: 'job_1',
    status: 'succeeded',
    itemCount: 7,
    cancelRequested: false,
    hardCancelled: false,
    detail: {},
    warnings: [],
    terminal: true,
    ...overrides,
  };
}

/** 构造一份作业结果摘要（不含明细），字段保持与后端口径一致。 */
function jobResult(overrides: Partial<ClusteringJobResultData> = {}): ClusteringJobResultData {
  return {
    jobId: 'job_1',
    runId: 'run_1',
    summary: {
      runId: 'run_1',
      totalSamples: 10,
      clusterCount: 3,
      noiseCount: 1,
      largestClusterSize: 5,
      smallestClusterSize: 2,
      avgClusterSize: 3,
      algorithm: 'dbscan',
      profileId: 'p-1',
      modelId: 'm-1',
      implementationVersion: 'legacy-v1',
      embeddingDimension: 1024,
      cacheHit: true,
      elapsedMs: 120,
      gatewayMs: 130,
      warnings: [],
    },
    clusters: [],
    aggregate: {},
    visualization: [],
    detail: { itemCount: 10 },
    warnings: [],
    ...overrides,
  };
}

test('历史列表按 limit 与可选 status 拼查询参数', async () => {
  // Response 的 body 只能读一次，两次调用必须各自生成新的响应
  const stub = stubFetch(() =>
    jsonResponse({ success: true, data: { jobs: [job()] }, error: null }),
  );
  try {
    const all = await fetchClusteringJobs(50);
    const succeeded = await fetchClusteringJobs(10, 'succeeded');

    assert.equal(stub.calls[0].url, '/api/clustering/jobs?limit=50');
    assert.equal(stub.calls[1].url, '/api/clustering/jobs?limit=10&status=succeeded');
    assert.equal(all[0].jobId, 'job_1');
    assert.equal(succeeded.length, 1);
    // 历史来源必须是同源相对路径：换了后端地址也不该在前端写死
    for (const call of stub.calls) {
      assert.ok(!/^https?:\/\//.test(call.url), `不应硬编码绝对地址：${call.url}`);
    }
  } finally {
    stub.restore();
  }
});

test('展开历史记录只读结果摘要接口，不去拉全量明细', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: jobResult(), error: null }),
  );
  try {
    const data = await fetchClusteringJobResult('job_1');
    assert.equal(stub.calls[0].url, '/api/clustering/jobs/job_1/result');
    assert.equal(data.summary.clusterCount, 3);
    assert.equal(data.detail.itemCount, 10);
  } finally {
    stub.restore();
  }
});

test('作业历史按时间新→旧排序且不改动入参数组', () => {
  const input = [
    job({ jobId: 'old', createdAt: '2026-01-01T00:00:00Z' }),
    job({ jobId: 'new', createdAt: '2026-03-01T00:00:00Z' }),
    job({ jobId: 'mid', createdAt: '2026-02-01T00:00:00Z' }),
  ];
  const ordered = sortJobsNewestFirst(input);
  assert.deepEqual(ordered.map((item) => item.jobId), ['new', 'mid', 'old']);
  // 排序是给展示用的，不应把调用方持有的列表顺序改掉
  assert.deepEqual(input.map((item) => item.jobId), ['old', 'new', 'mid']);
});

test('createdAt 相同或缺失时用 jobId 兜底，顺序稳定可复现', () => {
  const same = [
    job({ jobId: 'b', createdAt: '2026-01-01T00:00:00Z' }),
    job({ jobId: 'a', createdAt: '2026-01-01T00:00:00Z' }),
  ];
  assert.deepEqual(sortJobsNewestFirst(same).map((item) => item.jobId), ['a', 'b']);
  const missing = [job({ jobId: 'x', createdAt: null }), job({ jobId: 'y', createdAt: null })];
  assert.deepEqual(sortJobsNewestFirst(missing).map((item) => item.jobId), ['x', 'y']);
});

test('耗时优先用 startedAt→finishedAt，未完成显示"进行中"', () => {
  assert.equal(
    formatJobDuration(
      job({
        startedAt: '2026-01-01T00:00:00Z',
        finishedAt: '2026-01-01T00:02:05Z',
        createdAt: '2026-01-01T00:00:00Z',
      }),
    ),
    '2 分 5 秒',
  );
  assert.equal(
    formatJobDuration(
      job({
        status: 'running',
        terminal: false,
        startedAt: '2026-01-01T00:00:00Z',
        createdAt: '2026-01-01T00:00:00Z',
      }),
    ),
    '进行中',
  );
  assert.equal(formatJobDuration(job({ createdAt: null })), '—');
  // 走兜底口径：没有 startedAt 时用 createdAt→updatedAt
  assert.equal(
    formatJobDuration(
      job({
        startedAt: null,
        createdAt: '2026-01-01T00:00:00Z',
        finishedAt: null,
        updatedAt: '2026-01-01T01:30:00Z',
      }),
    ),
    '1 小时 30 分',
  );
});

test('记录时间缺失显示占位，无法解析时原样回显而不是 Invalid Date', () => {
  assert.equal(formatJobTime(null), '—');
  assert.equal(formatJobTime(''), '—');
  assert.equal(formatJobTime('not-a-date'), 'not-a-date');
  assert.match(formatJobTime('2026-03-01T00:00:00Z'), /2026/);
});

test('展开形态按状态分流：成功读结果、失败给原因、取消与执行中各自说明', () => {
  assert.equal(historyExpandKind(job({ status: 'succeeded' })), 'result');
  assert.equal(historyExpandKind(job({ status: 'failed', terminal: true })), 'failed');
  assert.equal(historyExpandKind(job({ status: 'cancelled', terminal: true, hardCancelled: true })), 'cancelled');
  assert.equal(historyExpandKind(job({ status: 'running', terminal: false })), 'running');
  assert.equal(historyExpandKind(job({ status: 'queued', terminal: false })), 'running');

  // 失败必须把后端给的原因和错误码写出来，而不是只显示"失败"两个字
  const failed = describeJobOutcome(
    job({ status: 'failed', error: { code: 'JOB_RESULT_UNAVAILABLE', message: '模型不可用' } }),
  );
  assert.match(failed, /模型不可用/);
  assert.match(failed, /JOB_RESULT_UNAVAILABLE/);
  // 取消是调用方意图，措辞不能写成失败
  assert.match(describeJobOutcome(job({ status: 'cancelled', hardCancelled: true })), /已取消/);
  assert.match(describeJobOutcome(job({ status: 'running', terminal: false })), /正在执行/);
  assert.match(describeJobOutcome(job({ status: 'queued', terminal: false, itemCount: 3 })), /排队/);
});

test('点击同一条收起、点击另一条切换，同一时刻只展开一条', () => {
  assert.equal(toggleExpandedJob(null, 'job_1'), 'job_1');
  assert.equal(toggleExpandedJob('job_1', 'job_1'), null);
  assert.equal(toggleExpandedJob('job_1', 'job_2'), 'job_2');
});

test('结果摘要包含统计口径，质量分缺失时整项省略', () => {
  const rows = summarizeJobResult(jobResult());
  const labels = rows.map(([label]) => label);
  assert.ok(labels.includes('样本总数'));
  assert.ok(labels.includes('网关耗时'));
  assert.ok(!labels.includes('簇质量分'));

  const withQuality = summarizeJobResult(
    jobResult({ summary: { ...jobResult().summary, qualityScore: 0.4213 } }),
  );
  assert.equal(withQuality.find(([label]) => label === '簇质量分')?.[1], '0.421');
});

test('簇概览按大小降序截断且不改动原数组', () => {
  const clusters = [group(0, 3), group(1, 40), group(2, 12), group(-1, 7)];
  const overview = clusterOverview(clusters, 2);
  assert.deepEqual(overview.map((item) => item.clusterId), [1, 2]);
  assert.deepEqual(clusters.map((item) => item.clusterId), [0, 1, 2, -1]);
});

test('明细分页页数至少为 1，页码收敛在合法区间', () => {
  assert.equal(jobItemsPageCount(0), 1);
  assert.equal(jobItemsPageCount(20), 1);
  assert.equal(jobItemsPageCount(21), 2);
  assert.equal(jobItemsPageCount(Number.NaN), 1);
  assert.equal(clampPage(5, 3), 2);
  assert.equal(clampPage(-1, 3), 0);
  assert.equal(clampPage(1, 1), 0);
});

test('未完成作业计数驱动"可刷新看进度"的提示', () => {
  const list = [
    job({ jobId: 'a', status: 'running', terminal: false }),
    job({ jobId: 'b', status: 'succeeded' }),
    job({ jobId: 'c', status: 'queued', terminal: false }),
  ];
  assert.equal(hasActiveJobs(list), true);
  assert.equal(countActiveJobs(list), 2);
  assert.equal(hasActiveJobs([job({ jobId: 'b' })]), false);
  assert.equal(countActiveJobs([]), 0);
});

test('历史状态的徽标文案与配色分级正确', () => {
  assert.equal(historyStatusLabel(job({ status: 'succeeded' })), '已完成');
  assert.equal(historyStatusLabel(job({ status: 'failed' })), '失败');
  assert.equal(historyStatusClass(job({ status: 'succeeded' })), 'is-ok');
  assert.equal(historyStatusClass(job({ status: 'failed' })), 'is-bad');
  assert.equal(historyStatusClass(job({ status: 'cancelled' })), 'is-warn');
  assert.equal(historyStatusClass(job({ status: 'running', terminal: false })), '');
});

test('历史记录优先用 Run ID 定位，缺失时退回 jobId', () => {
  assert.equal(jobRunLabel(job({ runId: 'run_9' })), 'run_9');
  assert.equal(jobRunLabel(job({ runId: null })), 'job_1');
});

/* ---------- 告警码分流（原始码只进调试窗口） ---------- */

test('两个原始告警码不进界面，可读提示保留，且分流顺序稳定', () => {
  const split = splitClusteringWarnings([
    'POLARITY_GUARD_TRIGGERED',
    '簇内文本高度重复，结果仅供参考',
    'SPEAR_VIZ_NOT_COMPUTED',
  ]);
  // 界面只看到中文提示；两个英文码一个都不出现
  assert.deepEqual(split.visible, ['簇内文本高度重复，结果仅供参考']);
  assert.deepEqual(split.debug, ['POLARITY_GUARD_TRIGGERED', 'SPEAR_VIZ_NOT_COMPUTED']);
  for (const code of DEBUG_ONLY_CLUSTERING_WARNINGS) {
    assert.ok(!split.visible.includes(code), `界面不应出现 ${code}`);
  }
});

test('告警分流去重、忽略空项，空输入安全', () => {
  assert.deepEqual(splitClusteringWarnings(undefined), { visible: [], debug: [] });
  assert.deepEqual(splitClusteringWarnings(null), { visible: [], debug: [] });
  assert.deepEqual(splitClusteringWarnings([]), { visible: [], debug: [] });

  // 同一个码可能同时出现在作业级与结果级 warnings，去重避免 React key 冲突
  const duplicated = splitClusteringWarnings(['A', 'B', 'A', 'B']);
  assert.deepEqual(duplicated.visible, ['A', 'B']);
  const repeated = splitClusteringWarnings([
    'SPEAR_VIZ_NOT_COMPUTED',
    'SPEAR_VIZ_NOT_COMPUTED',
  ]);
  assert.deepEqual(repeated.debug, ['SPEAR_VIZ_NOT_COMPUTED']);

  const blank = splitClusteringWarnings(['', '   ', '有效提示']);
  assert.deepEqual(blank.visible, ['有效提示']);
});

test('被隐藏的告警码写进浏览器控制台，没有可隐藏项时不刷屏', () => {
  const calls: unknown[][] = [];
  const original = console.debug;
  console.debug = ((...args: unknown[]) => {
    calls.push(args);
  }) as typeof console.debug;
  try {
    logDebugWarnings('聚类结果', ['SPEAR_VIZ_NOT_COMPUTED', 'POLARITY_GUARD_TRIGGERED']);
    // 界面上看不到的码，控制台必须看得到，否则等于把信息直接丢掉
    assert.equal(calls.length, 1);
    const line = String(calls[0][0]);
    assert.match(line, /聚类结果/);
    assert.match(line, /SPEAR_VIZ_NOT_COMPUTED/);
    assert.match(line, /POLARITY_GUARD_TRIGGERED/);

    logDebugWarnings('聚类结果', []);
    assert.equal(calls.length, 1);
  } finally {
    console.debug = original;
  }
});

/* ---------- 外部指标（6 项）与"未净化对照" ---------- */

function datasetItem(id: string, metadata: Record<string, unknown>): ClusteringDatasetItem {
  return { id, text: `文本 ${id}`, metadata };
}

const BASELINE: ClusteringOfflineBaseline = {
  source: 'unit-test',
  fullRun: true,
  fullRunItemCount: 20198,
  baselineKey: 'nr0_legacy',
  targetKey: 'spear_purified_retrieval_purification_on',
  rows: {
    nr0_legacy: { ari: 0.155, vm: 0.59, fms: 0.16, ami: 0.45, hs: 0.585, cs: 0.595, nClusters: 271 },
    spear_purified_retrieval_purification_on: { ari: 0.281, nClusters: 231 },
    paper_report: { ari: 0.281, vm: 0.685, fms: 0.285, ami: 0.588, hs: 0.669, cs: 0.702, nClusters: 231 },
  },
  comparison: { ari: 0.126, fms: 0.125, nClusters: -40, ariRelative: 0.8129 },
  notes: ['归档说明'],
};

test('真值列必须每条样本都有才算数，否则返回 null', () => {
  assert.equal(
    detectLabelField([datasetItem('a', { category: '一类' }), datasetItem('b', { category: '二类' })]),
    'category',
  );
  // 大小写不敏感：上传的 CSV 可能写成 Category
  assert.equal(detectLabelField([datasetItem('a', { Category: 'A' })]), 'Category');
  assert.equal(detectLabelField([datasetItem('a', { 类别: '一类' })]), '类别');
  // 部分标注不能算：会得到一个偏移的指标
  assert.equal(
    detectLabelField([datasetItem('a', { category: '一类' }), datasetItem('b', {})]),
    null,
  );
  assert.equal(detectLabelField([datasetItem('a', { note: 'x' })]), null);
  assert.equal(detectLabelField([]), null);
});

test('指标取值缺失显示占位而不是 0', () => {
  assert.equal(formatMetricValue(0.281), '0.281');
  assert.equal(formatMetricValue(0), '0.000');
  assert.equal(formatMetricValue(null), '—');
  assert.equal(formatMetricValue(undefined), '—');
  assert.equal(formatMetricValue(Number.NaN), '—');
});

test('差值格式化：聚类数走整数、其余三位小数带正负号', () => {
  assert.equal(formatDelta('ari', 0.126), '+0.126');
  assert.equal(formatDelta('fms', -0.02), '-0.020');
  assert.equal(formatDelta('nClusters', -40), '-40');
  assert.equal(formatDelta('ari', null), '—');
});

test('指标表与对比表都固定输出 6 行，不夹带相对增益', () => {
  const rows = metricRows({ ari: 0.281, vm: 0.685, nClusters: 231 });
  assert.deepEqual(rows.map((row) => row.label), ['ARI', 'VM', 'FMS', 'AMI', 'HS', 'CS']);
  assert.equal(rows[0].display, '0.281');
  assert.equal(rows[2].display, '—');

  // ariRelative 即使存在也不渲染：界面不替观众把差值换算成"提升百分比"
  const deltas = comparisonRows({ ari: 0.126, ariRelative: 0.8129 });
  assert.equal(deltas.length, 6);
  assert.deepEqual(deltas.map((row) => row.key), ['ari', 'vm', 'fms', 'ami', 'hs', 'cs']);
  assert.equal(deltas[0].display, '+0.126');
});

test('真值来源说明带上命中的列与覆盖条数', () => {
  assert.equal(describeGroundTruth({ field: 'category', labeled: 20198, total: 20198 }), '标注列 category · 20198 / 20198 条');
  assert.equal(describeGroundTruth(null), '');
});

test('只有"整份测试集 + 没有实测指标"的历史作业才回退到论文归档', () => {
  const fullJob = job({ jobId: 'full', itemCount: 20198 });
  const smallJob = job({ jobId: 'small', itemCount: 50 });
  const noMetrics = { runId: 'r', totalSamples: 20198 } as unknown as Parameters<typeof shouldUseArchivedBenchmark>[2];
  const withMetrics = { metrics: { ari: 0.2 } } as unknown as Parameters<typeof shouldUseArchivedBenchmark>[2];

  assert.equal(shouldUseArchivedBenchmark(fullJob, BASELINE, noMetrics), true);
  // 抽样运行与全量归档不可比，绝不拿归档数字去"补"
  assert.equal(shouldUseArchivedBenchmark(smallJob, BASELINE, noMetrics), false);
  // 算出了实测值就永远优先于归档
  assert.equal(shouldUseArchivedBenchmark(fullJob, BASELINE, withMetrics), false);
  assert.equal(shouldUseArchivedBenchmark(fullJob, null, noMetrics), false);
  assert.equal(
    shouldUseArchivedBenchmark(fullJob, { ...BASELINE, fullRun: false }, noMetrics),
    false,
  );
});

test('归档对比列只取 基线 → 目标，不把论文报告值并排进来', () => {
  const columns = archivedBenchmarkColumns(BASELINE);
  assert.deepEqual(columns.map((column) => column.key), [
    'nr0_legacy',
    'spear_purified_retrieval_purification_on',
  ]);
  // 论文报告值来自论文的运行环境，和本机实测列并排会被误读，因此不进这张表
  assert.ok(!columns.some((column) => column.key === 'paper_report'));
  assert.equal(archivedColumnLabel('nr0_legacy'), '基线 nr0');
  // 论文报告值不在这张表里，映射表也不该认识它（认识就会有人把列加回来）
  assert.equal(archivedColumnLabel('paper_report'), 'paper_report');
  // 不认识的键回显键名，而不是显示空白
  assert.equal(archivedColumnLabel('mystery_row'), 'mystery_row');
  // 缺行时整列省略
  const partial = archivedBenchmarkColumns({ ...BASELINE, targetKey: 'missing' });
  assert.deepEqual(partial.map((column) => column.key), ['nr0_legacy']);
});

test('实测指标与对照的判定只看 summary 上有没有对应字段', () => {
  const metricSummary = { metrics: { ari: 0.2 } } as unknown as Parameters<typeof hasComputedMetrics>[0];
  const controlSummary = { metrics: { ari: 0.2 }, controlMetrics: { ari: 0.1 } } as unknown as Parameters<typeof hasComputedMetrics>[0];
  assert.equal(hasComputedMetrics(metricSummary), true);
  assert.equal(hasControlMetrics(metricSummary), false);
  assert.equal(hasControlMetrics(controlSummary), true);
  assert.equal(hasComputedMetrics(null), false);
});

test('对照运行选项以 camelCase 透传给同步 /run', async () => {
  const stub = stubFetch(
    jsonResponse({ success: true, data: { summary: {}, clusters: [], items: [], visualization: [] }, error: null }),
  );
  try {
    await runClustering([{ id: 'a', text: '未设置警戒围栏', metadata: {} }], {
      profileId: 'spear_purified',
      controlPurify: true,
    });
    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.equal(body.options.controlPurify, true);
  } finally {
    stub.restore();
  }
});

test('对照运行选项同样透传给异步作业提交', async () => {
  const stub = stubFetch(
    jsonResponse({
      success: true,
      data: { jobId: 'job_1', status: 'queued', itemCount: 1, detail: {}, warnings: [] },
      error: null,
    }),
  );
  try {
    await submitClusteringJob([{ id: 'a', text: '未设置警戒围栏', metadata: {} }], {
      profileId: 'spear_purified',
      controlPurify: true,
    });
    const body = JSON.parse(String(stub.calls[0].init?.body));
    assert.equal(body.options.controlPurify, true);
  } finally {
    stub.restore();
  }
});
