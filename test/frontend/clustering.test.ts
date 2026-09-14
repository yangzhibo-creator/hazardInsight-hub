import assert from 'node:assert/strict';
import test from 'node:test';
import type { ClusteringClusterGroup, ClusteringResultItem } from '../../shared/clustering.js';
import {
  cancelClusteringJob,
  fetchClusteringBaseline,
  fetchClusteringHealth,
  fetchClusteringJob,
  fetchClusteringJobItems,
  fetchClusteringProfiles,
  runClustering,
  submitClusteringJob,
  uploadClusteringDataset,
} from '../../web/api/clustering.js';
import { describeJobStatus, isCancellable } from '../../web/components/ClusteringJobPanel.js';
import { MAX_VISIBLE, orderClusters } from '../../web/components/ClusterFilterBar.js';
import { NOISE_COLOR, clusterColor, clusterTint } from '../../web/lib/clusterColor.js';
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

test('job item paging sends filters as query params and never fetches everything', async () => {
  const stub = stubFetch(
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
