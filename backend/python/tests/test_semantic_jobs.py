"""异步聚类作业的契约测试。

这里**不**跑真实模型：作业模型要保证的性质（生命周期、幂等、容量、硬取消、
重启恢复、分页连续）都与"语义质量"无关。执行器用一个同步的假 handler 替身，
因此这些断言在任何机器上都稳定，也不会因为换模型而失效。

真实子进程执行路径由 `cluster-engine/tests/api/test_execution.py` 覆盖
（那里测的是"杀进程能真的让阻塞调用退出"）。
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from app.core.config import GatewaySettings
from app.core.errors import (
    DatasetNotFoundError,
    DatasetTooLargeError,
    IdempotencyConflictError,
    JobCapacityError,
    JobNotCancellableError,
    JobNotFoundError,
    JobResultError,
)
from app.services.semantic_jobs import (
    InlineJobRunner,
    JobRecord,
    SemanticJobService,
    persist_job_result,
)

RUN_ID = "run_" + "b" * 32


# ------------------------------------------------------------------ 替身


def fake_result(items: list[dict], *, run_id: str = RUN_ID) -> dict:
    """构造一份与网关 `run()` 同形的结果（摘要 / 簇 / 明细 / 抽样坐标）。"""

    detail_items = []
    sizes: dict[int, int] = {}
    for index, item in enumerate(items):
        cluster_id = index % 2
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
        detail_items.append(
            {
                "id": item["id"],
                "text": item["text"],
                "cluster_id": cluster_id,
                "cluster_label": f"簇{cluster_id}",
                "confidence": 0.9,
                "distance": 0.1,
                "keywords": [],
                "metadata": dict(item.get("metadata") or {}),
                "assignment_status": "core" if index % 3 else "noise",
                "noise_reason": None if index % 3 else "below_semantic_support",
                "confidence_version": "semantic-confidence-v1",
                "distance_metric": "cosine",
            }
        )
    summary = {
        "run_id": run_id,
        "total_samples": len(items),
        "cluster_count": len(sizes),
        "noise_count": 0,
        "largest_cluster_size": max(sizes.values()),
        "smallest_cluster_size": min(sizes.values()),
        "avg_cluster_size": round(len(items) / max(1, len(sizes)), 4),
        "algorithm": "semantic_auto_kmeans",
        "profile_id": "fixture-semantic",
        "model_id": "fixture",
        "implementation_version": "semantic-v1",
        "embedding_dimension": 8,
        "cache_hit": True,
        "elapsed_ms": 12,
        "gateway_ms": 15,
        "warnings": [],
    }
    return {
        "summary": summary,
        "clusters": [
            {"cluster_id": cluster_id, "label": f"簇{cluster_id}", "size": size}
            for cluster_id, size in sorted(sizes.items())
        ],
        "items": detail_items,
        "visualization": [{"id": detail_items[0]["id"], "x": 0.0, "y": 0.0, "cluster_id": 0}],
        "warnings": [],
    }


def make_handler(settings: GatewaySettings, *, delay: float = 0.0):
    """替身执行处理器：落盘分片后返回瘦身结果（与子进程 worker 同形）。"""

    def handler(job_id: str, payload: dict) -> dict:
        if delay:
            time.sleep(delay)
        result = fake_result(payload["items"])
        detail = persist_job_result(settings.job_store_path / job_id / "result", result)
        return {
            "summary": result["summary"],
            "clusters": result["clusters"],
            "visualization": result["visualization"],
            "warnings": [],
            "detail": detail,
        }

    return handler


class BlockingRunner:
    """执行会一直阻塞到被中断的替身，用来测"执行中取消"与容量占用。"""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.interrupted = threading.Event()

    def submit(self, job_id: str, payload: dict) -> dict:
        self.started.set()
        # 模拟"子进程被杀"：等待中断信号后抛错，与服务层观察到的现象一致
        self.interrupted.wait(timeout=5)
        raise RuntimeError("execution worker was killed")

    def interrupt(self) -> bool:
        self.interrupted.set()
        return True

    def close(self) -> None:
        return None


# ------------------------------------------------------------------ 夹具


def build_settings(tmp_path, **overrides) -> GatewaySettings:
    params = {
        "job_store": tmp_path / "clustering",
        "job_max_pending": 8,
        "job_max_concurrent": 1,
        "job_page_size": 4,
        "job_history_limit": 10,
    }
    params.update(overrides)
    return GatewaySettings(**params)


def items(count: int) -> list[dict]:
    return [{"id": f"s{index}", "text": f"第 {index} 条隐患描述", "metadata": {"area": "A"}} for index in range(count)]


def wait_terminal(service: SemanticJobService, job_id: str, timeout: float = 5.0) -> dict:
    """轮询到终态；超时即失败——作业不该有"永远活着"的中间态。"""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = service.get(job_id)
        if snapshot["terminal"]:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"作业 {job_id} 未在 {timeout}s 内进入终态")


def write_record(root, job_id: str, **overrides) -> JobRecord:
    """把一条作业记录直接写到磁盘，用来模拟"上次进程留下的状态"。"""

    record = JobRecord(job_id=job_id, **overrides)
    directory = root / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.json").write_text(json.dumps(record.as_dict()), encoding="utf-8")
    return record


def write_request(root, job_id: str, payload_items: list[dict], options: dict | None = None) -> None:
    (root / job_id).mkdir(parents=True, exist_ok=True)
    (root / job_id / "request.json").write_text(
        json.dumps({"items": payload_items, "options": options or {}, "source_name": None}),
        encoding="utf-8",
    )


@pytest.fixture
def settings(tmp_path):
    return build_settings(tmp_path)


@pytest.fixture
def service(settings):
    """启动一个使用同进程替身执行器的作业服务。"""

    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    job_service.start()
    try:
        yield job_service
    finally:
        job_service.stop()


# ------------------------------------------------------------------ 生命周期


def test_job_runs_to_success_and_serves_summary(service):
    """提交 → 执行 → 成功：状态流转正确，摘要接口给出统计与簇。"""

    submitted = service.submit({"items": items(6), "options": {"profileId": "fixture-semantic"}})
    assert submitted["status"] in {"queued", "running"}

    final = wait_terminal(service, submitted["jobId"])
    assert final["status"] == "succeeded"
    assert final["jobId"] == submitted["jobId"]
    assert final["runId"] == RUN_ID
    assert final["detail"]["itemCount"] == 6
    assert final["error"] is None

    result = service.summary(submitted["jobId"])
    assert result["runId"] == RUN_ID
    assert result["summary"]["total_samples"] == 6
    assert len(result["clusters"]) == 2
    # 摘要接口刻意不含明细：看统计与翻明细是两条独立请求
    assert "items" not in result


def test_paging_is_server_side_and_consistent(service):
    """明细走服务端分页：total 是筛选后总数，连续翻页不重不漏。"""

    submitted = service.submit({"items": items(9), "options": {}})
    wait_terminal(service, submitted["jobId"])
    job_id = submitted["jobId"]

    first = service.results(job_id, offset=0, limit=4)
    assert first["total"] == 9
    assert first["returned"] == 4
    assert first["hasMore"] is True
    assert first["jobId"] == job_id

    second = service.results(job_id, offset=4, limit=4)
    third = service.results(job_id, offset=8, limit=4)
    seen = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert len(seen) == 9 and len(set(seen)) == 9
    assert third["hasMore"] is False


def test_cluster_filter_and_filter_options(service):
    """按簇筛选 + 筛选取值清单：两者必须对得上（否则下拉里的数字是假的）。"""

    submitted = service.submit({"items": items(8), "options": {}})
    wait_terminal(service, submitted["jobId"])
    job_id = submitted["jobId"]

    options = service.filter_options(job_id)
    assert options["itemCount"] == 8
    checked = 0
    for entry in options["clusterIds"]:
        page = service.results(job_id, offset=0, limit=100, cluster_id=entry["clusterId"])
        assert page["total"] == entry["size"]
        assert {item["cluster_id"] for item in page["items"]} == {entry["clusterId"]}
        checked += page["total"]
    assert checked == 8


def test_results_are_unavailable_before_success(service):
    """未成功就不该能读结果，且错误要说清"当前状态是什么"。"""

    submitted = service.submit({"items": items(3), "options": {}})
    service.cancel(submitted["jobId"])  # 排队阶段取消
    with pytest.raises(JobResultError) as excinfo:
        service.results(submitted["jobId"], offset=0, limit=1)
    assert "cancelled" in excinfo.value.message


def test_unknown_job_is_not_found(service):
    with pytest.raises(JobNotFoundError):
        service.get("job_" + "f" * 32)
    with pytest.raises(JobNotFoundError):
        service.get("../../etc/passwd")


# ------------------------------------------------------------------ 幂等与容量


def test_idempotency_key_returns_the_same_job(service):
    """同一把键 + 同一份请求 = 同一个作业（重试不会变成两个任务）。"""

    payload = {"items": items(4), "options": {"idempotencyKey": "k-1"}}
    first = service.submit(payload, idempotency_key="k-1")
    second = service.submit(payload, idempotency_key="k-1")
    assert first["jobId"] == second["jobId"]
    # metadata 不参与聚类，因此"只改备注"的重复提交不该被判成冲突
    third = service.submit(
        {"items": [dict(item, metadata={"note": "改了备注"}) for item in payload["items"]], "options": {}},
        idempotency_key="k-1",
    )
    assert third["jobId"] == first["jobId"]


def test_same_key_with_different_request_is_a_conflict(service):
    """同一个键配不同请求必须明确报冲突，而不是悄悄返回旧结果。"""

    service.submit({"items": items(4), "options": {}}, idempotency_key="k-2")
    with pytest.raises(IdempotencyConflictError):
        service.submit({"items": items(5), "options": {}}, idempotency_key="k-2")


def test_camelcase_reference_database_is_normalized_and_persisted(service):
    """`knowledgeBaseId` 必须归一到引擎口径，并原样写进请求快照。

    这是「聚类时选择参考数据库」的最后一公里：键不归一，覆盖就**静默失效**——
    作业照跑，只是悄悄用了 profile 的默认库，结果对不上却没有任何报错。
    """

    submitted = service.submit(
        {
            "items": items(2),
            "options": {"profileId": "fixture-semantic", "knowledgeBaseId": "kb-2026q1"},
        }
    )
    stored = json.loads(
        (service.settings.job_store_path / submitted["jobId"] / "request.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["options"]["knowledge_base_id"] == "kb-2026q1"
    assert "knowledgeBaseId" not in stored["options"]


def test_same_key_with_another_reference_database_is_a_conflict(service):
    """同一把幂等键 + 换了一个参考库 = 另一份请求，必须报冲突而不是复用旧作业。"""

    service.submit({"items": items(4), "options": {"knowledgeBaseId": "kb-a"}}, idempotency_key="k-kb")
    with pytest.raises(IdempotencyConflictError):
        service.submit(
            {"items": items(4), "options": {"knowledgeBaseId": "kb-b"}}, idempotency_key="k-kb"
        )
    # 同一个库的重复提交仍然应当拿回同一个作业
    again = service.submit(
        {"items": items(4), "options": {"knowledgeBaseId": "kb-a"}}, idempotency_key="k-kb"
    )
    assert again["jobId"]


def test_capacity_limit_rejects_beyond_queued_plus_running(tmp_path):
    """活跃作业数达到上限后必须拒绝——队列不是无限的，它只是把 OOM 往后推。"""

    settings = build_settings(tmp_path, job_max_pending=1, job_max_concurrent=1)
    runner = BlockingRunner()
    job_service = SemanticJobService(settings, runner=runner)
    job_service.start()
    try:
        job_service.submit({"items": items(2), "options": {}})
        assert runner.started.wait(timeout=5), "第一个作业应当已开始执行"
        with pytest.raises(JobCapacityError):
            job_service.submit({"items": items(2), "options": {}})
    finally:
        runner.interrupted.set()
        job_service.stop()


# ------------------------------------------------------------------ 取消


def test_cancel_queued_job_ends_immediately(service):
    """排队中的作业取消后立刻是终态，且不占用执行资源。"""

    submitted = service.submit({"items": items(2), "options": {}})
    snapshot = service.cancel(submitted["jobId"])
    assert snapshot["status"] == "cancelled"
    assert snapshot["terminal"] is True
    # hard_cancelled 只在真的中断了执行进程时才为 True
    assert snapshot["cancelRequested"] is True


def test_cancel_running_job_interrupts_the_executor(tmp_path):
    """执行中的取消必须落到执行器上（interrupt），不只是改个状态位。"""

    settings = build_settings(tmp_path)
    runner = BlockingRunner()
    job_service = SemanticJobService(settings, runner=runner)
    job_service.start()
    try:
        submitted = job_service.submit({"items": items(2), "options": {}})
        assert runner.started.wait(timeout=5), "作业应当已开始执行"
        job_service.cancel(submitted["jobId"])
        final = wait_terminal(job_service, submitted["jobId"])
        # 执行器抛错，但调用方意图是取消，因此终态是 cancelled 而不是 failed
        assert final["status"] == "cancelled"
        assert final["hardCancelled"] is True
        assert runner.interrupted.is_set()
    finally:
        job_service.stop()


def test_cancel_terminal_job_is_rejected(service):
    submitted = service.submit({"items": items(3), "options": {}})
    wait_terminal(service, submitted["jobId"])
    with pytest.raises(JobNotCancellableError):
        service.cancel(submitted["jobId"])


# ------------------------------------------------------------------ 数据集引用


def test_dataset_reference_replaces_inline_items(service):
    """按引用提交：请求体里只有 ID，样本从落盘的数据集读取。"""

    reference = service.datasets.save(source_name="隐患.csv", items=items(6), warnings=["TEXT_COLUMN_GUESSED"])
    submitted = service.submit({"options": {"datasetId": reference["dataset_id"]}})
    assert submitted["datasetId"] == reference["dataset_id"]
    assert submitted["itemCount"] == 6

    final = wait_terminal(service, submitted["jobId"])
    assert final["status"] == "succeeded"


def test_option_keys_accept_both_casings(service):
    """`profileId` 与 `profile_id` 必须落到同一个 profile。

    键名大小写写错**不会报错**：网关会把 `profileId` 当成未知键，于是请求悄悄
    走成"自动挑选 profile"这条完全不同的路径——结果对不上却没有任何日志。
    这里钉住两种写法等价。
    """

    camel = service.submit({"items": items(2), "options": {"profileId": "fixture-semantic"}})
    snake = service.submit({"items": items(2), "options": {"profile_id": "fixture-semantic"}})
    assert camel["profileId"] == snake["profileId"] == "fixture-semantic"


def test_missing_dataset_reference_is_not_found(service):
    with pytest.raises(DatasetNotFoundError):
        service.submit({"options": {"datasetId": "ds_" + "a" * 32}})


def test_dataset_over_limit_is_rejected(tmp_path):
    settings = build_settings(tmp_path, dataset_max_items=3)
    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    with pytest.raises(DatasetTooLargeError):
        job_service.datasets.save(source_name="大文件", items=items(4))


# ------------------------------------------------------------------ 重启恢复


def test_running_job_is_failed_after_restart(tmp_path):
    """执行中的作业在进程重启后必须离开 running——否则前端会永远转圈。"""

    settings = build_settings(tmp_path)
    job_id = "job_" + "1" * 32
    write_record(settings.job_store_path, job_id, status="running", item_count=4)

    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    job_service.start()
    try:
        snapshot = job_service.get(job_id)
        assert snapshot["status"] == "failed"
        assert snapshot["error"]["code"] == "GATEWAY_RESTARTED"
    finally:
        job_service.stop()


def test_running_job_with_complete_artifacts_is_promoted(tmp_path):
    """跑完但还没记账的作业应认定成功——manifest 是提交标记，它存在就说明数据完整。"""

    settings = build_settings(tmp_path)
    job_id = "job_" + "2" * 32
    persist_job_result(settings.job_store_path / job_id / "result", fake_result(items(4)))
    write_record(settings.job_store_path, job_id, status="running", run_id=RUN_ID, item_count=4)

    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    job_service.start()
    try:
        snapshot = job_service.get(job_id)
        assert snapshot["status"] == "succeeded"
        # 恢复出来的成功作业同样可以分页读明细
        page = job_service.results(job_id, offset=0, limit=2)
        assert page["total"] == 4
    finally:
        job_service.stop()


def test_queued_job_is_requeued_after_restart(tmp_path):
    """排队中的作业从未开始，重启后重新入队执行（重跑不产生半成品）。"""

    settings = build_settings(tmp_path)
    job_id = "job_" + "3" * 32
    write_record(settings.job_store_path, job_id, status="queued", item_count=4)
    write_request(settings.job_store_path, job_id, items(4))

    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    job_service.start()
    try:
        final = wait_terminal(job_service, job_id)
        assert final["status"] == "succeeded"
    finally:
        job_service.stop()


def test_terminal_jobs_are_pruned_but_active_ones_survive(tmp_path):
    """历史清理只删终态作业；活跃作业的目录必须留着，否则会删掉正在跑的任务。"""

    settings = build_settings(tmp_path, job_history_limit=1)
    job_service = SemanticJobService(settings, runner=InlineJobRunner(make_handler(settings)))
    job_service.start()
    try:
        first = job_service.submit({"items": items(2), "options": {}})
        wait_terminal(job_service, first["jobId"])
        second = job_service.submit({"items": items(2), "options": {}})
        wait_terminal(job_service, second["jobId"])

        assert not (settings.job_store_path / first["jobId"]).exists()
        assert (settings.job_store_path / second["jobId"]).exists()
    finally:
        job_service.stop()


def test_list_returns_newest_first(service):
    """列表按创建时间倒序，前端据此恢复"上次那批跑到哪了"。"""

    created = [service.submit({"items": items(2), "options": {}})["jobId"] for _ in range(3)]
    for job_id in created:
        wait_terminal(service, job_id)
    listed = [entry["jobId"] for entry in service.list(limit=10)]
    assert listed[:3] == list(reversed(created))
    assert all(entry["status"] == "succeeded" for entry in service.list(limit=10, status="succeeded"))
