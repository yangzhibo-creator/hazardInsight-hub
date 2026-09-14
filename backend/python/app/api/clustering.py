"""聚类相关路由。

路由层只做「准入校验 + 委托」，业务逻辑全部在 `ClusteringGatewayService`
与 `SemanticJobService`；所有失败都通过抛出领域异常交给统一处理器，
路由里不写 `try/except` 拼错误 JSON。

同步与异步两条入口并存（对应实施计划 10.3 的"旧同步接口保持"）：

* `/run` —— 同步执行，行为与新增作业模型之前逐字一致，小样本即点即看；
* `/jobs/*` —— 提交后立刻返回 `jobId`，轮询状态、分页取明细、可硬取消。
  大样本必须走这条：它不会把 HTTP 请求挂在一次几十分钟的计算上。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_cluster_service, get_job_service
from app.core.config import get_settings
from app.core.errors import ClusteringTimeoutError, FileTooLargeError
from app.schemas.clustering import (
    AlgorithmsData,
    AlgorithmsEnvelope,
    BaselineData,
    BaselineEnvelope,
    ClusteringData,
    ClusteringEnvelope,
    ClusteringJobEnvelope,
    ClusteringJobRequest,
    ClusteringJobResultData,
    ClusteringJobResultEnvelope,
    ClusteringRunRequest,
    DatasetEnvelope,
    DatasetListData,
    DatasetListEnvelope,
    DatasetPreview,
    JobFilterOptions,
    JobFilterOptionsEnvelope,
    JobItemsData,
    JobItemsEnvelope,
    JobListData,
    JobListEnvelope,
    ProfilesData,
    ProfilesEnvelope,
    SampleData,
    SampleEnvelope,
)
from app.services.clustering_service import ClusteringGatewayService
from app.services.semantic_jobs import SemanticJobService
from app.utils.validators import validate_upload_extension

router = APIRouter(prefix="/clustering", tags=["clustering"])


def _normalize_filename(raw_name: str) -> str:
    """修正 multipart 文件名的编码。

    浏览器以 UTF-8 发送文件名，但 multipart 默认按 latin1 解码，含中文的文件名
    会变成乱码。这里按主项目 hazard-test 路由的既有做法还原一次。
    """

    if not raw_name:
        return "dataset"
    if any("\u4e00" <= char <= "\u9fff" for char in raw_name):
        return raw_name
    try:
        return raw_name.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw_name


@router.get("/profiles", response_model=ProfilesEnvelope, summary="列出可用 profile")
async def list_profiles(
    refresh: bool = Query(default=False, description="是否强制重新校验模型与 profile"),
    service: ClusteringGatewayService = Depends(get_cluster_service),
) -> ProfilesEnvelope:
    """返回 cluster-engine 中可通过 API 使用的配置档及其可用状态。"""

    profiles = await run_in_threadpool(service.list_profiles, refresh=refresh)
    return ProfilesEnvelope(success=True, data=ProfilesData(profiles=profiles), error=None)


@router.get("/algorithms", response_model=AlgorithmsEnvelope, summary="列出可用聚类算法")
async def list_algorithms(
    service: ClusteringGatewayService = Depends(get_cluster_service),
) -> AlgorithmsEnvelope:
    """返回引擎暴露的聚类算法及其依赖可用性。"""

    algorithms = await run_in_threadpool(service.list_algorithms)
    return AlgorithmsEnvelope(success=True, data=AlgorithmsData(algorithms=algorithms), error=None)


@router.get("/sample", response_model=SampleEnvelope, summary="读取内置示例数据")
async def sample_dataset(
    service: ClusteringGatewayService = Depends(get_cluster_service),
) -> SampleEnvelope:
    """返回随项目分发的示例数据集（真实核电工程隐患抽样）。"""

    payload = await run_in_threadpool(service.load_sample_dataset)
    return SampleEnvelope(
        success=True,
        data=SampleData(source_name=payload["source_name"], items=payload["items"]),
        error=None,
    )


@router.get("/baseline", response_model=BaselineEnvelope, summary="读取离线基准结果")
async def offline_baseline(
    service: ClusteringGatewayService = Depends(get_cluster_service),
) -> BaselineEnvelope:
    """返回已归档的指标，供实时计算失败或时间不足时兜底展示。

    响应里带 `source` / `verified_at` / `fullRun`，界面必须原样展示出处——
    归档数字与实时结果是两回事，不能混为一谈。
    """

    payload = await run_in_threadpool(service.load_offline_baseline)
    return BaselineEnvelope(success=True, data=BaselineData(**payload), error=None)


@router.post("/datasets", response_model=DatasetEnvelope, summary="上传并解析数据文件")
async def upload_dataset(
    file: UploadFile = File(..., description="CSV / XLSX / JSON / TXT"),
    service: ClusteringGatewayService = Depends(get_cluster_service),
    jobs: SemanticJobService = Depends(get_job_service),
) -> DatasetEnvelope:
    """解析上传的数据文件，规范成聚类样本列表（含自动识别文本列）。

    解析结果同时落成一份**数据集引用**并回传 `datasetId`：样本量大时，
    提交作业只需带上这个 ID，不必把上百 MB 的样本再传一遍。
    """

    settings = get_settings()
    filename = _normalize_filename(file.filename or "dataset")
    validate_upload_extension(filename)
    data = await file.read()
    if not data:
        raise FileTooLargeError("上传的文件内容为空。")
    if len(data) > settings.upload_max_bytes:
        raise FileTooLargeError(
            f"文件大小 {len(data) / 1024 / 1024:.1f} MB 超过上限 "
            f"{settings.upload_max_bytes / 1024 / 1024:.0f} MB。"
        )
    preview = await run_in_threadpool(service.parse_dataset, data, filename)
    reference = await run_in_threadpool(
        jobs.datasets.save,
        source_name=preview["source_name"],
        items=preview["items"],
        warnings=preview.get("warnings") or [],
    )
    return DatasetEnvelope(
        success=True,
        data=DatasetPreview(**preview, dataset_id=reference["dataset_id"]),
        error=None,
    )


@router.get("/datasets", response_model=DatasetListEnvelope, summary="列出数据集引用")
async def list_datasets(
    limit: int = Query(default=20, ge=1, le=100, description="返回条数上限"),
    jobs: SemanticJobService = Depends(get_job_service),
) -> DatasetListEnvelope:
    """列出已保存的数据集引用（新→旧），供前端选择"用哪份数据再跑一次"。"""

    datasets = await run_in_threadpool(jobs.datasets.list, limit=limit)
    return DatasetListEnvelope(success=True, data=DatasetListData(datasets=datasets), error=None)


@router.post("/run", response_model=ClusteringEnvelope, summary="执行一次聚类")
async def run_clustering(
    body: ClusteringRunRequest,
    service: ClusteringGatewayService = Depends(get_cluster_service),
) -> ClusteringEnvelope:
    """同步执行聚类并返回统计、簇摘要、逐条结果与二维可视化坐标。

    **样本量大请改用 `/jobs`。** 这条路径会把整份明细放进响应，并在 HTTP
    请求上等满整个计算过程；它的价值在于小样本"即点即看"，行为与新增作业
    模型之前完全一致。
    """

    settings = get_settings()
    payload = {
        "items": [item.model_dump() for item in body.items],
        "options": body.options.model_dump(),
    }
    try:
        data = await asyncio.wait_for(
            run_in_threadpool(service.run, payload),
            timeout=settings.timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        # 线程无法被强制取消，因此这里只中止等待；引擎侧任务结束后会自行释放单飞锁。
        # 需要"真的能停下"的场景请走 /jobs（它跑在独立子进程里，取消=杀进程）。
        raise ClusteringTimeoutError(
            f"聚类执行超过 {settings.timeout_seconds:.0f} 秒仍未返回，已中止等待。"
            "请减少样本数量、改用更轻量的 profile，或改用异步作业接口后重试。"
        ) from exc
    return ClusteringEnvelope(success=True, data=ClusteringData(**data), error=None)


# ------------------------------------------------------------------ 异步作业


@router.post("/jobs", response_model=ClusteringJobEnvelope, summary="提交异步聚类作业")
async def submit_job(
    body: ClusteringJobRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        description="幂等键；也可改用 options.idempotencyKey",
    ),
    jobs: SemanticJobService = Depends(get_job_service),
) -> ClusteringJobEnvelope:
    """提交一个聚类作业并立刻返回作业状态（`status=queued`）。

    提交只做准入与入队：解析 profile、清点样本、把样本快照落盘。
    真正的计算在独立子进程里进行，因此这条请求不会挂住。
    """

    payload = {
        "items": [item.model_dump() for item in body.items],
        "options": body.options.model_dump(),
    }
    key = idempotency_key or body.options.idempotency_key
    record = await run_in_threadpool(jobs.submit, payload, idempotency_key=key)
    return ClusteringJobEnvelope(success=True, data=record, error=None)


@router.get("/jobs", response_model=JobListEnvelope, summary="列出作业")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="返回条数上限"),
    status: str | None = Query(default=None, description="按状态过滤，如 running"),
    jobs: SemanticJobService = Depends(get_job_service),
) -> JobListEnvelope:
    """列出作业（新→旧）。前端据此恢复"上次那批跑到哪了"。"""

    records = await run_in_threadpool(jobs.list, limit=limit, status=status)
    return JobListEnvelope(success=True, data=JobListData(jobs=records), error=None)


@router.get("/jobs/{job_id}", response_model=ClusteringJobEnvelope, summary="查询作业状态")
async def get_job(
    job_id: str,
    jobs: SemanticJobService = Depends(get_job_service),
) -> ClusteringJobEnvelope:
    """轮询作业状态。`terminal=true` 时前端应停止轮询。"""

    record = await run_in_threadpool(jobs.get, job_id)
    return ClusteringJobEnvelope(success=True, data=record, error=None)


@router.post("/jobs/{job_id}/cancel", response_model=ClusteringJobEnvelope, summary="取消作业")
async def cancel_job(
    job_id: str,
    jobs: SemanticJobService = Depends(get_job_service),
) -> ClusteringJobEnvelope:
    """取消作业：排队中的直接出队；执行中的杀掉执行子进程（硬取消）。"""

    record = await run_in_threadpool(jobs.cancel, job_id)
    return ClusteringJobEnvelope(success=True, data=record, error=None)


@router.get(
    "/jobs/{job_id}/result",
    response_model=ClusteringJobResultEnvelope,
    summary="读取作业结果摘要",
)
async def get_job_result(
    job_id: str,
    jobs: SemanticJobService = Depends(get_job_service),
) -> ClusteringJobResultEnvelope:
    """返回统计、簇摘要、抽样坐标与明细元信息（**不含明细**）。"""

    data = await run_in_threadpool(jobs.summary, job_id)
    return ClusteringJobResultEnvelope(success=True, data=ClusteringJobResultData(**data), error=None)


@router.get(
    "/jobs/{job_id}/items",
    response_model=JobItemsEnvelope,
    summary="分页读取作业明细",
)
async def get_job_items(
    job_id: str,
    offset: int = Query(default=0, ge=0, description="起始位置（按筛选后的结果计）"),
    limit: int | None = Query(default=None, ge=1, le=500, description="本页条数，默认取配置值"),
    cluster_id: int | None = Query(default=None, description="只看某个簇；-1 为未归类桶"),
    assignment_status: str | None = Query(
        default=None,
        description="按归属状态过滤：core / borderline / small_coherent / duplicate_only / noise / invalid",
    ),
    keyword: str | None = Query(default=None, description="按文本关键词过滤（大小写不敏感）"),
    jobs: SemanticJobService = Depends(get_job_service),
) -> JobItemsEnvelope:
    """服务端分页读取明细。

    返回里的 `total` 是**筛选后的总数**，前端据此渲染真实页数——
    而不是"先拿 100 条再在本地假装总数就是 100"。
    按文本筛选时若命中了扫描上限，`truncated=true` 会如实说明结果不完整。
    """

    page = await run_in_threadpool(
        jobs.results,
        job_id,
        offset=offset,
        limit=limit,
        cluster_id=cluster_id,
        assignment_status=assignment_status,
        keyword=keyword,
    )
    return JobItemsEnvelope(success=True, data=JobItemsData(**page), error=None)


@router.get(
    "/jobs/{job_id}/filters",
    response_model=JobFilterOptionsEnvelope,
    summary="读取明细筛选取值",
)
async def get_job_filters(
    job_id: str,
    jobs: SemanticJobService = Depends(get_job_service),
) -> JobFilterOptionsEnvelope:
    """返回可见的簇及各自条数（只读索引，与样本量无关）。"""

    options = await run_in_threadpool(jobs.filter_options, job_id)
    return JobFilterOptionsEnvelope(success=True, data=JobFilterOptions(**options), error=None)
