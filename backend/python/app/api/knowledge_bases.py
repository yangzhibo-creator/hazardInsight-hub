"""偏差数据库（聚类语义知识库）路由。

挂在 `/api/clustering/knowledge-bases` 之下，因此既走 Node 的聚类反向代理，
也在现场单容器（``app_entry.py`` 同时提供 API 与前端）下同源可用。

* ``GET  /``                列出已有知识库（元信息，不含条目内容）
* ``GET  /{id}``            查看单个知识库详情与一页条目
* ``POST /``                上传语料，提交异步生成任务（multipart）
* ``GET  /jobs``            列出生成任务
* ``GET  /jobs/{job_id}``   轮询生成进度

路由顺序有讲究：``/jobs*`` 必须注册在 ``/{kb_id}`` 之前，否则 ``GET /jobs``
会被当作 kb_id="jobs" 命中详情分支。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_knowledge_base_service
from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.core.errors import InvalidInputError
from app.schemas.clustering import (
    KnowledgeBaseBuildJob,
    KnowledgeBaseBuildJobEnvelope,
    KnowledgeBaseBuildJobListData,
    KnowledgeBaseBuildJobListEnvelope,
    KnowledgeBaseDetailData,
    KnowledgeBaseDetailEnvelope,
    KnowledgeBaseInfo,
    KnowledgeBaseListData,
    KnowledgeBaseListEnvelope,
)
from app.services.knowledge_base_service import KnowledgeBaseService
from app.utils.validators import validate_upload_extension

router = APIRouter(prefix="/clustering/knowledge-bases", tags=["knowledge-bases"])


def _normalize_filename(raw_name: str) -> str:
    """修正 multipart 文件名的编码（浏览器按 UTF-8 发，multipart 默认 latin1 解码）。

    与 `clustering.py` 的 `_normalize_filename` 同一口径：中文文件名不还原会变乱码。
    """

    if not raw_name:
        return "corpus.txt"
    if any("\u4e00" <= char <= "\u9fff" for char in raw_name):
        return raw_name
    try:
        return raw_name.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw_name


@router.get("", response_model=KnowledgeBaseListEnvelope, summary="列出偏差数据库")
async def list_knowledge_bases(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseListEnvelope:
    """列出 cluster-engine 中已构建的知识库（新→旧）。"""

    items = await run_in_threadpool(service.list)
    return KnowledgeBaseListEnvelope(
        success=True,
        data=KnowledgeBaseListData(knowledge_bases=[KnowledgeBaseInfo(**item) for item in items]),
        error=None,
    )


@router.get("/jobs", response_model=KnowledgeBaseBuildJobListEnvelope, summary="列出生成任务")
async def list_build_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="返回条数上限"),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseBuildJobListEnvelope:
    """列出知识库生成任务（新→旧），供页面恢复"上次那个传到哪了"。"""

    jobs = await run_in_threadpool(service.list_jobs, limit=limit)
    return KnowledgeBaseBuildJobListEnvelope(
        success=True,
        data=KnowledgeBaseBuildJobListData(jobs=[KnowledgeBaseBuildJob(**job) for job in jobs]),
        error=None,
    )


@router.get("/jobs/{job_id}", response_model=KnowledgeBaseBuildJobEnvelope, summary="查询生成进度")
async def get_build_job(
    job_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseBuildJobEnvelope:
    """轮询生成任务状态；`terminal=true` 时前端应停止轮询。"""

    job = await run_in_threadpool(service.get_job, job_id)
    return KnowledgeBaseBuildJobEnvelope(success=True, data=KnowledgeBaseBuildJob(**job), error=None)


@router.get("/{kb_id}", response_model=KnowledgeBaseDetailEnvelope, summary="查看知识库详情")
async def get_knowledge_base(
    kb_id: str,
    offset: int = Query(default=0, ge=0, description="条目起始位置"),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="本页条目数"),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseDetailEnvelope:
    """返回知识库元信息与一页条目内容。

    `entriesSource` 说明条目从哪里读到（构建快照 / 上传语料 / 历史来源 / 不可用）：
    历史库若没有文本快照，页面必须如实说明"只能看元信息"，而不是显示一个空列表。
    """

    detail = await run_in_threadpool(service.detail, kb_id, offset=offset, limit=limit)
    return KnowledgeBaseDetailEnvelope(
        success=True, data=KnowledgeBaseDetailData(**detail), error=None
    )


@router.post("", response_model=KnowledgeBaseBuildJobEnvelope, summary="上传语料生成知识库")
async def build_knowledge_base(
    file: UploadFile = File(..., description="语料文件：TXT（每行一条）/ CSV / XLSX / JSON"),
    name: str = Form(default="", description="知识库名称（可读名，写进清单）"),
    mode: str = Form(default="purified", description="purified=大模型净化后入库；raw=原文直接入库"),
    ident: str | None = Form(default=None, description="可选：自定义知识库标识（ASCII）"),
    model_id: str | None = Form(default=None, description="可选：指定向量模型，默认与检索 profile 一致"),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseBuildJobEnvelope:
    """上传语料并提交一次生成任务，立刻返回 `jobId`。

    生成是逐条 LLM 推理 + 向量化 + 建索引，可能持续数分钟到数小时，因此
    不在这条请求里同步等待；请轮询 `/jobs/{jobId}` 获取进度。
    """

    filename = _normalize_filename(file.filename or "corpus.txt")
    validate_upload_extension(filename)
    data = await file.read()
    if not data:
        raise InvalidInputError("上传的语料文件内容为空。")

    job = await run_in_threadpool(
        service.submit_build,
        data=data,
        filename=filename,
        display_name=(name or "").strip(),
        kind=(mode or "purified").strip(),
        ident=(ident or "").strip() or None,
        model_id=(model_id or "").strip() or None,
    )
    return KnowledgeBaseBuildJobEnvelope(success=True, data=KnowledgeBaseBuildJob(**job), error=None)
