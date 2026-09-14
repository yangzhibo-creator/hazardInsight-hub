"""FastAPI 应用入口。

启动：

```bash
cd backend/python
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

装配内容：请求 ID 中间件 → CORS → 统一错误处理 → 路由（健康检查 + 聚类）。
聚类能力本身来自 `cluster-engine`（`retrain_cluster`），本文件不承载任何算法。
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.clustering import router as clustering_router
from app.api.deps import get_cluster_service, get_job_service, get_knowledge_base_service
from app.api.error_handlers import register_error_handlers
from app.api.health import router as health_router
from app.api.knowledge_bases import router as knowledge_bases_router
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """构造 FastAPI 应用（封装成函数便于测试注入与复用）。"""

    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """启动时预热引擎、拉起作业派发线程；关闭时回收执行子进程。

        启停顺序是有意的：先 `preload()`（后台线程，不阻塞启动），再启动作业派发。
        派发线程内部会做一次**重启恢复**——把上次遗留的 running 作业标成失败、
        把排队中的重新入队，否则刷新页面会看到一批永远停在 running 的僵尸作业。
        """

        logger.info("%s v%s 启动中…", settings.app_name, settings.app_version)
        get_cluster_service().preload()
        jobs = get_job_service()
        jobs.start()
        # 上次进程遗留的"生成中"知识库任务不会自己复活，标记为失败而不是让它永远转圈
        get_knowledge_base_service().recover_stale_jobs()
        try:
            yield
        finally:
            # 回收子进程：不这么做的话，长驻的 spawn 子进程会跟着 uvicorn
            # 一起变成孤儿，把模型内存一直占着。
            jobs.stop()
            logger.info("%s 已停止", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "HazardInsight Hub 聚类分析服务。算法与向量化由 cluster-engine "
            "(retrain-cluster) 提供，本服务负责协议转换与结果加工。"
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        """为每条请求分配 request_id，并在响应头回填，便于前后端对齐排查。"""

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    # 直连调试（不经 Node/Vite 代理）时需要 CORS；走代理时同源，不受影响。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )

    register_error_handlers(app)

    # 偏差数据库路由：/api/clustering/knowledge-bases/*
    # 放在聚类路由之前注册，避免将来新增的聚类通配路由抢先匹配到该前缀。
    app.include_router(knowledge_bases_router, prefix="/api")
    # 聚类路由：/api/clustering/*
    app.include_router(clustering_router, prefix="/api")
    # 健康检查：/api/health（规范要求）与 /api/clustering/health（便于前端代理）
    app.include_router(health_router, prefix="/api")
    app.include_router(health_router, prefix="/api/clustering", include_in_schema=False)

    return app


app = create_app()
