"""FastAPI 依赖。

沿用源项目 `medica-report-review/app/api/deps.py` 的写法：用 `lru_cache`
把服务对象做成进程内单例，避免每条请求都重建引擎与模型注册表。
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.clustering_service import ClusteringGatewayService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.semantic_jobs import SemanticJobService


@lru_cache
def get_cluster_service() -> ClusteringGatewayService:
    """返回聚类网关服务单例。"""

    return ClusteringGatewayService(get_settings())


@lru_cache
def get_job_service() -> SemanticJobService:
    """返回异步作业服务单例。

    作业服务拿到同一个网关服务实例，是为了让作业与同步 `/run` 共用一把执行槽位
    ——两者的资源瓶颈是同一份模型内存，各锁各的等于没有并发上限。
    """

    return SemanticJobService(get_settings(), service=get_cluster_service())


@lru_cache
def get_knowledge_base_service() -> KnowledgeBaseService:
    """返回偏差数据库服务单例。

    与聚类服务共享同一个引擎实例：同一份编码器缓存、同一份 Qwen 净化器。
    各自新建引擎会在显存里并存两份大模型。
    """

    return KnowledgeBaseService(get_settings(), get_cluster_service())
