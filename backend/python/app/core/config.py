"""网关配置。

沿用源项目 `medica-report-review` 的配置写法（pydantic-settings `BaseSettings`
+ `lru_cache` 单例），环境变量前缀为 `HAZARD_`。

这里**不**承载算法参数：算法与向量化口径在 `cluster-engine/configs/` 下，
由 `retrain_cluster.Settings` 负责加载。`engine_config` 只是指向那份配置的入口。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `backend/python` 目录。所有相对路径都以它为基准解析，与进程 cwd 无关。
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class GatewaySettings(BaseSettings):
    """聚类网关运行配置。"""

    app_name: str = "HazardInsight Clustering Gateway"
    app_version: str = "1.0.0"

    #: cluster-engine 的应用配置（相对 backend/python 解析，禁止绝对路径硬编码）
    engine_config: Path = Field(default=Path("cluster-engine") / "configs" / "app.toml")

    #: 内置示例数据（真实核电工程隐患抽样，由 scripts/build_sample_dataset.py 生成）
    sample_dataset: Path = Field(default=Path("samples") / "hazard_samples.json")

    #: 允许跨域的前端来源，逗号分隔。
    cors_origins: str = (
        "http://localhost:5175,http://127.0.0.1:5175,"
        "http://localhost:3001,http://127.0.0.1:3001"
    )

    #: 未显式指定 profile 时使用的兜底 profile；留空表示由网关自动挑选。
    default_profile_id: str | None = None

    #: 单次聚类请求的墙钟超时（秒）。超过则向前端返回 504。
    #: 放到 1 小时：30 万条样本的向量化 + 聚类远超 120 秒。
    #: 需与 cluster-engine 的 `app.toml:timeout_seconds` 保持一致，否则网关会先中止等待。
    timeout_seconds: float = 3600.0

    #: 上传数据文件大小上限（字节）。200 MiB，与引擎的 max_body_bytes 对齐。
    upload_max_bytes: int = 200 * 1024 * 1024

    #: 簇摘要（展示辅助）配置。
    digest_top_keywords: int = 6
    digest_representatives: int = 3

    # ------------------------------------------------------------------ 异步作业

    #: 作业与数据集的产物根目录（相对 backend/python 解析）。
    #:
    #: 与 cluster-engine 的 `artifacts/` 分开：引擎存的是"这次计算算出了什么"，
    #: 这里存的是"调用方提交的任务与它的展示态明细"，生命周期也不同
    #: （作业可以被取消并整体清理，引擎结果不该被任务状态牵着走）。
    job_store: Path = Field(default=Path("artifacts") / "clustering")

    #: 排队中 + 执行中的作业上限，超出即 429。
    #: 不是"越大越好"：每个作业最终都要占一份模型内存，队列长度只是把 OOM
    #: 从"立刻拒绝"推迟成"跑一半崩掉"。
    job_max_pending: int = 8

    #: 同时执行的作业数。聚类是 CPU/内存密集型，默认串行。
    #: 需要并行请显式调大，并同步评估模型内存 × 并发数。
    job_max_concurrent: int = 1

    #: 作业结果分页的默认页大小（前端可不传 limit 直接用这个值）。
    job_page_size: int = 100

    #: 保留的历史作业数（按创建时间倒序）。超出后清理最旧的**终态**作业，
    #: 活跃作业永远不清理——否则会把正在跑的任务目录删掉。
    job_history_limit: int = 50

    #: 单个数据集引用允许保留的样本数上限。
    dataset_max_items: int = 300_000

    #: 保留的数据集引用数量；超出后清理最旧的（不影响已提交的作业，
    #: 因为作业在提交时就把样本快照落盘了）。
    dataset_history_limit: int = 20

    # ------------------------------------------------------------------ 偏差数据库

    #: 知识库生成作业记录的根目录（相对 backend/python 解析）。
    #: 知识库本体写在 cluster-engine 的 `artifacts/knowledge_bases/<id>/`，
    #: 这里只存"哪次上传、跑到哪一步、生成了哪个库"这类任务态。
    knowledge_base_store: Path = Field(default=Path("artifacts") / "knowledge_bases")

    #: 单个知识库允许的条目上限。净化是逐条 LLM 推理，上限决定了单次生成的时间上界，
    #: 不设限会让"上传一个 30 万行的大文件"把现场演示拖成数小时。
    knowledge_base_max_items: int = 200_000

    #: 知识库生成作业的保留数量（新→旧，超出后清理最旧的终态作业）。
    knowledge_base_history_limit: int = 20

    #: 生成作业的净化分块大小：每处理完一块就落一次进度。
    #: 太大会让进度长时间不动，太小会放大调度开销。
    knowledge_base_purify_chunk: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HAZARD_",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """把逗号分隔的来源串解析为列表。"""

        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def resolve_path(self, path: Path) -> Path:
        """把相对路径解析为绝对路径（基准为 `backend/python`，而非进程 cwd）。"""

        return path if path.is_absolute() else (BACKEND_ROOT / path)

    @property
    def engine_config_path(self) -> Path:
        """cluster-engine 的 app.toml 绝对路径。"""

        return self.resolve_path(self.engine_config)

    @property
    def sample_dataset_path(self) -> Path:
        """内置示例数据文件绝对路径。"""

        return self.resolve_path(self.sample_dataset)

    @property
    def job_store_path(self) -> Path:
        """作业与数据集产物的绝对根目录。"""

        return self.resolve_path(self.job_store)

    @property
    def knowledge_base_store_path(self) -> Path:
        """知识库生成作业记录的绝对根目录。"""

        return self.resolve_path(self.knowledge_base_store)


@lru_cache
def get_settings() -> GatewaySettings:
    """返回网关配置单例（一次进程生命周期内只读一次环境变量）。"""

    return GatewaySettings()
