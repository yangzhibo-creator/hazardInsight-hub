"""异步聚类作业：提交、排队、执行、取消与分页读结果。

**为什么要有它。** 同步 `/run` 把「执行」和「等待」绑在一条 HTTP 请求上：
30 万条样本要跑几十分钟，期间前端只能干等，浏览器刷新就等于丢掉任务，
想停也停不下来（请求已经发出去了）。作业模型把这两件事拆开：

* 提交只做准入与入队，立刻返回 `job_id`；
* 执行在**独立的 spawn 子进程**里跑，网关主进程始终可控；
* 明细按分片落盘，前端翻页取数，不必把整份结果下载下来。

**取消为什么能"真停"。** 聚类跑的是 numpy/sklearn 紧循环，Python 层没有抢占点，
线程也杀不掉。因此取消实现为"杀掉执行子进程"——这是唯一能保证资源被释放的做法
（`retrain_cluster.api.execution.SyncExecutor.interrupt`）。作业状态里「取消」是
独立状态而不是失败的一种：失败要人查原因，取消是调用方的意图。

**两层容量。** 排队上限 `job_max_pending` 与并行上限 `job_max_concurrent` 分开：
前者防止队列无限增长，后者防止多个作业同时各占一份模型内存。默认并行 1。

**幂等。** 带 `idempotencyKey` 重复提交同一份请求会拿回同一个作业（前端重试、
代理重发都不会变成两个任务）；同一个键配不同请求则明确报冲突，而不是悄悄返回旧结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid

from app.core.config import GatewaySettings, get_settings
from app.core.constants import (
    DEFAULT_PAGE_SIZE,
    JOB_ACTIVE_STATUSES,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JOB_TERMINAL_STATUSES,
    MAX_PAGE_SIZE,
)
from app.core.errors import (
    ClusterGatewayError,
    DatasetNotFoundError,
    DatasetTooLargeError,
    InvalidInputError,
    JobCapacityError,
    JobNotCancellableError,
    JobNotFoundError,
    JobResultError,
    IdempotencyConflictError,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

_JOB_ID_PATTERN = re.compile(r"job_[0-9a-f]{32}")
_DATASET_ID_PATTERN = re.compile(r"ds_[0-9a-f]{32}")

#: 选项键的别名表：键名去掉下划线并小写后 -> 内部 snake_case 键名。
#: 只登记"确实有两种写法在流通"的键（HTTP camelCase 与引擎 snake_case）。
_OPTION_ALIASES = {
    "profileid": "profile_id",
    "datasetid": "dataset_id",
    "idempotencykey": "idempotency_key",
    "reducemethod": "reduce_method",
    # 参考数据库（偏差数据库）：漏了这一条时 camelCase 的 `knowledgeBaseId`
    # 会被原样保留，而下游读的是 `knowledge_base_id`——于是覆盖**静默失效**，
    # 作业照跑，只是用了 profile 的默认库。这是最难查的一类问题。
    "knowledgebaseid": "knowledge_base_id",
}


# ------------------------------------------------------------------ 小工具


def _now() -> float:
    """当前时间戳（epoch 秒）。用 epoch 而不是 monotonic：作业要跨进程重启存活。"""

    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _iso(timestamp: float | None) -> str | None:
    """epoch 秒 → UTC ISO 字符串；缺失即为 None（前端不必猜"0 是什么时候"）。"""

    if not timestamp:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def _normalize_options(options: dict) -> dict:
    """把选项键统一成网关/引擎内部的 snake_case 口径。

    同时接受 camelCase 与 snake_case。HTTP 契约是 camelCase，网关内部与引擎一律
    snake_case，因此在边界处归一化。不这么做的话，`profileId` 与 `profile_id`
    会被当成两个不同的键：写错的那个**不会报错**，只会让请求悄悄跑成自动挑选的
    另一个 profile——这是最难查的一类问题（结果对不上，但没有一行日志说哪里错）。
    """

    normalized: dict = {}
    for key, value in options.items():
        canonical = _OPTION_ALIASES.get(str(key).replace("_", "").lower(), key)
        normalized[canonical] = value
    return normalized


def _request_fingerprint(options: dict, items: list[dict], dataset_id: str | None = None) -> str:
    """请求指纹：用来判断"同一个幂等键是不是同一份请求"。

    参与指纹的是**会改变结果**的口径：profile / 算法 / 参考数据库 / 净化开关，
    以及 (id, text) 序列；不含 metadata 与耗时类字段——metadata 不参与聚类，
    把它算进来会让"只改备注"的重复提交被误判为冲突。

    参考数据库与净化开关必须参与：它们是"换个库、换条链路"的一级开关，
    漏掉的话换库重跑会拿回上一个库的作业，用户看到的结果与所选库对不上。

    只给了 `datasetId` 时用引用 ID 代替样本：本函数要在**读取数据集之前**就能算出结果，
    否则一个被清理掉的数据集会让"重复提交"抛 404，而不是正常返回原作业。
    """

    digest = hashlib.sha256()
    digest.update((options.get("profile_id") or "").encode("utf-8"))
    digest.update(b"\x1b")
    digest.update((options.get("algorithm") or "").encode("utf-8"))
    digest.update(b"\x1b")
    digest.update((options.get("knowledge_base_id") or "").encode("utf-8"))
    digest.update(b"\x1b")
    digest.update(("purify=on" if options.get("purify") is True else "purify=off" if options.get("purify") is False else "").encode("utf-8"))
    if not items and dataset_id:
        digest.update(b"\x1d")
        digest.update(dataset_id.encode("utf-8"))
        return digest.hexdigest()
    for item in items:
        digest.update(b"\x1f")
        digest.update(str(item.get("id", "")).encode("utf-8"))
        digest.update(b"\x1e")
        digest.update(str(item.get("text", "")).encode("utf-8"))
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    """原子写 JSON：先写临时文件再替换，避免半截文件被后续读取方当成完整记录。"""

    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


# ------------------------------------------------------------------ 数据集引用


class DatasetStore:
    """数据集引用：把解析好的样本落盘，让作业提交不必再传一遍全量样本。

    没有它的话，10 万条样本的提交请求体是上百 MB 的 JSON——每次重试都要重传，
    代理与网关都得为它准备一个与样本量同阶的缓冲区。有了引用，提交请求只剩一个 ID。
    """

    def __init__(self, directory: Path, *, max_items: int, history_limit: int):
        self.directory = Path(directory)
        self.max_items = max_items
        self.history_limit = history_limit
    def _path(self, dataset_id: str) -> Path:
        if not isinstance(dataset_id, str) or not _DATASET_ID_PATTERN.fullmatch(dataset_id):
            raise DatasetNotFoundError("数据集引用不存在或已过期。")
        return self.directory / f"{dataset_id}.json"

    def save(self, *, source_name: str, items: list[dict], warnings: list[str] | None = None) -> dict:
        """保存一份数据集，返回其元信息（含 datasetId）。"""

        if len(items) > self.max_items:
            raise DatasetTooLargeError(f"数据集包含 {len(items)} 条样本，超过上限 {self.max_items} 条。")
        dataset_id = _new_id("ds")
        payload = {
            "dataset_id": dataset_id,
            "source_name": source_name,
            "total": len(items),
            "warnings": list(warnings or []),
            "created_at": _now(),
            "items": items,
        }
        _atomic_json(self._path(dataset_id), payload)
        self._prune()
        return {
            "dataset_id": dataset_id,
            "source_name": source_name,
            "total": len(items),
            "warnings": list(warnings or []),
            "created_at": _iso(payload["created_at"]),
        }

    def load(self, dataset_id: str) -> dict:
        """读回数据集；文件缺失/损坏统一转成"引用不存在"。"""

        path = self._path(dataset_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise DatasetNotFoundError("数据集引用不存在或已过期。") from None
        except (ValueError, OSError):
            raise DatasetNotFoundError("数据集引用已损坏，请重新上传。") from None

    def items(self, dataset_id: str) -> list[dict]:
        return list(self.load(dataset_id).get("items") or [])

    def list(self, *, limit: int = 20) -> list[dict]:
        """按创建时间倒序列出已知数据集引用（只读索引，不解码样本）。"""

        entries = []
        if not self.directory.is_dir():
            return entries
        for path in self.directory.glob("ds_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue  # 损坏的历史引用不该让列表接口整体失败
            entries.append(
                {
                    "dataset_id": payload.get("dataset_id") or path.stem,
                    "source_name": payload.get("source_name"),
                    "total": int(payload.get("total") or 0),
                    "warnings": list(payload.get("warnings") or []),
                    "created_at": _iso(payload.get("created_at")),
                }
            )
        entries.sort(key=lambda item: item["created_at"] or "", reverse=True)
        return entries[:limit]

    def _prune(self) -> None:
        """按创建时间倒序保留最近 N 份，多余的删除。"""

        entries = self.list(limit=10_000)
        for stale in entries[self.history_limit :]:
            try:
                self._path(stale["dataset_id"]).unlink(missing_ok=True)
            except ClusterGatewayError:
                continue


# ------------------------------------------------------------------ 作业记录


@dataclass
class JobRecord:
    """作业的可持久化状态。字段改动必须同时改 `from_dict` 的容错读取。"""

    job_id: str
    status: str = JOB_QUEUED
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    started_at: float | None = None
    finished_at: float | None = None

    profile_id: str | None = None
    algorithm: str | None = None
    implementation_version: str | None = None
    dataset_id: str | None = None
    item_count: int = 0

    idempotency_key: str | None = None
    request_fingerprint: str | None = None

    cancel_requested: bool = False
    #: 取消是否真的落到了进程上（子进程执行器为 True，同进程执行器为 False）。
    hard_cancelled: bool = False

    run_id: str | None = None
    detail: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    error_code: str | None = None
    error_message: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in JOB_TERMINAL_STATUSES

    def touch(self, status: str) -> None:
        self.status = status
        self.updated_at = _now()

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> dict:
        """对外描述（camelCase，含可读时间与进度口径）。"""

        return {
            "jobId": self.job_id,
            "status": self.status,
            "createdAt": _iso(self.created_at),
            "updatedAt": _iso(self.updated_at),
            "startedAt": _iso(self.started_at),
            "finishedAt": _iso(self.finished_at),
            "profileId": self.profile_id,
            "algorithm": self.algorithm,
            "implementationVersion": self.implementation_version,
            "datasetId": self.dataset_id,
            "itemCount": self.item_count,
            "cancelRequested": self.cancel_requested,
            "hardCancelled": self.hard_cancelled,
            "runId": self.run_id,
            "detail": dict(self.detail),
            "warnings": list(self.warnings),
            "error": (
                {"code": self.error_code, "message": self.error_message} if self.error_code else None
            ),
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "JobRecord":
        """从磁盘容错读取：字段缺失用默认值，未知字段忽略。

        容错是有意的——升级后旧记录可能少几个字段，让整个作业列表报错
        比丢掉一条历史记录糟糕得多。
        """

        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


# ------------------------------------------------------------------ 结果落盘


def persist_job_result(directory: Path, result: dict) -> dict:
    """把一次执行的结果写成"摘要 + 分片明细"，返回供分页用的元信息。

    这一步是作业路径与同步路径的关键差别：同步接口把明细整份放进 HTTP 响应，
    作业则把它写到磁盘、只回传一个引用。调用方（子进程）因此可以把大对象
    在本地消化掉，不必通过 Pipe 搬运上百 MB 的明细。
    """

    from retrain_cluster.artifacts.semantic_runs import SemanticRunStore

    summary = dict(result.get("summary") or {})
    run_id = summary.get("run_id") or result.get("run_id")
    if not run_id:
        raise JobResultError("执行结果缺少 run_id，无法建立分页索引。")
    store = SemanticRunStore(Path(directory))
    committed = store.save(
        summary=summary,
        clusters=list(result.get("clusters") or []),
        aggregate=dict(result.get("aggregate") or {}),
        items=list(result.get("items") or []),
        visualization=list(result.get("visualization") or []),
        manifest={
            "schema_version": 3,
            "run_id": run_id,
            "profile_id": summary.get("profile_id"),
            "algorithm": summary.get("algorithm"),
            "implementation_version": summary.get("implementation_version"),
            "warnings": list(result.get("warnings") or summary.get("warnings") or []),
        },
    )
    return {
        "runId": run_id,
        "itemCount": int(committed["item_count"]),
        "shardCount": int(committed["shard_count"]),
        "shardSize": int(committed["shard_size"]),
    }


def run_job_payload(service, job_dir: Path, payload: dict) -> dict:
    """执行一次作业并落盘明细，返回瘦身后的结果。

    `service` 是 `ClusteringGatewayService`。这里刻意只依赖它的 `run()`，
    因此同步路径与作业路径共用同一份"引擎结果 → 网关契约"的映射逻辑，
    不会出现两条入口对同一份引擎输出给出不同摘要的漂移。
    """

    result = service.run(payload)
    detail = persist_job_result(job_dir / "result", result)
    return {
        "summary": dict(result.get("summary") or {}),
        "clusters": list(result.get("clusters") or []),
        "visualization": list(result.get("visualization") or []),
        "warnings": list(result.get("warnings") or []),
        "detail": detail,
    }


# ------------------------------------------------------------------ 执行器


def job_worker_main(connection, settings) -> None:
    """spawn 子进程主循环：构造网关服务，循环接收作业并回传瘦身结果。

    只在子进程里加载引擎与模型：主进程因此不会被模型加载或算法崩溃带走，
    取消也只需要杀掉这个进程。协议与 `retrain_cluster.api.execution.worker_main`
    保持一致（`ready` 握手 + `ok`/`error` 二元应答），便于两端复用同一套执行器。
    """

    try:
        from app.services.clustering_service import ClusteringGatewayService

        service = ClusteringGatewayService(settings)
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:  # 约定：收到 None 表示退出
                break
            job_id, payload = request
            try:
                outcome = run_job_payload(service, settings.job_store_path / job_id, payload)
                connection.send(("ok", outcome))
            except Exception as exc:  # noqa: BLE001 - 业务/未知异常都只回传码与消息
                code = getattr(exc, "code", None) or "CLUSTERING_FAILED"
                status = getattr(exc, "status", 500)
                message = getattr(exc, "message", None) or str(exc) or "Clustering execution failed"
                connection.send(("error", (code, message, status)))
    except (EOFError, BrokenPipeError):
        pass  # 父进程已退出，正常收场
    finally:
        connection.close()


class SubprocessJobRunner:
    """在常驻 spawn 子进程中执行作业；取消 = 杀子进程（硬取消）。"""

    def __init__(self, settings: GatewaySettings, worker=job_worker_main):
        from retrain_cluster.api.execution import SyncExecutor

        self.executor = SyncExecutor(settings, worker=worker)

    def submit(self, job_id: str, payload: dict) -> dict:
        return self.executor.execute((job_id, payload))

    def interrupt(self) -> bool:
        return bool(self.executor.interrupt())

    def close(self) -> None:
        self.executor.close()


class InlineJobRunner:
    """同进程执行作业，供测试与"子进程不可用"时的降级使用。

    它**没有硬取消能力**：同进程执行只有协作式取消，`interrupt()` 恒返回 False，
    服务层据此把"已请求但没落地"如实标出来，而不是假装取消生效了。
    """

    def __init__(self, handler):
        self.handler = handler

    def submit(self, job_id: str, payload: dict) -> dict:
        return self.handler(job_id, payload)

    def interrupt(self) -> bool:
        return False

    def close(self) -> None:
        return None


def inline_handler(settings: GatewaySettings):
    """构造同进程执行处理器（惰性建服务，避免导入模块就加载引擎）。"""

    holder: dict = {}

    def handler(job_id: str, payload: dict) -> dict:
        from app.services.clustering_service import ClusteringGatewayService

        service = holder.get("service")
        if service is None:
            service = holder["service"] = ClusteringGatewayService(settings)
        return run_job_payload(service, settings.job_store_path / job_id, payload)

    return handler


# ------------------------------------------------------------------ 作业服务


class SemanticJobService:
    """作业的生命周期管理：入队、派发、取消、分页与历史清理。

    线程模型只有两个：
      * 提交线程（HTTP 请求线程）只写记录并入队；
      * **一个**派发线程按 `job_max_concurrent` 取作业执行。
    没有线程池：聚类作业的重资源是"模型内存"，线程多了只会让 OOM 来得更早。
    """

    def __init__(
        self,
        settings: GatewaySettings | None = None,
        *,
        service=None,
        runner=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.job_store_path
        self._service = service  # ClusteringGatewayService，仅用于占用执行槽位
        self._runner = runner
        self.datasets = DatasetStore(
            self.root / "datasets",
            max_items=self.settings.dataset_max_items,
            history_limit=self.settings.dataset_history_limit,
        )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._records: dict[str, JobRecord] = {}
        self._queue: list[str] = []
        self._active = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 路径

    def _job_dir(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
            raise JobNotFoundError("作业不存在或已被清理。")
        return self.root / job_id

    def _record_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    # ------------------------------------------------------------------ 生命周期

    def start(self) -> None:
        """启动派发线程并做一次重启恢复。可重复调用（幂等）。"""

        with self._lock:
            self._recover_locked()
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._dispatch_loop, name="clustering-jobs", daemon=True)
            self._thread.start()
            logger.info("聚类作业派发线程已启动（并行上限 %s）", self.settings.job_max_concurrent)

    def stop(self, *, timeout: float = 5.0) -> None:
        """停掉派发线程与执行子进程；正在跑的作业保持 running 状态留给下次恢复判定。"""

        with self._lock:
            self._stop.set()
            self._cv.notify_all()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        runner = self._runner
        if runner is not None:
            try:
                runner.close()
            except Exception as exc:  # noqa: BLE001 - 关停失败不应影响进程退出
                logger.warning("关闭作业执行器失败：%s", exc)

    # ------------------------------------------------------------------ 提交

    def submit(self, payload: dict, *, idempotency_key: str | None = None) -> dict:
        """提交一个作业，返回其完整状态（含 jobId）。

        `payload` 形如 `{"items": [...], "options": {...}}`，与同步 `/run` 同构；
        也可以只给 `options.datasetId` 引用一份已落盘的数据集。
        """

        options = _normalize_options(payload.get("options") or {})
        key = (idempotency_key or "").strip() or None
        profile_id = options.get("profile_id")
        dataset_id = options.get("dataset_id")
        raw_items = list(payload.get("items") or [])

        # 幂等判定放在读取数据集之前：重复提交一个数据集已被清理的请求，
        # 应当拿回原作业，而不是 404。
        fingerprint = _request_fingerprint(options, raw_items, dataset_id)
        with self._lock:
            existing = self._find_by_key_locked(key)
            if existing is not None:
                if existing.request_fingerprint and fingerprint != existing.request_fingerprint:
                    raise IdempotencyConflictError(
                        "该幂等键已用于另一份不同的请求；请为新的请求换一个键。"
                    )
                return existing.describe()

        items, resolved_dataset_id, source_name = self._resolve_items(payload)
        if not items:
            raise InvalidInputError("没有可提交的样本，请提供 items 或 datasetId。")

        with self._lock:
            # 同一个键的并发提交：先到者赢，后到者拿到同一个作业
            if key:
                raced = self._find_by_key_locked(key)
                if raced is not None:
                    return raced.describe()
            if self._active_count_locked() >= self.settings.job_max_pending:
                raise JobCapacityError(
                    f"排队与执行中的作业已达上限 {self.settings.job_max_pending} 个，请稍后重试或先取消已有作业。"
                )
            record = JobRecord(
                job_id=_new_id("job"),
                profile_id=profile_id,
                dataset_id=resolved_dataset_id,
                item_count=len(items),
                idempotency_key=key,
                request_fingerprint=fingerprint,
            )
            self._records[record.job_id] = record
            self._persist_locked(record)
            # 样本快照落盘：数据集引用之后被清理也不影响这个作业
            _atomic_json(
                self._job_dir(record.job_id) / "request.json",
                {"items": items, "options": options, "source_name": source_name},
            )
            self._queue.append(record.job_id)
            self._cv.notify_all()
            logger.info("作业已入队：%s（%d 条样本）", record.job_id, len(items))
            return record.describe()

    def _resolve_items(self, payload: dict) -> tuple[list[dict], str | None, str | None]:
        """把请求解析成 (样本列表, datasetId, 来源名)。

        显式给了 items 就以 items 为准；只给 datasetId 时从数据集引用读取。
        """

        options = _normalize_options(payload.get("options") or {})
        dataset_id = options.get("dataset_id")
        raw_items = payload.get("items")
        if raw_items:
            items = [
                {
                    "id": str(item.get("id") or f"row-{index + 1}"),
                    "text": "" if item.get("text") is None else str(item.get("text")),
                    "metadata": dict(item.get("metadata") or {}),
                }
                for index, item in enumerate(raw_items)
            ]
            if len({item["id"] for item in items}) != len(items):
                raise InvalidInputError("样本 ID 必须唯一，请检查输入数据。")
            return items, None, None
        if dataset_id:
            payload_out = self.datasets.load(str(dataset_id))
            return list(payload_out.get("items") or []), str(dataset_id), payload_out.get("source_name")
        return [], None, None

    def _find_by_key_locked(self, key: str | None) -> JobRecord | None:
        if not key:
            return None
        for record in self._records.values():
            if record.idempotency_key == key:
                return record
        return None

    def _active_count_locked(self) -> int:
        return sum(1 for record in self._records.values() if record.status in JOB_ACTIVE_STATUSES)

    # ------------------------------------------------------------------ 查询

    def get(self, job_id: str) -> dict:
        """返回作业状态；内存里没有就尝试从磁盘恢复（网关重启后仍可查询）。"""

        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                record = self._load_locked(job_id)
            if record is None:
                raise JobNotFoundError("作业不存在或已被清理。")
            return record.describe()

    def list(self, *, limit: int = 20, status: str | None = None) -> list[dict]:
        """按创建时间倒序返回作业列表。"""

        with self._lock:
            self._load_all_locked()
            records = [
                record
                for record in self._records.values()
                if status is None or record.status == status
            ]
        records.sort(key=lambda item: item.created_at, reverse=True)
        return [record.describe() for record in records[:limit]]

    def summary(self, job_id: str) -> dict:
        """返回成功作业的摘要 / 簇 / 抽样坐标 / 明细分页元信息（都不含明细本体）。"""

        record = self._require_record(job_id)
        store, run_id = self._open_store(record)
        loaded = store.load(run_id)
        return {
            "jobId": record.job_id,
            "runId": run_id,
            "summary": loaded["summary"],
            "clusters": loaded["clusters"],
            "aggregate": loaded["aggregate"],
            "visualization": loaded["visualization"],
            "detail": dict(record.detail),
            "warnings": list(record.warnings),
        }

    def results(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        cluster_id: int | None = None,
        assignment_status: str | None = None,
        keyword: str | None = None,
    ) -> dict:
        """分页读取明细（服务端分页；前端永远只拿到一页）。"""

        record = self._require_record(job_id)
        store, run_id = self._open_store(record)
        default_limit = self.settings.job_page_size or DEFAULT_PAGE_SIZE
        page = store.page(
            run_id,
            offset=max(0, int(offset)),
            limit=int(limit) if limit else min(default_limit, MAX_PAGE_SIZE),
            cluster_id=cluster_id,
            assignment_status=assignment_status,
            keyword=keyword,
        )
        # 空页时 run_id 由 store 自己带出，这里统一补成作业的 run_id
        page["runId"] = run_id
        page["jobId"] = record.job_id
        return page

    def filter_options(self, job_id: str) -> dict:
        """明细分页的筛选取值清单（供前端渲染簇筛选控件）。"""

        record = self._require_record(job_id)
        store, run_id = self._open_store(record)
        options = store.filter_options(run_id)
        options["jobId"] = record.job_id
        return options

    def _require_record(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                record = self._load_locked(job_id)
            if record is None:
                raise JobNotFoundError("作业不存在或已被清理。")
            return record

    def _open_store(self, record: JobRecord):
        """打开作业明细仓库；状态不对或产物缺失时给出能解释原因的错误。"""

        from retrain_cluster.artifacts.semantic_runs import SemanticRunStore

        if record.status != JOB_SUCCEEDED:
            raise JobResultError(f"作业尚未成功完成（当前状态：{record.status}），暂无结果可读。")
        run_id = record.run_id
        if not run_id:
            raise JobResultError("作业缺少 run_id，无法定位结果产物。")
        store = SemanticRunStore(self._job_dir(record.job_id) / "result")
        if not store.exists(run_id):
            raise JobResultError("作业结果产物缺失或已被清理，请重新提交。")
        return store, run_id

    # ------------------------------------------------------------------ 取消

    def cancel(self, job_id: str) -> dict:
        """取消作业：排队中的直接出队；执行中的杀掉子进程。"""

        with self._lock:
            record = self._records.get(job_id) or self._load_locked(job_id)
            if record is None:
                raise JobNotFoundError("作业不存在或已被清理。")
            if record.terminal:
                raise JobNotCancellableError(f"作业已结束（{record.status}），无法取消。")
            record.cancel_requested = True
            record.updated_at = _now()
            if record.status == JOB_QUEUED:
                if job_id in self._queue:
                    self._queue.remove(job_id)
                record.touch(JOB_CANCELLED)
                record.finished_at = _now()
                self._persist_locked(record)
                logger.info("作业在排队阶段被取消：%s", job_id)
                return record.describe()
            self._persist_locked(record)

        # 执行中：把取消落到进程上。interrupt 返回 False 说明执行器不支持硬取消
        # （同进程执行），此时只记录"已请求"，由执行线程在返回时收敛成 cancelled。
        runner = self._runner
        hard = False
        if runner is not None:
            try:
                hard = bool(runner.interrupt())
            except Exception as exc:  # noqa: BLE001 - 取消失败也要给出确定状态
                logger.warning("中断作业执行器失败：%s", exc)
        with self._lock:
            record = self._records.get(job_id) or record
            record.hard_cancelled = hard
            self._persist_locked(record)
            logger.info("作业取消已下发：%s（硬取消=%s）", job_id, hard)
            return record.describe()

    # ------------------------------------------------------------------ 派发

    def _dispatch_loop(self) -> None:
        """单线程派发：按并行上限取作业执行。"""

        while not self._stop.is_set():
            with self._cv:
                while not self._stop.is_set() and (
                    not self._queue or self._active >= max(1, self.settings.job_max_concurrent)
                ):
                    self._cv.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                job_id = self._queue.pop(0)
                self._active += 1
            try:
                self._execute(job_id)
            except Exception as exc:  # noqa: BLE001 - 派发线程绝不能因单个作业而死
                logger.exception("作业派发异常：%s（%s）", job_id, exc)
            finally:
                with self._cv:
                    self._active -= 1
                    self._cv.notify_all()

    def _execute(self, job_id: str) -> None:
        """执行一个作业并落定终态。"""

        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status != JOB_QUEUED:
                return  # 已被取消/清理
            if record.cancel_requested:
                record.touch(JOB_CANCELLED)
                record.finished_at = _now()
                self._persist_locked(record)
                return
            record.touch(JOB_RUNNING)
            record.started_at = _now()
            self._persist_locked(record)

        try:
            payload = self._load_request(job_id)
        except Exception as exc:  # noqa: BLE001
            self._finish(job_id, JOB_FAILED, code="JOB_REQUEST_UNAVAILABLE", message=str(exc))
            return

        runner = self._ensure_runner()
        slot = self._service.run_slot() if self._service is not None else _null_slot()
        try:
            with slot:
                outcome = runner.submit(job_id, payload)
        except ClusterGatewayError as exc:
            self._settle_error(job_id, exc.code, exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - 引擎异常也要落成可读状态
            code = getattr(exc, "code", None) or "CLUSTERING_FAILED"
            message = getattr(exc, "message", None) or str(exc) or "Clustering execution failed"
            self._settle_error(job_id, code, message)
            return

        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            if record.cancel_requested:
                # 取消与完成几乎同时发生：以调用方意图为准，但结果产物保留，
                # 前端刷新时仍能看到"已取消"而不是一个说不清的失败
                record.touch(JOB_CANCELLED)
            else:
                record.run_id = (outcome.get("detail") or {}).get("runId")
                record.detail = dict(outcome.get("detail") or {})
                record.warnings = list(outcome.get("warnings") or [])
                record.status = JOB_SUCCEEDED
            record.finished_at = _now()
            self._persist_locked(record)
            # 清理与"进入终态"在同一把锁内完成：否则轮询方可能在看到 succeeded 的
            # 那一瞬间发现历史目录还没收拾，据此算出的"保留条数"与实际不符。
            self._prune_locked()

    def _settle_error(self, job_id: str, code: str, message: str) -> None:
        """把执行失败收敛成终态；被取消的作业即使报错也记成 cancelled。"""

        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            if record.cancel_requested:
                record.touch(JOB_CANCELLED)
            else:
                record.touch(JOB_FAILED)
                record.error_code = code
                record.error_message = message
            record.finished_at = _now()
            self._persist_locked(record)
            self._prune_locked()

    def _finish(self, job_id: str, status: str, *, code: str | None = None, message: str | None = None) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.touch(status)
            record.finished_at = _now()
            if code:
                record.error_code = code
                record.error_message = message
            self._persist_locked(record)
            self._prune_locked()

    def _ensure_runner(self):
        """惰性创建执行器；默认走子进程，因此取消是硬取消。"""

        if self._runner is None:
            self._runner = SubprocessJobRunner(self.settings)
        return self._runner

    def _load_request(self, job_id: str) -> dict:
        path = self._job_dir(job_id) / "request.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise JobResultError("作业请求快照已丢失，无法执行。") from None
        except (ValueError, OSError):
            raise JobResultError("作业请求快照损坏，无法执行。") from None
        return {"items": payload.get("items") or [], "options": payload.get("options") or {}}

    # ------------------------------------------------------------------ 持久化与恢复

    def _persist_locked(self, record: JobRecord) -> None:
        """落盘作业记录。调用方必须已持有 `self._lock`。"""

        _atomic_json(self._record_path(record.job_id), record.as_dict())

    def _load_locked(self, job_id: str) -> JobRecord | None:
        """从磁盘读一条记录；内存里已有就以内存为准（内存即最新）。"""

        if job_id in self._records:
            return self._records[job_id]
        try:
            path = self._record_path(job_id)
        except JobNotFoundError:
            return None
        try:
            record = JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError, OSError, TypeError):
            return None
        self._records[record.job_id] = record
        return record

    def _load_all_locked(self) -> None:
        if not self.root.is_dir():
            return
        for path in (self.root).glob("job_*/job.json"):
            self._load_locked(path.parent.name)

    def _recover_locked(self) -> None:
        """重启恢复：排队中的作业重新入队，执行中的作业标成失败。

        为什么要区分：排队中的作业**从未开始**，重跑不会产生半成品，符合调用方意图；
        执行中的作业的子进程已随网关一起死掉，结果不可信——把它标成
        `GATEWAY_RESTARTED` 比让它永远停在 running 更诚实。

        若结果产物其实已经完整落盘（子进程写完了、父进程还没记账），
        直接认定成功：manifest 是提交标记，它存在就说明数据是完整的。
        """

        self._load_all_locked()
        requeued = 0
        for record in self._records.values():
            if record.status == JOB_QUEUED:
                if record.job_id not in self._queue:
                    self._queue.append(record.job_id)
                    requeued += 1
                continue
            if record.status != JOB_RUNNING:
                continue
            if self._result_is_complete(record):
                record.status = JOB_SUCCEEDED
                record.hard_cancelled = False
                record.finished_at = record.finished_at or _now()
            else:
                record.status = JOB_FAILED
                record.error_code = "GATEWAY_RESTARTED"
                record.error_message = "网关在执行过程中重启，作业结果不可信，请重新提交。"
                record.finished_at = _now()
            self._persist_locked(record)
        if requeued:
            self._cv.notify_all()
            logger.info("重启恢复：%d 个排队中的作业已重新入队", requeued)

    def _result_is_complete(self, record: JobRecord) -> bool:
        if not record.run_id:
            return False
        try:
            from retrain_cluster.artifacts.semantic_runs import SemanticRunStore

            return SemanticRunStore(self._job_dir(record.job_id) / "result").exists(record.run_id)
        except Exception:  # noqa: BLE001 - 恢复判定失败按"不完整"处理，宁可重跑
            return False

    def _prune_locked(self) -> None:
        """清理超出保留数量的**终态**作业目录；活跃作业永不清理。"""

        records = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
        keep = self.settings.job_history_limit
        stale = [record for record in records if record.terminal][keep:]
        for record in stale:
            import shutil

            shutil.rmtree(self._job_dir(record.job_id), ignore_errors=True)
            self._records.pop(record.job_id, None)


class _null_slot:
    """没有网关服务时的空槽位（测试注入场景）。"""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


__all__ = [
    "DatasetStore",
    "InlineJobRunner",
    "JobRecord",
    "SemanticJobService",
    "SubprocessJobRunner",
    "inline_handler",
    "job_worker_main",
    "persist_job_result",
    "run_job_payload",
]
