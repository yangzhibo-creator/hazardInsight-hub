"""聚类网关服务：把 cluster-engine 的算法能力包装成 HazardInsight 的接口契约。

分工严格遵循"不重复实现算法"的原则：

* **算法、向量化、检索增强、特征融合、降维、参数校验** —— 全部调用
  `retrain_cluster` 的既有实现，网关一行算法都没有重写；
* **网关只负责**：加载引擎、解析 profile 元信息、把上传文件规范成样本、
  调用引擎执行、把引擎结果加工成前端需要的结构（统计 / 簇摘要 / 二维坐标）。

引擎结果文件的落盘（`artifacts/runs/<run_id>/`）由引擎自己完成，
网关不干预，从而保留完整的可复现追溯链。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import GatewaySettings, get_settings
from app.core.constants import (
    AUTO_PROFILE_ALGORITHM_PREFERENCE,
    MAX_ITEMS,
    MAX_TEXT_CHARS,
    MIN_ITEMS,
    MIN_SEMANTIC_ITEMS,
    NOISE_CLUSTER_ID,
)
from app.core.errors import (
    ClusterGatewayError,
    ClusteringError,
    EngineUnavailableError,
    InvalidInputError,
    ProfileUnavailableError,
    SampleDatasetError,
    ServiceBusyError,
)
from app.core.logger import get_logger
from app.processors.tabular_reader import read_records
from app.processors.text_cleaner import clean_hazard_text
from app.services.cluster_digest import build_digest
from app.services.cluster_evaluation import compare_metrics, evaluate_metrics, extract_ground_truth
from app.utils.helpers import truncate
from app.utils.validators import validate_item_count

logger = get_logger(__name__)

#: 本地模型校验（逐个文件算 SHA-256）代价较高，因此结果在进程内缓存。
_MODEL_READY_TTL_SECONDS = 600.0

#: profile 可用性缓存有效期。
_PROFILE_STATUS_TTL_SECONDS = 600.0


def _from_engine_error(exc: Exception) -> ClusterGatewayError:
    """把 cluster-engine 的 `ClusterError` 转成网关错误，保留其错误码与状态码。"""

    error = ClusterGatewayError(str(getattr(exc, "message", exc)))
    error.code = getattr(exc, "code", "CLUSTERING_FAILED")
    error.status = getattr(exc, "status", 500)
    return error


#: 实现版本 -> 能力描述。取自引擎的 Strategy 注册表，网关不自己发明能力定义。
_CAPABILITY_CACHE: dict[str, dict[str, Any]] = {}


def _capability_of(implementation_version: str) -> dict[str, Any]:
    """返回某实现版本的能力（能否单条、能否 cache-only）。未登记则按最保守处理。"""

    if implementation_version not in _CAPABILITY_CACHE:
        try:
            from retrain_cluster.services.strategies import list_strategy_versions

            for entry in list_strategy_versions():
                _CAPABILITY_CACHE[entry["implementation_version"]] = {
                    "implementation_version": entry["implementation_version"],
                    "supports_single_item": bool(entry.get("supports_single_item")),
                    "supports_cache_only": bool(entry.get("supports_cache_only")),
                    # 校准版本是 profile 级别的（同一版本可挂不同校准），
                    # 因此这里恒为 None，真实值在 ProfileInfo.calibration_id 上。
                    "calibration_version": entry.get("calibration_version"),
                    # spear-v1 的输入层净化是版本级能力，前端据此显示"净化"开关
                    "purification": bool(entry.get("purification")),
                }
        except Exception as exc:  # noqa: BLE001 - 引擎依赖缺失不应让元信息接口变成 500
            logger.debug("能力表加载失败，退回保守默认值：%s", exc)
        _CAPABILITY_CACHE.setdefault(
            implementation_version,
            {
                "implementation_version": implementation_version,
                "supports_single_item": False,
                "supports_cache_only": False,
                "calibration_version": None,
                "purification": False,
            },
        )
    return _CAPABILITY_CACHE[implementation_version]


class ClusteringGatewayService:
    """单例式的聚类网关服务。

    引擎的加载是惰性且带锁的：配置缺失或依赖未安装时，健康检查会明确报告
    `degraded`，而不是在首个请求时抛出难以理解的导入错误。
    """

    def __init__(self, settings: GatewaySettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine: Any = None
        self._engine_settings: Any = None
        self._catalog: Any = None
        self._engine_error: str | None = None
        self._load_lock = threading.Lock()

        # 单飞：聚类是 CPU/内存密集型任务，同一时刻只允许一个请求进入引擎。
        self._run_lock = threading.Lock()

        # 元信息缓存，避免每次列表请求都重新校验 1.3GB 本地模型
        # 注意：这里的名字不能叫 `_model_ready`，否则会遮蔽同名的 `_model_ready()` 方法
        # （实例属性优先于类方法查找，调用时会变成 "dict object is not callable"）。
        self._model_ready_cache: dict[str, tuple[bool, str | None, float]] = {}
        self._profile_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ 引擎加载

    def preload(self) -> None:
        """在后台线程预热引擎与模型校验，让首个请求不必等待磁盘校验。

        本地模型校验需要逐个文件计算 SHA-256（约 1.3GB），放在启动时做一次，
        后续 10 分钟内的健康检查与 profile 列表都可直接命中缓存。
        """

        def warm() -> None:
            try:
                self._load_engine()
                if self._engine is None:
                    return
                for model_id in self._catalog.models:
                    self._model_ready(model_id)
                self.list_profiles()
                logger.info("聚类引擎预热完成")
            except Exception as exc:  # noqa: BLE001 - 预热失败不应影响服务启动
                logger.warning("聚类引擎预热失败：%s", exc)

        threading.Thread(target=warm, name="cluster-preload", daemon=True).start()

    def invalidate_caches(self) -> None:
        """清空元信息缓存，强制下次请求重新校验模型与 profile。"""

        with self._cache_lock:
            self._model_ready_cache.clear()
            self._profile_cache = None

    def _load_engine(self) -> None:
        """惰性加载 cluster-engine（配置、Catalog、ClusteringService）。"""

        if self._engine is not None or self._engine_error is not None:
            return
        with self._load_lock:
            if self._engine is not None or self._engine_error is not None:
                return
            config_path = self.settings.engine_config_path
            if not config_path.is_file():
                self._engine_error = f"未找到聚类引擎配置：{config_path}"
                logger.error("%s", self._engine_error)
                return
            try:
                from retrain_cluster.config import Catalog, Settings as EngineSettings
                from retrain_cluster.services.clustering import ClusteringService

                engine_settings = EngineSettings.load(str(config_path))
                self._catalog = Catalog(engine_settings)
                self._engine_settings = engine_settings
                self._engine = ClusteringService(engine_settings)
                logger.info(
                    "聚类引擎已加载：%d 个 profile，%d 个模型",
                    len(self._catalog.profiles),
                    len(self._catalog.models),
                )
            except Exception as exc:  # noqa: BLE001 - 依赖缺失/配置错误都要转成明确状态
                self._engine_error = (
                    f"聚类引擎加载失败：{exc}。请在 backend/python 下执行 "
                    f"pip install -r requirements.txt 后重试。"
                )
                logger.error("%s", self._engine_error)

    def _require_engine(self) -> Any:
        """取回引擎实例；不可用时抛 `EngineUnavailableError`。"""

        self._load_engine()
        if self._engine is None:
            raise EngineUnavailableError(self._engine_error or "聚类引擎不可用。")
        return self._engine

    def require_engine(self) -> Any:
        """公开的引擎获取入口。

        「偏差数据库」服务需要复用同一份引擎实例（同一份编码器缓存、同一份
        Qwen 净化器），否则会各加载一次 1.3GB 向量模型，甚至在 GPU 上并存两份
        Qwen。这里把原来的私有方法显式开放出来，避免跨模块访问下划线成员。
        """

        return self._require_engine()

    def run_slot(self):
        """占用"同一时刻只跑一个聚类"的执行槽位，返回上下文管理器。

        作业路径与同步 `/run` 共用同一个槽位（而不是各拿一把锁）：两者的资源瓶颈
        是同一份模型内存，各锁各的会让"并发上限"变成一句空话——同步请求照跑，
        作业也照跑，内存该爆还是爆。抢不到就抛 `SERVICE_BUSY`（429）。
        """

        @contextmanager
        def _slot():
            if not self._run_lock.acquire(blocking=False):
                raise ServiceBusyError("已有聚类任务正在执行，请稍后重试。")
            try:
                yield
            finally:
                self._run_lock.release()

        return _slot()

    # ------------------------------------------------------------------ 元信息

    def _model_ready(self, model_id: str) -> tuple[bool, str | None]:
        """校验模型可用性（带进程内缓存）。返回 (是否可用, 不可用原因)。"""

        now = time.monotonic()
        with self._cache_lock:
            cached = self._model_ready_cache.get(model_id)
            if cached and now - cached[2] < _MODEL_READY_TTL_SECONDS:
                return cached[0], cached[1]
        ready, reason = True, None
        try:
            from retrain_cluster.embeddings.registry import check_model

            check_model(self._catalog.model(model_id))
        except Exception as exc:  # noqa: BLE001 - 依赖缺失/校验和不符都在此收敛
            ready, reason = False, getattr(exc, "code", "MODEL_UNAVAILABLE")
            logger.warning("模型 %s 不可用：%s", model_id, exc)
        with self._cache_lock:
            self._model_ready_cache[model_id] = (ready, reason, now)
        return ready, reason

    def _profile_status(self, profile: Any) -> tuple[bool, str | None]:
        """判断某个 profile 当前能否执行。

        复用引擎的 `validate_params` / `check_model` / `read_manifest` 三个校验点，
        但把最贵的模型校验按 `model_id` 缓存，避免 20 个 profile 重复校验同一份
        1.3GB 权重（引擎自带的 `/api/v1/profiles` 每次都全量校验，较慢）。
        """

        try:
            from retrain_cluster.clustering.registry import validate_params
            from retrain_cluster.config import model_fingerprint
            from retrain_cluster.retrieval.chroma import read_manifest

            validate_params(profile.algorithm, profile.algorithm_params, profile.backend)
            model = self._catalog.model(profile.model_id)
            ready, reason = self._model_ready(profile.model_id)
            if not ready:
                return False, reason
            if profile.features.n_results:
                read_manifest(
                    self._catalog.settings.artifacts_dir / "knowledge_bases",
                    profile.knowledge_base_id,
                    model_fingerprint(model),
                    model["dimension"],
                )
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, getattr(exc, "code", "PROFILE_UNAVAILABLE")

    @staticmethod
    def _purification_status(profile: Any) -> dict[str, Any] | None:
        """净化的可用性快照；legacy / semantic profile 返回 None。

        只做无副作用的探测（不加载 55 GB 权重），并把 ``model_path`` 从对外
        载荷里剔除——profile 列表是前端可见的，不应暴露宿主机路径。
        """

        spec = getattr(profile, "purification", None)
        if spec is None:
            return None
        try:
            from retrain_cluster.purification import probe

            payload = probe(spec).as_dict()
            payload.pop("model_path", None)
            return payload
        except Exception as exc:  # noqa: BLE001 - 探测失败不能拖垮元信息接口
            logger.warning("净化状态探测失败（%s），按降级处理。", exc)
            return {
                "enabled": bool(getattr(spec, "enabled", False)),
                "requested_backend": getattr(spec, "backend", "unknown"),
                "effective_backend": "rule",
                "degraded": True,
                "reason": "purification_probe_failed",
                "guarded": bool(getattr(spec, "guard", True)),
            }

    def list_profiles(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """列出引擎中可通过 API 使用的 profile 及其可用状态。"""

        self._require_engine()
        now = time.monotonic()
        with self._cache_lock:
            if not refresh and self._profile_cache and now - self._profile_cache[0] < _PROFILE_STATUS_TTL_SECONDS:
                return self._profile_cache[1]

        from retrain_cluster.clustering.registry import API_ALGORITHMS, WARNINGS

        profiles: list[dict[str, Any]] = []
        for profile in self._catalog.profiles.values():
            if profile.algorithm not in API_ALGORITHMS:
                continue
            available, reason = self._profile_status(profile)
            warnings = list(WARNINGS.get(profile.algorithm, []))
            purification = self._purification_status(profile)
            if purification and purification.get("degraded"):
                # 净化是软依赖：降级不影响 profile 可用，但必须让调用方看见
                warnings.append("PURIFIER_DEGRADED_TO_RULE")
            profiles.append(
                {
                    "profile_id": profile.profile_id,
                    "algorithm": profile.algorithm,
                    "model_id": profile.model_id,
                    "knowledge_base_id": profile.knowledge_base_id,
                    "features": asdict(profile.features),
                    "algorithm_params": profile.algorithm_params,
                    "implementation_version": profile.implementation_version,
                    "max_samples": min(MAX_ITEMS, profile.max_samples),
                    "available": available,
                    "unavailable_reason": reason,
                    "warnings": warnings,
                    # semantic profile 的阈值来自校准文件，profile 只引用它的 ID；
                    # 把 ID 透出，前端可以据此说明"这批阈值尚未经人工校准验证"。
                    "calibration_id": getattr(profile, "calibration_id", None),
                    "capability": _capability_of(profile.implementation_version),
                    "purification": purification,
                }
            )
        profiles.sort(
            key=lambda item: (
                not item["available"],
                item["features"].get("n_results", 0),
                item["algorithm"],
            )
        )
        with self._cache_lock:
            self._profile_cache = (now, profiles)
        return profiles

    def list_algorithms(self) -> list[dict[str, Any]]:
        """列出引擎暴露给 API 的算法及其可用性。

        实现版本按算法逐个判定，而不是一律写 `legacy-v1`：`semantic_auto_kmeans`
        走的是 `semantic-v1`，两者的准入规则与能力并不相同，混报会让前端
        无法判断"这个算法能不能只传 1 条"。
        """

        self._require_engine()
        from retrain_cluster.clustering.registry import API_ALGORITHMS, WARNINGS, available
        from retrain_cluster.services.strategies import LEGACY_VERSION, SEMANTIC_ALGORITHMS, SEMANTIC_VERSION

        return [
            {
                "algorithm": name,
                "available": bool(available(name)),
                "backend": "python",
                "implementation_version": SEMANTIC_VERSION if name in SEMANTIC_ALGORITHMS else LEGACY_VERSION,
                "max_samples": MAX_ITEMS,
                "warnings": list(WARNINGS.get(name, [])),
                "capability": _capability_of(
                    SEMANTIC_VERSION if name in SEMANTIC_ALGORITHMS else LEGACY_VERSION
                ),
            }
            for name in API_ALGORITHMS
        ]

    def engine_status(self) -> dict[str, Any]:
        """返回引擎加载状态，供健康检查使用。"""

        self._load_engine()
        config_file = self.settings.engine_config_path.name
        if self._engine is None:
            return {
                "config_file": config_file,
                "loaded": False,
                "message": self._engine_error,
            }

        from retrain_cluster.clustering.registry import API_ALGORITHMS, available

        model_ids = list(self._catalog.models.keys())
        model_id = model_ids[0] if model_ids else None
        spec = self._catalog.models.get(model_id) if model_id else None
        ready, reason = self._model_ready(model_id) if model_id else (False, "MODEL_NOT_CONFIGURED")
        profiles = self.list_profiles()
        return {
            "config_file": config_file,
            "loaded": True,
            "profiles_total": len(profiles),
            "profiles_available": sum(1 for profile in profiles if profile["available"]),
            "algorithms_available": [name for name in API_ALGORITHMS if available(name)],
            "model_id": model_id,
            "model_provider": spec.get("provider") if spec else None,
            "model_dimension": spec.get("dimension") if spec else None,
            "message": None if ready else f"向量模型不可用（{reason}）",
        }

    # ------------------------------------------------------------------ 示例与上传

    def load_sample_dataset(self) -> dict[str, Any]:
        """读取内置示例数据集（真实核电工程隐患抽样）。"""

        path: Path = self.settings.sample_dataset_path
        if not path.is_file():
            raise SampleDatasetError(f"内置示例数据不存在：{path.name}。")
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SampleDatasetError(f"内置示例数据解析失败：{exc}") from exc
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not items:
            raise SampleDatasetError("内置示例数据中没有可用样本。")
        dataset = payload.get("dataset_id", path.stem) if isinstance(payload, dict) else path.stem
        return {"source_name": f"示例数据 · {dataset}", "items": items}

    def load_offline_baseline(self) -> dict[str, Any]:
        """读取离线基准结果（现场兜底展示）。

        数字来自归档文件而不是现场计算，因此调用方必须能看到 `source` 与
        `full_run`：抽样口径的指标不具可比性，这一点不能让界面替它隐瞒。
        相对增益在网关侧统一计算，避免前端各自实现一套公式。
        """

        path = Path(__file__).resolve().parents[1] / "data" / "offline_baseline.json"
        if not path.is_file():
            raise ClusteringError("离线基准结果文件不存在，无法离线展示。")
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ClusteringError(f"离线基准结果解析失败：{exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), dict):
            raise ClusteringError("离线基准结果结构不合法。")

        rows = payload["rows"]
        base_key = payload.get("baseline_key") or next(iter(rows), None)
        target_key = payload.get("target_key") or next((key for key in rows if key != base_key), None)
        comparison: dict[str, float] = {}
        base, target = rows.get(base_key) or {}, rows.get(target_key) or {}
        # 自由字典的键不会被 pydantic 的别名生成器改写，因此这里直接产出 camelCase，
        # 与前端 TypeScript 类型（shared/clustering.ts）逐字对齐。
        for metric, output_key in (
            ("ari", "ari"),
            ("vm", "vm"),
            ("fms", "fms"),
            ("ami", "ami"),
            ("hs", "hs"),
            ("cs", "cs"),
            ("score", "score"),
            ("n_clusters", "nClusters"),
            ("noise_ratio", "noiseRatio"),
        ):
            left, right = base.get(metric), target.get(metric)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                comparison[output_key] = round(float(right) - float(left), 4)
        if base.get("ari"):
            comparison["ariRelative"] = round(
                (float(target["ari"]) - float(base["ari"])) / float(base["ari"]), 4
            )
        payload["comparison"] = comparison
        return payload

    def parse_dataset(self, data: bytes, filename: str) -> dict[str, Any]:
        """把上传的数据文件规范成样本列表。

        返回 `{source_name, total, items, warnings}`，字段与 `DatasetPreview` 契约一一对应
        （`total` 是清洗后真正可用的样本数，不等于原始行数，因此必须显式回报）。
        """

        records, warnings = read_records(data, filename)
        if not records:
            raise InvalidInputError("文件中没有解析出可用样本，请检查文本列是否为空。")

        items: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        truncated = 0
        skipped: list[str] = []

        for index, record in enumerate(records, start=1):
            raw_id = str(record.get("id") or f"row-{index}").strip()
            identifier, id_cut = truncate(raw_id, 128)
            identifier = identifier or f"row-{index}"
            if identifier in seen_ids:
                seen_ids[identifier] += 1
                identifier = f"{identifier}#{seen_ids[identifier]}"
            else:
                seen_ids[identifier] = 1
            text = str(record.get("text") or "")
            cleaned = clean_hazard_text(text, limit=MAX_TEXT_CHARS)
            if not cleaned.strip():
                skipped.append(raw_id)
                continue
            if len(cleaned) < len(text.strip()):
                truncated += 1
            metadata = {
                key: str(value)
                for key, value in record.items()
                if key not in {"id", "text"} and str(value).strip()
            }
            items.append({"id": identifier, "text": cleaned, "metadata": metadata})
            if len(items) >= MAX_ITEMS:
                warnings.append(f"样本数已达上限 {MAX_ITEMS} 条，其余数据被忽略。")
                break

        if id_cut:
            warnings.append("存在超过 128 字符的 ID，已截断。")
        if truncated:
            warnings.append(f"{truncated} 条文本超过 {MAX_TEXT_CHARS} 字符，已截断后参与聚类。")
        if skipped:
            warnings.append(f"{len(skipped)} 条记录文本为空，已跳过。")
        if not items:
            raise InvalidInputError("文件中没有可用的非空文本。")
        return {
            "source_name": filename,
            "total": len(items),
            "items": items,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ 执行聚类

    def _apply_knowledge_base(self, profile: Any, knowledge_base_id: str | None) -> tuple[Any, str | None]:
        """把请求级知识库覆盖应用到 profile（「偏差数据库」选择器）。

        复用引擎的 `with_knowledge_base`，保证"网关本地复现特征用的 profile"与
        "引擎执行用的 profile"是同一份——否则 legacy 路径会标签按新库算、
        散点按旧库画。

        返回 `(profile, 提示)`：当知识库条目数少于 profile 的近邻数 ``k`` 时，
        引擎会把 ``k`` 下调为条目数；这属于"确实改变了本次检索口径"，必须作为
        告警回传给界面，而不是悄悄发生。
        """

        if not knowledge_base_id:
            return profile, None
        try:
            overridden = self._require_engine().with_knowledge_base(profile, knowledge_base_id)
        except Exception as exc:  # noqa: BLE001 - 非法覆盖（如非检索 profile）转成网关错误
            raise _from_engine_error(exc) from exc
        note = None
        if overridden.features.n_results < profile.features.n_results:
            note = (
                f"知识库「{knowledge_base_id}」只有 {overridden.features.n_results} 条，"
                f"少于 profile 声明的近邻数 k={profile.features.n_results}；"
                f"本次已自动将 k 下调为 {overridden.features.n_results}。"
            )
        return overridden, note

    @staticmethod
    def _knowledge_base_request_field(profile: Any) -> dict[str, Any]:
        """引擎请求里显式带上 profile 生效的知识库。

        引擎按 `profile_id` 重新解析配置，若不带上这个字段，它会用清单里的默认
        知识库，忽略请求级覆盖——于是"网关本地 profile"与"引擎实际执行的 profile"
        分叉。只在检索增强 profile 上带，纯向量 profile 带了会被引擎拒绝。
        """

        if profile.features.n_results and profile.knowledge_base_id:
            return {"knowledge_base_id": profile.knowledge_base_id}
        return {}

    def _resolve_profile(self, options: dict[str, Any]) -> tuple[Any, list[str]]:
        """决定本次请求使用哪个 profile，并回传因知识库覆盖产生的告警。

        优先级：显式 `profile_id` > 指定 `algorithm` 的可用 profile > 配置的
        `default_profile_id` > 自动挑选（纯向量 profile 优先）。
        """

        warnings: list[str] = []
        catalog = self._catalog
        knowledge_base_id = options.get("knowledge_base_id")
        profile_id = options.get("profile_id") or self.settings.default_profile_id
        if profile_id:
            try:
                profile = catalog.profile(profile_id)
            except Exception as exc:  # noqa: BLE001
                raise _from_engine_error(exc) from exc
            # 先应用知识库覆盖，再按覆盖后的知识库校验可用性：
            # 否则"默认库缺失、但用户另选了一个可用库"会被错误地判为不可执行。
            profile, note = self._apply_knowledge_base(profile, knowledge_base_id)
            if note:
                warnings.append(note)
            available, reason = self._profile_status(profile)
            if not available:
                raise ProfileUnavailableError(
                    f"profile「{profile_id}」当前不可执行（{reason}）。"
                    "若是检索增强 profile，请先构建对应知识库。"
                )
            return profile, warnings

        profiles = self.list_profiles()
        algorithms = [options["algorithm"]] if options.get("algorithm") else []

        def matches(profile: dict[str, Any]) -> bool:
            if algorithms and profile["algorithm"] not in algorithms:
                return False
            # 指定了知识库时只考虑检索增强 profile：把知识库传给纯向量 profile
            # 会被引擎拒绝，自动挑选必须提前避开，而不是挑完再报 422。
            if knowledge_base_id and not int(profile["features"].get("n_results", 0) or 0):
                return False
            return True

        candidates = [profile for profile in profiles if profile["available"] and matches(profile)]
        if not candidates and algorithms:
            raise ProfileUnavailableError(
                f"算法「{algorithms[0]}」当前没有可用的 profile。"
                f"可用算法：{', '.join(sorted({p['algorithm'] for p in profiles if p['available']}))}"
            )
        if not candidates and knowledge_base_id:
            raise ProfileUnavailableError(
                "没有可用的检索增强 profile 来使用所选知识库；请显式选择一个检索增强 profile。"
            )
        if not candidates:
            raise ProfileUnavailableError("当前没有可用的 profile，请检查引擎配置与模型文件。")

        def rank(profile: dict[str, Any]) -> tuple[int, int, str]:
            try:
                preferred = AUTO_PROFILE_ALGORITHM_PREFERENCE.index(profile["algorithm"])
            except ValueError:
                preferred = len(AUTO_PROFILE_ALGORITHM_PREFERENCE)
            return (profile["features"].get("n_results", 0), preferred, profile["profile_id"])

        selected = sorted(candidates, key=rank)[0]
        profile, note = self._apply_knowledge_base(catalog.profile(selected["profile_id"]), knowledge_base_id)
        if note:
            warnings.append(note)
        return profile, warnings

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行一次聚类，返回加工后的结果。

        参数:
            payload: 形如 `{"items": [{"id","text","metadata"}], "options": {...}}`。

        分派顺序很关键（对应实施计划 10.1 的「增量接入」）：先解析 profile，
        再按 `implementation_version` 决定走哪条路径，最后才做**版本相关**的
        准入校验。因此 legacy 的行为与改动前逐字一致，而 semantic-v1 可以在
        自己那条分支上放开「至少 2 条」这类 legacy 约束，而不是把 legacy
        也一起放宽。
        """

        started = time.monotonic()
        options = dict(payload.get("options") or {})
        raw_items = payload.get("items") or []

        # 与版本无关的早退（空请求 / 超出服务上限）：放在加载引擎之前，
        # 让明显非法的请求不必先付出「校验 1.3GB 本地模型」的代价。
        validate_item_count(len(raw_items), minimum=MIN_SEMANTIC_ITEMS)

        sample_ids: list[str] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items, start=1):
            texts.append("" if item.get("text") is None else str(item.get("text")))
            sample_ids.append(str(item.get("id") or f"row-{index}"))
            metadata.append(dict(item.get("metadata") or {}))
        if len(set(sample_ids)) != len(sample_ids):
            raise InvalidInputError("样本 ID 必须唯一，请检查输入数据。")

        from retrain_cluster.clustering.registry import validate_params
        from retrain_cluster.services.strategies import LEGACY_VERSION, SEMANTIC_VERSION, SPEAR_VERSION

        engine = self._require_engine()
        profile, kb_warnings = self._resolve_profile(options)
        implementation = getattr(profile, "implementation_version", LEGACY_VERSION)

        # 引擎在真正执行前先校验超参：参数不合法立刻返回 422 语义，不做无谓计算。
        try:
            validate_params(profile.algorithm, profile.algorithm_params, profile.backend)
        except Exception as exc:  # noqa: BLE001
            raise _from_engine_error(exc) from exc

        if implementation == SEMANTIC_VERSION:
            result = self._run_semantic(
                engine=engine,
                profile=profile,
                sample_ids=sample_ids,
                texts=texts,
                metadata=metadata,
                started=started,
            )
        elif implementation == SPEAR_VERSION:
            # SPEAR 的净化发生在引擎内部：网关只负责把"开/关净化"这一意图
            # 原样透传，并把引擎返回的净化报告挂到契约上，不重新实现净化。
            validate_item_count(len(raw_items), minimum=MIN_ITEMS)
            result = self._run_spear(
                engine=engine,
                profile=profile,
                sample_ids=sample_ids,
                texts=texts,
                metadata=metadata,
                options=options,
                started=started,
            )
        else:
            # legacy-v1：仍然是「至少 2 条」；语义路径的放宽不外溢到这里。
            validate_item_count(len(raw_items), minimum=MIN_ITEMS)
            result = self._run_legacy(
                engine=engine,
                profile=profile,
                sample_ids=sample_ids,
                texts=texts,
                metadata=metadata,
                options=options,
                started=started,
            )

        # 知识库覆盖带来的口径变化（如 k 被下调）必须出现在结果告警里，不能只在日志
        if kb_warnings:
            result.setdefault("summary", {}).setdefault("warnings", []).extend(kb_warnings)

        self._attach_evaluation(
            result=result,
            engine=engine,
            profile=profile,
            sample_ids=sample_ids,
            texts=texts,
            metadata=metadata,
            options=options,
            implementation=implementation,
        )
        return result

    # ------------------------------------------------------------------ 指标评估

    def _attach_evaluation(
        self,
        *,
        result: dict[str, Any],
        engine: Any,
        profile: Any,
        sample_ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]],
        options: dict[str, Any],
        implementation: str,
    ) -> None:
        """把 6 项外部指标（以及可选的「未净化对照」）挂到 summary 上。

        真值来自数据集自带的标签列（如 ``category``）；**每条样本都带标签才算**，
        否则只给告警、不猜——用不完整标注算出来的 ARI 比没有 ARI 更危险。

        ``controlPurify=true`` 时额外跑一遍「净化关」作为对照，并与主运行做差值，
        这正是论文里 nr0 与完整框架（nr-1）的对比口径。对照只在 spear-v1 上成立：
        其它 profile 本来就没有净化步骤。
        """

        from retrain_cluster.services.strategies import SPEAR_VERSION

        summary = result.setdefault("summary", {})
        warnings = summary.setdefault("warnings", [])
        truth, field = extract_ground_truth(metadata)
        wants_control = bool(options.get("control_purify"))

        if truth is None:
            if wants_control:
                # 勾了对照却没有真值：如实说明，而不是给出一个静默的空结果
                warnings.append("CONTROL_METRICS_NO_GROUND_TRUTH")
            return

        predicted = [int(item["cluster_id"]) for item in result.get("items") or []]
        try:
            metrics = evaluate_metrics(truth, predicted)
        except Exception as exc:  # noqa: BLE001 - 指标是附加信息，失败不应让聚类整体失败
            logger.warning("外部指标计算失败：%s", exc)
            warnings.append("METRICS_COMPUTATION_FAILED")
            return
        summary["metrics"] = metrics
        summary["ground_truth"] = {"field": field, "labeled": len(truth), "total": len(metadata)}
        if metrics.get("score") == -1.0:
            warnings.append("METRICS_UNAVAILABLE_SINGLE_CLUSTER")

        if not wants_control:
            return
        if implementation != SPEAR_VERSION:
            # 没有净化能力的 profile 不存在"未净化对照"；不能假装跑过
            warnings.append("CONTROL_RUN_SKIPPED_UNSUPPORTED_PROFILE")
            return

        control = self._run_spear(
            engine=engine,
            profile=profile,
            sample_ids=sample_ids,
            texts=texts,
            metadata=metadata,
            # 对照固定为"净化关"：与请求级 purify 覆盖走同一条语义，不新增口径
            options={**options, "purify": False},
            started=time.monotonic(),
        )
        control_predicted = [int(item["cluster_id"]) for item in control.get("items") or []]
        try:
            control_metrics = evaluate_metrics(truth, control_predicted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("对照运行指标计算失败：%s", exc)
            warnings.append("CONTROL_METRICS_COMPUTATION_FAILED")
            return
        summary["control_metrics"] = control_metrics
        # 方向与离线基准一致：base=净化关（nr0 口径），target=本次主运行（净化开）
        summary["metrics_comparison"] = compare_metrics(control_metrics, metrics)

    # ------------------------------------------------------------------ legacy 分支

    def _run_legacy(
        self,
        *,
        engine: Any,
        profile: Any,
        sample_ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]],
        options: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        """legacy-v1：引擎算标签，网关复现特征变换并加工展示结构。

        流程：引擎执行聚类（算法权威结果）→ 取回/复用向量 → 复现引擎的特征变换
        → PCA 降维得到二维坐标 → 生成簇摘要。这一路径的行为与新增 semantic-v1
        之前完全相同，一行都没有改动。
        """

        from retrain_cluster.features.pipeline import FeaturePipeline
        from retrain_cluster.features.reduction import DimReducer
        from retrain_cluster.types import EmbeddingBatch

        cleaned: list[str] = []
        for index, text in enumerate(texts, start=1):
            value = clean_hazard_text(text, limit=MAX_TEXT_CHARS)
            if not value.strip():
                raise InvalidInputError(f"第 {index} 条样本文本为空。")
            cleaned.append(value)
        texts = cleaned

        if not self._run_lock.acquire(blocking=False):
            raise ServiceBusyError("已有聚类任务正在执行，请稍后重试。")
        try:
            # ① 引擎执行聚类：算法结果与产物落盘都由引擎负责
            engine_request = {
                "profile_id": profile.profile_id,
                "items": [{"id": ident, "text": text} for ident, text in zip(sample_ids, texts)],
            }
            engine_request.update(self._knowledge_base_request_field(profile))
            try:
                result = engine.cluster(engine_request, enforce_api_limits=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("聚类执行失败：%s", exc)
                raise _from_engine_error(exc) from exc

            labels = np.asarray([item["cluster_id"] for item in result["assignments"]], dtype=int)
            if labels.shape != (len(sample_ids),):
                raise ClusteringError("引擎返回的标签数量与样本数不一致。")

            # ② 取回向量并复现引擎的特征变换（命中引擎自有缓存时不会重复推理）
            try:
                values, model_hash, cache_hit = engine.embeddings(texts, profile.model_id)
                retriever = engine.retriever(profile, model_hash)
                transformed = FeaturePipeline(retriever, self._engine_settings.query_batch).transform(
                    EmbeddingBatch(sample_ids, values, model_hash).values, profile.features
                )
            except Exception as exc:  # noqa: BLE001
                raise _from_engine_error(exc) from exc

            # ③ 二维坐标仅用于可视化，不参与任何聚类计算
            visualization: list[dict[str, Any]] = []
            warnings = list(result.get("warnings", []))
            if options.get("visualize", True) and options.get("reduce_method", "pca") == "pca":
                if transformed.shape[1] >= 2 and transformed.shape[0] >= 2:
                    try:
                        coords = DimReducer("PCA").fit_transform(transformed, 2)
                        visualization = [
                            {
                                "id": sample_ids[index],
                                "x": round(float(coords[index][0]), 6),
                                "y": round(float(coords[index][1]), 6),
                                "cluster_id": int(labels[index]),
                            }
                            for index in range(len(sample_ids))
                        ]
                    except Exception as exc:  # noqa: BLE001 - 可视化失败不应让聚类失败
                        logger.warning("二维降维失败，跳过散点坐标：%s", exc)
                        warnings.append("二维可视化坐标生成失败，结果本身不受影响。")
                else:
                    warnings.append("样本或向量维度不足，未生成二维可视化坐标。")

            # ④ 簇摘要：代表样本、关键词、元数据分布（展示辅助，不改动聚类结果）
            clusters, items = build_digest(
                sample_ids=sample_ids,
                texts=texts,
                labels=labels,
                vectors=transformed,
                metadata=metadata,
                top_keywords=self.settings.digest_top_keywords,
                representatives=self.settings.digest_representatives,
            )

            non_noise = [cluster for cluster in clusters if cluster["cluster_id"] != NOISE_CLUSTER_ID]
            sizes = [cluster["size"] for cluster in non_noise]
            gateway_ms = round((time.monotonic() - started) * 1000)
            summary = {
                "run_id": result["run_id"],
                "total_samples": len(sample_ids),
                "cluster_count": len(non_noise),
                "noise_count": int((labels == NOISE_CLUSTER_ID).sum()),
                "largest_cluster_size": max(sizes) if sizes else 0,
                "smallest_cluster_size": min(sizes) if sizes else 0,
                "avg_cluster_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
                "algorithm": result["algorithm"],
                "profile_id": result["profile_id"],
                "model_id": profile.model_id,
                "implementation_version": result.get("implementation_version") or profile.implementation_version,
                "embedding_dimension": int(transformed.shape[1]),
                "cache_hit": bool(cache_hit),
                "elapsed_ms": int(result.get("elapsed_ms", 0)),
                "gateway_ms": gateway_ms,
                "warnings": warnings,
            }
            return {
                "summary": summary,
                "clusters": clusters,
                "items": items,
                "visualization": visualization,
            }
        finally:
            self._run_lock.release()

    # ------------------------------------------------------------------ semantic 分支

    def _run_semantic(
        self,
        *,
        engine: Any,
        profile: Any,
        sample_ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        """semantic-v1：引擎产出完整语义载荷，网关只做契约拼装与守恒校验。

        与 legacy 分支最本质的差别是**不做二次加工**：

        * 不复用 `FeaturePipeline` / `DimReducer` —— 那会把同一批文本再编码一次，
          既慢，又让「向量口径」出现两个来源（引擎的语义空间 vs 网关复现的变换）；
        * clusters / items / visualization 原样透传，保证"展示的就是引擎算的"；
        * 命名、质量分、归属状态、二维抽样全部用引擎的语义口径，
          网关不重新打分、不重命名、不重新抽样。

        网关只补两件引擎不知道的事：网关总耗时，以及按契约做的守恒断言
        ——宁可报错，也不返回对不上的结果。
        """

        if not self._run_lock.acquire(blocking=False):
            raise ServiceBusyError("已有聚类任务正在执行，请稍后重试。")
        try:
            try:
                result = engine.cluster(
                    {
                        "profile_id": profile.profile_id,
                        "items": [
                            {"id": ident, "text": text, "metadata": meta}
                            for ident, text, meta in zip(sample_ids, texts, metadata)
                        ],
                    },
                    enforce_api_limits=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("语义聚类执行失败：%s", exc)
                raise _from_engine_error(exc) from exc
        finally:
            self._run_lock.release()

        semantic = result.get("semantic")
        if not isinstance(semantic, dict):
            raise ClusteringError("引擎未返回 semantic-v1 结果集。")
        engine_summary = dict(semantic.get("summary") or {})
        clusters = list(semantic.get("clusters") or [])
        items = list(semantic.get("items") or [])
        visualization = list(semantic.get("visualization") or [])
        if len(items) != len(sample_ids):
            raise ClusteringError("引擎返回的逐条结果数量与样本数不一致。")

        warnings = list(result.get("warnings") or [])
        reduction = engine_summary.get("reduction") or {}
        # 原始模型维度（而非降维后的拟合维度）才是"向量模型是几维"的答案
        source_dim = int(reduction.get("source_dim") or 0)
        misses = int(engine_summary.get("embedding_cache_misses") or 0)
        non_noise = [cluster for cluster in clusters if int(cluster["cluster_id"]) != NOISE_CLUSTER_ID]
        sizes = [int(cluster["size"]) for cluster in non_noise]
        gateway_ms = round((time.monotonic() - started) * 1000)

        summary: dict[str, Any] = {
            # —— 兼容字段：与 legacy 同名字段语义对齐，旧前端无需改动 ——
            "run_id": result["run_id"],
            "total_samples": int(engine_summary.get("total_samples") or len(sample_ids)),
            "cluster_count": int(engine_summary.get("cluster_count") or len(non_noise)),
            "noise_count": int(engine_summary.get("noise_count") or 0),
            "largest_cluster_size": int(engine_summary.get("largest_cluster_size") or (max(sizes) if sizes else 0)),
            "smallest_cluster_size": int(engine_summary.get("smallest_cluster_size") or (min(sizes) if sizes else 0)),
            "avg_cluster_size": float(engine_summary.get("avg_cluster_size") or 0.0),
            "algorithm": result["algorithm"],
            "profile_id": result["profile_id"],
            "model_id": profile.model_id,
            "implementation_version": result.get("implementation_version", "semantic-v1"),
            "embedding_dimension": source_dim,
            # 语义路径的 cache_hit 口径是"本次没有发生任何新的向量推理"，
            # 与 legacy 的"批缓存命中"是同一个问题的两种表述。
            "cache_hit": misses == 0,
            "elapsed_ms": int(result.get("elapsed_ms", 0)),
            "gateway_ms": gateway_ms,
            "warnings": warnings,
            # —— semantic-v1 扩展字段：全部来自引擎，网关不重算 ——
            "invalid_count": int(engine_summary.get("invalid_count") or 0),
            "duplicate_count": int(engine_summary.get("duplicate_count") or 0),
            "unique_count": int(engine_summary.get("unique_count") or 0),
            "coverage": engine_summary.get("coverage"),
            "noise_ratio": engine_summary.get("noise_ratio"),
            "selected_k": engine_summary.get("selected_k"),
            "final_k": engine_summary.get("final_k"),
            "auto_k_status": engine_summary.get("auto_k_status"),
            "quality_score": engine_summary.get("quality_score"),
            "weighted_quality_score": engine_summary.get("weighted_quality_score"),
            "overall_quality": engine_summary.get("overall_quality"),
            "quality_version": engine_summary.get("quality_version"),
            "confidence_version": engine_summary.get("confidence_version"),
            "naming_version": engine_summary.get("naming_version"),
            "normalization_version": engine_summary.get("normalization_version"),
            "calibration_version": engine_summary.get("calibration_version"),
            "calibration_status": engine_summary.get("calibration_status"),
            "effective_model_id": engine_summary.get("effective_model_id"),
            "device": engine_summary.get("device"),
            "seed": engine_summary.get("seed"),
            "embedding_cache_hits": engine_summary.get("embedding_cache_hits"),
            "embedding_cache_misses": engine_summary.get("embedding_cache_misses"),
            "embedding_cache_hit_ratio": engine_summary.get("embedding_cache_hit_ratio"),
            "cache_only": bool(engine_summary.get("cache_only", False)),
            "fallback_reason": engine_summary.get("fallback_reason"),
            "visualization_sample_count": engine_summary.get("visualization_sample_count") or len(visualization),
            "visualization_total_count": engine_summary.get("visualization_total_count"),
            "stage_timings": engine_summary.get("stage_timings"),
            "diagnostics": engine_summary.get("diagnostics"),
            "auto_k": engine_summary.get("auto_k"),
            "reduction": reduction or None,
            "postprocess": engine_summary.get("postprocess"),
        }
        return {
            "summary": summary,
            "clusters": clusters,
            "items": items,
            "visualization": visualization,
        }

    # ------------------------------------------------------------------ spear 分支

    def _run_spear(
        self,
        *,
        engine: Any,
        profile: Any,
        sample_ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]],
        options: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        """spear-v1：净化 + 表示增强 + 聚类，标签由引擎给出，网关只做契约拼装。

        与 legacy 分支最关键的差别是**网关不再复现一次向量**。原因是 SPEAR 的
        聚类空间来自"净化后的文本"，而网关手里只有原文；若为了画散点再编码一次
        原文，展示用的坐标就与真实聚类空间不是同一个口径。宁可如实标注
        "本路径不生成二维坐标"，也不返回一张与算法无关的图。
        """

        if not self._run_lock.acquire(blocking=False):
            raise ServiceBusyError("已有聚类任务正在执行，请稍后重试。")
        try:
            engine_request: dict[str, Any] = {
                "profile_id": profile.profile_id,
                "items": [
                    {"id": ident, "text": text, "metadata": meta}
                    for ident, text, meta in zip(sample_ids, texts, metadata)
                ],
            }
            # 请求级净化开关：只透传"开/关"，后端/护栏仍按 profile 决定
            override = options.get("purify")
            if override is not None:
                engine_request["purification"] = {"enabled": bool(override)}
            # 请求级知识库覆盖：SPEAR 检索增强 profile 允许改用偏差数据库
            engine_request.update(self._knowledge_base_request_field(profile))
            try:
                result = engine.cluster(engine_request, enforce_api_limits=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SPEAR 聚类执行失败：%s", exc)
                raise _from_engine_error(exc) from exc
        finally:
            self._run_lock.release()

        labels = np.asarray([item["cluster_id"] for item in result["assignments"]], dtype=int)
        if labels.shape != (len(sample_ids),):
            raise ClusteringError("引擎返回的标签数量与样本数不一致。")

        report = dict(result.get("purification") or {})
        condensed = {sample["id"]: sample["condensed"] for sample in report.get("samples") or []}
        warnings = list(result.get("warnings") or [])
        # 诚实标注：本路径没有生成二维坐标，前端不应把它当成"点了太少"
        warnings.append("SPEAR_VIZ_NOT_COMPUTED")

        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)

        clusters: list[dict[str, Any]] = []
        for cluster_id in sorted(groups, key=lambda value: (value == NOISE_CLUSTER_ID, value)):
            members = groups[cluster_id]
            is_noise = cluster_id == NOISE_CLUSTER_ID
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "label": "未归类" if is_noise else f"簇 {cluster_id}",
                    "size": len(members),
                    "keywords": [],
                    "cohesion": None,
                    "representative_samples": [
                        {
                            "id": sample_ids[index],
                            "text": condensed.get(sample_ids[index]) or texts[index],
                            "confidence": None,
                            "distance": None,
                        }
                        for index in members[: self.settings.digest_representatives]
                    ],
                    "metadata_distribution": {},
                    "is_noise_bucket": is_noise,
                }
            )

        items: list[dict[str, Any]] = []
        for index, cluster_id in enumerate(labels):
            ident = sample_ids[index]
            items.append(
                {
                    "id": ident,
                    "text": texts[index],
                    "cluster_id": int(cluster_id),
                    "cluster_label": "未归类" if int(cluster_id) == NOISE_CLUSTER_ID else f"簇 {int(cluster_id)}",
                    "confidence": None,
                    "distance": None,
                    "keywords": [],
                    "metadata": metadata[index],
                    # 只有引擎确实返回了对照（上限 50 条）时才有值，其余为 null
                    "purified_text": condensed.get(ident),
                }
            )

        non_noise = [cluster for cluster in clusters if cluster["cluster_id"] != NOISE_CLUSTER_ID]
        sizes = [cluster["size"] for cluster in non_noise]
        gateway_ms = round((time.monotonic() - started) * 1000)
        try:
            embedding_dimension = int(self._catalog.model(profile.model_id)["dimension"])
        except Exception:  # noqa: BLE001 - 维度只用于展示，取不到不影响结果
            embedding_dimension = 0
        summary = {
            "run_id": result["run_id"],
            "total_samples": len(sample_ids),
            "cluster_count": len(non_noise),
            "noise_count": int((labels == NOISE_CLUSTER_ID).sum()),
            "largest_cluster_size": max(sizes) if sizes else 0,
            "smallest_cluster_size": min(sizes) if sizes else 0,
            "avg_cluster_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
            "algorithm": result["algorithm"],
            "profile_id": result["profile_id"],
            "model_id": profile.model_id,
            "implementation_version": result.get("implementation_version") or profile.implementation_version,
            "embedding_dimension": embedding_dimension,
            "cache_hit": False,
            "elapsed_ms": int(result.get("elapsed_ms", 0)),
            "gateway_ms": gateway_ms,
            "warnings": warnings,
            "purification": report or None,
        }
        return {
            "summary": summary,
            "clusters": clusters,
            "items": items,
            "visualization": [],
        }
