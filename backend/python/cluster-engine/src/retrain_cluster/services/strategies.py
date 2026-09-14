"""实现版本注册表：把 profile 的 ``implementation_version`` 映射到具体 Strategy。

这是"增量接入"的关键闸门（实施计划 10.1）：

* ``legacy-v1`` → 走既有路径，**一行行为都不改**；
* ``semantic-v1`` → 走新的语义策略。

Catalog 在加载阶段就做了字段联合校验（版本 ↔ 算法 ↔ 特征规范），
因此这里的分派不需要再猜"这个组合是不是合法"。分派表按版本号显式登记，
而不是按算法名 `if/elif` 猜——否则新增算法时很容易悄悄落回 legacy 分支。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..clustering.registry import SEMANTIC_ALGORITHMS as _REGISTERED_SEMANTIC_ALGORITHMS
from ..clustering.registry import ALGORITHMS as _REGISTERED_ALGORITHMS
from ..errors import ClusterError

#: 历史实现版本（默认，行为冻结）。
LEGACY_VERSION = "legacy-v1"

#: 新的中文语义自动聚类实现版本。
SEMANTIC_VERSION = "semantic-v1"

#: SPEAR（语义净化 + 表示增强 + 凝聚层次聚类）实现版本。
SPEAR_VERSION = "spear-v1"

#: semantic-v1 当前支持的算法标识。
#:
#: **从算法注册表派生，不在这里手写一份。** 手写副本一旦与
#: `clustering.registry.ALGORITHMS` 漂移，能力表就会声称支持一个根本没有实现的
#: 算法（例如计划里排在阶段 4 的 `semantic_hdbscan`）：网关会照此把该算法报成
#: `semantic-v1`、按语义规则放行单条请求，而真正的失败要等到运行时才暴露。
#: 派生之后"注册了才算支持"成为结构性事实，不需要靠人工同步维持。
SEMANTIC_ALGORITHMS = tuple(_REGISTERED_SEMANTIC_ALGORITHMS)

#: spear-v1 支持的算法：复用 legacy 的全部矩阵算法（SPEAR 只是换了输入与表示口径，
#: 不改算法集合）。semantic_* 属于另一条实现路径，不能混进来。
SPEAR_ALGORITHMS = tuple(name for name in _REGISTERED_ALGORITHMS if not name.startswith("semantic_"))


@dataclass(frozen=True)
class StrategyMeta:
    """Strategy 的版本与能力元信息，供 /profiles、/algorithms 与产物清单使用。"""

    implementation_version: str
    algorithms: tuple[str, ...]
    supports_single_item: bool = False
    supports_cache_only: bool = False
    calibration_version: str | None = None
    features: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "implementation_version": self.implementation_version,
            "algorithms": list(self.algorithms),
            "supports_single_item": self.supports_single_item,
            "supports_cache_only": self.supports_cache_only,
            "calibration_version": self.calibration_version,
            "features": dict(self.features),
        }


class Strategy(Protocol):
    """上层策略契约。

    底层算法注册表仍然只认"矩阵进、标签出"；Strategy 才是接收
    规范化文本、ID/频次映射、模型服务、缓存与 profile 的那一层。
    """

    meta: StrategyMeta

    def run(self, items, *, enforce_api_limits: bool = False) -> dict: ...


def strategy_for_version(implementation_version: str) -> str | None:
    """返回该版本对应的策略标识；legacy 返回 None（表示走既有路径）。"""

    if implementation_version == LEGACY_VERSION:
        return None
    if implementation_version == SEMANTIC_VERSION:
        return "semantic"
    if implementation_version == SPEAR_VERSION:
        return "spear"
    raise ClusterError("INVALID_PROFILE", "Unknown implementation version", 422)


def build_strategy(profile, *, settings, catalog, encoders, text_cache=None, embedder=None, retriever_for=None, purifiers=None) -> Any | None:
    """按 profile 构造 Strategy；legacy profile 返回 None。

    这里通过 ``strategy_for_version`` 查表而不是判断算法名，
    因此"新算法 + 旧版本"这种非法组合会在 Catalog 阶段就被拒绝，
    不会走到这里再静默降级。

    ``embedder`` / ``retriever_for`` / ``purifiers`` 由 ClusteringService 注入：
    SPEAR 必须复用 legacy 的嵌入缓存与检索器实例（缓存命中与实例复用是
    "关掉净化后逐位一致"的前提），而不是自己另建一套。
    """

    kind = strategy_for_version(profile.implementation_version)
    if kind is None:
        return None
    if kind == "semantic":
        # 延迟导入：legacy 路径完全不需要加载语义模块与其依赖
        from .semantic_clustering import SemanticAutoKMeansStrategy

        return SemanticAutoKMeansStrategy(
            profile=profile,
            settings=settings,
            catalog=catalog,
            encoders=encoders,
            text_cache=text_cache,
        )
    if kind == "spear":
        from .spear_clustering import SpearStrategy

        return SpearStrategy(
            profile=profile,
            settings=settings,
            catalog=catalog,
            encoders=encoders,
            embedder=embedder,
            retriever_for=retriever_for,
            purifiers=purifiers,
        )
    raise ClusterError("INVALID_PROFILE", "Unknown strategy kind", 422)


def list_strategy_versions() -> list[dict]:
    """列出当前代码支持的全部实现版本（供能力清单接口使用）。"""

    return [
        {
            "implementation_version": LEGACY_VERSION,
            "algorithms": None,  # None 表示"沿用算法注册表的 API 列表"
            "supports_single_item": False,
            "supports_cache_only": False,
            "calibration_version": None,
        },
        {
            "implementation_version": SEMANTIC_VERSION,
            "algorithms": list(SEMANTIC_ALGORITHMS),
            "supports_single_item": True,
            "supports_cache_only": True,
            "calibration_version": None,  # 由 profile 自带，运行时才确定
        },
        {
            "implementation_version": SPEAR_VERSION,
            "algorithms": list(SPEAR_ALGORITHMS),
            "supports_single_item": False,
            "supports_cache_only": False,
            "calibration_version": None,
            "purification": True,  # SPEAR 的输入层净化是这一版的标志性能力
        },
    ]


__all__ = [
    "LEGACY_VERSION",
    "SEMANTIC_ALGORITHMS",
    "SEMANTIC_VERSION",
    "SPEAR_ALGORITHMS",
    "SPEAR_VERSION",
    "Strategy",
    "StrategyMeta",
    "build_strategy",
    "list_strategy_versions",
    "strategy_for_version",
]
