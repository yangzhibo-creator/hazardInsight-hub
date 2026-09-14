"""贯穿全流程的数据契约。

这里的 dataclass 是各层之间传递数据的唯一形式：数据加载 -> 编码 -> 特征变换 -> 聚类，
每一环都只认这些结构，从而保证"数据是什么"和"怎么算"相互独立。
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np
from .errors import ClusterError
from .artifacts.fingerprints import fingerprint


@dataclass
class TextDataset:
    """一份文本数据集，附带可选的真实标签（用于评估）。"""

    dataset_id: str  # 数据集标识（由来源文件内容哈希派生）
    sample_ids: list[str]  # 每条文本的唯一 ID，保证结果可回溯到原始样本
    texts: list[str]  # 文本本身
    labels: np.ndarray | None = None  # 真实标签；无监督场景为 None
    label_names: dict[int, str] = field(default_factory=dict)  # 标签编号 -> 可读名称
    source_fingerprint: str = ""  # 来源文件的内容指纹，便于复现与审计


@dataclass
class EmbeddingBatch:
    """一批文本向量。"""

    sample_ids: list[str]  # 与 values 的行一一对应
    values: np.ndarray  # 形状 (n_samples, dimension) 的浮点矩阵
    model_fingerprint: str  # 编码模型指纹：换模型/换设备即失效
    processing_fingerprint: str = "raw-v1"  # 预处理口径版本（是否归一化等）


@dataclass
class NeighborBatch:
    """检索结果：每个样本的 k 个近邻。"""

    ids: list[list[str]]  # 每个样本的近邻 ID 列表，外层长度 = 样本数
    values: Any  # 近邻向量，形状 (n_samples, k, dimension)
    knowledge_base_id: str  # 来源知识库标识
    distances: Any = None  # 可选的距离，当前流程未使用


@dataclass(frozen=True)
class FeatureSpec:
    """特征融合与降维的口径。

    frozen=True 是有意为之：该对象一旦构造就不可变，避免在流程中途被修改
    导致"同一配置产生不同结果"。所有非法取值都在构造时立刻报错。
    """

    beta: float = 1.0  # 融合权重：beta*原始向量 + (1-beta)*邻居均值
    n_results: int = 0  # 检索近邻数 k；0 表示不做检索增强
    pca_dim: int = 0  # 降维目标维度；0 表示不降维
    reduction: str = "PCA"  # 降维算法："PCA" 或 "UMAP"
    version: str = "legacy-v1"  # 特征处理的行为版本，决定边界路径（见 pipeline.py）

    def __post_init__(self):
        # beta 必须在 [0, 1] 且为有限数，否则融合结果无意义
        if not np.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ClusterError("INVALID_PROFILE", "beta must be within [0, 1]", 422)
        # 类型必须严格是 int（布尔是 int 的子类，这里要排除 True/False）
        if type(self.n_results) is not int or self.n_results < 0 or type(self.pca_dim) is not int or self.pca_dim < 0:
            raise ClusterError("INVALID_PROFILE", "Inference requires resolved nonnegative dimensions and k", 422)
        # 版本白名单：约束住可以被复现的行为集合。
        # semantic-v1 是"标记位"，语义流程不经过 FeaturePipeline，
        # 它的 L2/PCA 由 features/semantic.py 按自己的确定性规则完成。
        # spear-v1 复用 legacy 的 FeaturePipeline（检索增强 + 降维），
        # 唯一差异是"送进管道的矩阵来自净化后文本"。
        if self.reduction not in {"PCA", "UMAP"} or self.version not in {
            "legacy-v1",
            "legacy-radbscan-v1",
            "semantic-v1",
            "spear-v1",
        }:
            raise ClusterError("INVALID_PROFILE", "Unknown feature behavior version", 422)


#: 合法的净化后端（与 purification.registry 的实现一一对应）。
#: 放在 types 里是为了让 profile 的构造期校验与运行时实现共用同一份白名单，
#: 避免"配置写了一个后端名、运行时悄悄回落到默认值"这种静默不一致。
PURIFICATION_BACKENDS = ("auto", "qwen", "rule", "cache")


@dataclass(frozen=True)
class PurificationSpec:
    """SPEAR 阶段 1（语义净化）的配置口径，对应 profile JSON 里的 ``purification``。

    为什么要独立于 ``FeatureSpec``：
    * 生命周期不同——净化是**输入层**变换，特征规范是**表示层**变换；
    * 依赖不同——净化依赖本地 LLM（可选），特征只依赖向量模型；
    * 开关语义不同——``enabled=False`` 时整条链路必须退化成改动前的行为，
      这需要单独一个显式字段来承载，而不是靠某个数值恰好等于默认值。
    """

    enabled: bool = True  # 总开关：false 时退化为改动前的纯向量/检索路径
    backend: str = "auto"  # auto | qwen | rule | cache
    model_path: str | None = None  # 本地 Qwen 权重目录（backend 需要 LLM 时必填）
    cache_file: str | None = None  # 预计算映射 JSON（backend=cache / auto 优先命中）
    device: str = "cuda"
    batch_size: int = 32
    max_new_tokens: int = 64
    guard: bool = True  # 极性保真护栏（默认必须开，见陷阱 P1）

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.guard, bool):
            raise ClusterError("INVALID_PROFILE", "Purification toggles must be booleans", 422)
        if self.backend not in PURIFICATION_BACKENDS:
            raise ClusterError("INVALID_PROFILE", "Unknown purification backend", 422)
        # 声明了需要权重的后端，就必须给出路径；否则配置在语义上不完整，
        # 应该在加载期失败，而不是等到运行时才发现"无处可加载"。
        if self.backend == "qwen" and not self.model_path:
            raise ClusterError("INVALID_PROFILE", "Purification backend 'qwen' requires a model path", 422)
        if self.backend == "cache" and not self.cache_file:
            raise ClusterError("INVALID_PROFILE", "Purification backend 'cache' requires a cache file", 422)
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ClusterError("INVALID_PROFILE", "Purification batch size must be a positive integer", 422)
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or not 1 <= self.max_new_tokens <= 4096
        ):
            raise ClusterError("INVALID_PROFILE", "Purification max_new_tokens is out of range", 422)


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    """一次聚类任务所需的全部已解析配置（对应 configs/profiles/*.json）。

    它是可复现性的核心：只要这份配置相同、输入相同、模型相同，结果就应当相同。
    """

    profile_id: str  # 配置标识，API 请求通过它选择算法与参数
    algorithm: str  # 聚类算法名，需能在 clustering.registry.ALGORITHMS 中找到
    model_id: str  # 使用的编码模型，需能在 models.toml 中找到
    algorithm_params: dict  # 传给算法函数的超参（不含特征相关字段）
    features: FeatureSpec = field(default_factory=FeatureSpec)  # 特征处理口径
    knowledge_base_id: str | None = None  # 检索增强所用的知识库
    implementation_version: str = "legacy-v1"  # 实现版本，当前只支持历史口径
    backend: str = "python"  # 执行后端，当前冻结为纯 Python
    compatibility_status: str = "legacy"  # 与历史实验的兼容性标记
    source_result: str | None = None  # 历史来源结果文件（仅作溯源）
    source_sha256: str | None = None  # 上述文件的校验和
    max_samples: int = 300_000  # 单批样本数上限（与 configs/profiles/*.json 一致）
    calibration_id: str | None = None  # semantic-v1 引用的校准配置标识；legacy 为 None
    #: SPEAR（spear-v1）的净化口径；其它版本为 None。
    purification: PurificationSpec | None = None
    capability: dict = field(default_factory=dict)  # 策略能力元信息（批量上限等，可缺省）

    @classmethod
    def from_dict(cls, raw):
        """从 profile JSON 字典构造，并在构造阶段完成全部合法性校验。

        校验的关键点是**字段联合约束**：版本、算法、特征规范三者必须自洽。
        这样"新算法配旧版本"这类错误组合会在加载配置时就以
        ``INVALID_PROFILE`` 失败，而不是等到编码之后才发现行为不对。
        """
        try:
            fields = {**raw, "features": FeatureSpec(**raw.get("features", {}))}
            # purification 是可选子对象：缺省表示"这个 profile 不做净化"（legacy/semantic），
            # 显式给出时按 PurificationSpec 的构造规则校验。
            if "purification" in raw and raw["purification"] is not None:
                fields["purification"] = PurificationSpec(**raw["purification"])
            obj = cls(**fields)
        except (TypeError, ValueError):
            # 字段名不对/类型不对，统一收敛为领域错误，避免泄漏底层异常细节
            raise ClusterError("INVALID_PROFILE", "Profile fields are invalid", 422) from None
        if obj.max_samples < 2:
            raise ClusterError("INVALID_PROFILE", "Unsupported sample limit", 422)
        if obj.implementation_version == "semantic-v1":
            if not obj.algorithm.startswith("semantic_"):
                raise ClusterError("INVALID_PROFILE", "semantic-v1 requires a semantic algorithm", 422)
            if obj.features.version != "semantic-v1":
                raise ClusterError("INVALID_PROFILE", "semantic-v1 requires the semantic feature spec", 422)
            if obj.features.n_results:
                raise ClusterError("INVALID_PROFILE", "semantic-v1 must not use retrieval augmentation", 422)
            if not obj.calibration_id:
                raise ClusterError("INVALID_PROFILE", "semantic-v1 requires a calibration id", 422)
            return obj
        if obj.implementation_version == "spear-v1":
            # SPEAR 复用 legacy 的算法与特征管道，但必须显式声明净化口径：
            # "没写 purification" 与 "写了 purification.enabled=false" 是两种不同意图，
            # 前者属于配置错误，后者才是"关掉净化做对照"。
            if obj.features.version != "spear-v1":
                raise ClusterError("INVALID_PROFILE", "spear-v1 requires the spear feature spec", 422)
            if obj.algorithm.startswith("semantic_"):
                raise ClusterError("INVALID_PROFILE", "spear-v1 does not support semantic algorithms", 422)
            if obj.purification is None:
                raise ClusterError("INVALID_PROFILE", "spear-v1 requires a purification spec", 422)
            # 与 legacy 同一条约束：要检索近邻就必须有知识库
            if obj.features.n_results and not obj.knowledge_base_id:
                raise ClusterError("INVALID_PROFILE", "Retrieval profile needs a knowledge base", 422)
            return obj

        if obj.implementation_version != "legacy-v1":
            raise ClusterError("INVALID_PROFILE", "Unsupported implementation version", 422)
        if obj.algorithm.startswith("semantic_"):
            raise ClusterError(
                "INVALID_PROFILE", "Semantic algorithms require implementation_version=semantic-v1", 422
            )
        # 需要检索近邻就必须指定知识库，否则无从查起
        if obj.features.n_results and not obj.knowledge_base_id:
            raise ClusterError("INVALID_PROFILE", "Retrieval profile needs a knowledge base", 422)
        # RADBSCAN 的历史行为依赖 legacy-radbscan-v1 这条特殊的特征顺序，不能被别的手滑改掉
        if obj.algorithm == "radbscan" and obj.features.version != "legacy-radbscan-v1":
            raise ClusterError("INVALID_PROFILE", "RADBSCAN requires its historical feature order", 422)
        return obj

    @property
    def fingerprint(self):
        """配置指纹：任何字段变化都会改变它，用于产物溯源与缓存键。"""
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class SemanticCalibration:
    """semantic-v1 的阈值与权重（对应 configs/semantic-calibration-v1.json）。

    为什么要单独一份、而不是塞进算法参数：

    * **口径不同**：算法参数是"算多少"的预算，校准参数是"算得对不对"的判据；
    * **生命周期不同**：换模型必须重校准，但预算可以不变；
    * **可审计**：校准结果带 ``status``（``experimental`` / ``validated``）
      与来源指纹。没有校准产物时只允许标 ``experimental``，
      这样"质量分"就不会被当成已验证的概率对外展示。

    所有阈值都在这里校验取值范围，绝不在代码里硬编码"0.7 就相似"。
    """

    version: str
    t_sem: float  # 最低成员语义支持
    t_pair: float  # 小簇成员互相支持
    t_merge: float  # 合并支持
    t_margin: float  # 最近/次近中心间隔
    t_single: float  # 单主题绝对门槛
    weights: dict  # Auto-K 组合分权重
    quality_weights: dict  # 簇质量分权重（C/S/R/E）
    confidence_weights: dict  # 归属置信度权重（A/M/L）
    scaling: dict  # 分量到 [0,1] 的绝对门槛映射
    naming: dict  # 命名与关键词参数
    model_id: str | None = None
    status: str = "experimental"
    scope: str | None = None
    source_fingerprint: str | None = None

    def __post_init__(self):
        for name in ("t_sem", "t_pair", "t_merge", "t_margin", "t_single"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
                raise ClusterError("INVALID_PROFILE", f"Calibration threshold {name} must be within (0, 1)", 422)
        if not self.is_ordered():
            raise ClusterError(
                "INVALID_PROFILE", "Calibration thresholds must satisfy t_sem <= t_single <= t_merge", 422
            )
        if self.status not in {"experimental", "validated"}:
            raise ClusterError("INVALID_PROFILE", "Calibration status must be experimental or validated", 422)
        for name, weights in (
            ("weights", self.weights),
            ("quality_weights", self.quality_weights),
            ("confidence_weights", self.confidence_weights),
            ("scaling", self.scaling),
        ):
            if not isinstance(weights, dict) or not weights:
                raise ClusterError("INVALID_PROFILE", f"Calibration {name} must be a non-empty mapping", 422)
            for key, value in weights.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ClusterError("INVALID_PROFILE", f"Calibration {name}.{key} must be numeric", 422)

    def is_ordered(self) -> bool:
        """阈值单调性：成员支持 ≤ 单主题门槛 ≤ 合并门槛。

        这个顺序不是审美要求——如果 ``t_merge`` 低于 ``t_sem``，
        就会出现"两条文本都不够格算同一主题，却够格合并两个簇"的荒谬结果。
        """

        return float(self.t_sem) <= float(self.t_single) <= float(self.t_merge)

    def thresholds(self) -> dict:
        """扁平化的阈值字典，供只关心数值的调用方使用。"""

        return {
            "t_sem": float(self.t_sem),
            "t_pair": float(self.t_pair),
            "t_merge": float(self.t_merge),
            "t_margin": float(self.t_margin),
            "t_single": float(self.t_single),
        }

    @classmethod
    def from_dict(cls, raw):
        """从 JSON 字典构造。

        兼容两种写法：阈值直接放在顶层，或放在 ``thresholds`` 子字典里。
        前者更紧凑，后者便于把"阈值"与"权重"分区存放。
        """

        payload = {key: value for key, value in raw.items() if key != "thresholds"}
        nested = raw.get("thresholds") or {}
        if not isinstance(nested, dict):
            raise ClusterError("INVALID_PROFILE", "Calibration thresholds must be a mapping", 422)
        payload.update(nested)
        try:
            return cls(**payload)
        except (TypeError, ValueError):
            raise ClusterError("INVALID_PROFILE", "Calibration fields are invalid", 422) from None

    def as_dict(self) -> dict:
        return asdict(self)
