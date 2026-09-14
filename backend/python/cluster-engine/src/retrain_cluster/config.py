"""配置加载：应用设置（app.toml）、模型目录（models.toml）与运行时配置（profiles/*.json）。

配置优先级（后者覆盖前者）：文件默认值 < 配置文件 < 环境变量 < 显式 overrides。
这样既能让运维用环境变量快速改部署参数，又不破坏代码内的默认契约。
"""

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import tomllib
from .errors import ClusterError
from .types import ResolvedPipelineConfig, SemanticCalibration
from .artifacts.fingerprints import fingerprint, file_hash


def resolve_path(base, value):
    """把配置里的相对路径解析成绝对路径（相对于配置文件所在目录）。

    支持 ``~`` 展开；绝对路径原样保留。这样配置文件可以被放在任意位置而路径仍正确。
    """
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


@dataclass
class Settings:
    """应用级设置，集中所有可调开关与资源上限。"""

    config_path: Path
    models_file: Path
    profiles_dir: Path
    artifacts_dir: Path
    # 默认值与 configs/app.toml 保持一致（30 万条 / 200 MiB / 1 小时）
    max_samples: int = 300_000  # 单次请求最大样本数
    max_text_length: int = 4000  # 单条文本最大字符数
    max_body_bytes: int = 200 * 1024 * 1024  # 请求体最大字节数（200 MiB）
    timeout_seconds: float = 3600.0  # 单次聚类执行超时
    host: str = "127.0.0.1"
    port: int = 8000
    query_batch: int = 64  # 检索分批大小
    #: semantic-v1 的校准配置（阈值/权重），相对配置文件所在目录解析。
    calibration_file: Path | None = None
    # 实验（调参）默认参数
    experiment: dict = field(default_factory=lambda: {"n_trials": 200, "n_jobs": 8, "seed": 42, "startup_trials": 15})

    @classmethod
    def load(cls, path=None, overrides=None):
        """加载并校验配置。

        路径优先级：显式 path 参数 > 环境变量 RETRAIN_CONFIG > 默认 configs/app.toml。
        """
        path = Path(path or os.environ.get("RETRAIN_CONFIG", "configs/app.toml")).resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        app = raw.get("app", {})
        # 逐项用环境变量覆盖（RETRAIN_XXX），便于容器化部署时免改文件
        for name in ["models_file", "profiles_dir", "artifacts_dir", "max_samples", "timeout_seconds", "host", "port"]:
            env = os.environ.get("RETRAIN_" + name.upper())
            if env is not None:
                app[name] = env
        # 显式 overrides 优先级最高；值为 None 表示"不覆盖"
        app.update({k: v for k, v in (overrides or {}).items() if v is not None})
        # TOML 里数字可能被写成字符串（尤其经环境变量传入），这里统一转成 int
        for name in ["max_samples", "max_text_length", "max_body_bytes", "port", "query_batch"]:
            if name in app:
                app[name] = int(app[name])
        if "timeout_seconds" in app:
            app["timeout_seconds"] = float(app["timeout_seconds"])
        # 路径类字段统一解析为绝对路径
        paths = {
            name: resolve_path(base, app.pop(name, default))
            for name, default in [
                ("models_file", "models.toml"),
                ("profiles_dir", "profiles"),
                ("artifacts_dir", "../artifacts"),
                ("calibration_file", "semantic-calibration-v1.json"),
            ]
        }
        exp = {"n_trials": 200, "n_jobs": 8, "seed": 42, "startup_trials": 15, **raw.get("experiment", {})}
        # 实验参数同样支持环境变量覆盖
        for name, env in [("n_trials", "N_TRIALS"), ("n_jobs", "N_JOBS"), ("seed", "SEED")]:
            if env in os.environ:
                exp[name] = int(os.environ[env])
        # 三个输入数据文件路径也解析成绝对路径
        for name in ["reference", "test", "knowledge_texts"]:
            if name in exp:
                exp[name] = str(resolve_path(base, exp[name]))
        result = cls(path, **paths, **app, experiment=exp)
        # 基本不变量：样本数至少 2（聚类前提），超时为正，分批大小为正值
        if result.max_samples < 2 or result.timeout_seconds <= 0 or result.query_batch < 1:
            raise ValueError("Invalid application limits")
        return result


class Catalog:
    """模型目录与配置档（profile）目录，进程启动时一次性加载并校验。"""

    def __init__(self, settings):
        self.settings = settings
        raw = tomllib.loads(settings.models_file.read_text(encoding="utf-8"))
        self.models = {}
        for model in raw.get("models", []):
            model = dict(model)
            if type(model.get("dimension")) is not int or model["dimension"] < 1:
                raise ValueError("Model dimension must be a positive integer")
            if model.get("batch_size", 64) < 1:
                raise ValueError("Model batch size must be positive")
            # 远程编码器由服务端决定是否归一化，客户端不应再做一次（会改变历史口径）
            if model["provider"] == "openai_compatible" and model.get("normalize", False):
                raise ValueError("Legacy remote encoder does not apply client-side normalization")
            # 各 provider 的向量池化方式必须与其实现匹配，否则向量语义会变
            expected_pooling = {"simcse": "cls", "sccl": "mean", "openai_compatible": "provider"}.get(model["provider"])
            if expected_pooling and model.get("pooling", expected_pooling) != expected_pooling:
                raise ValueError("Pooling setting does not match provider implementation")
            # 本地模型必须用绝对路径，避免依赖运行时 cwd
            if model["provider"] != "openai_compatible":
                model["source"] = str(resolve_path(settings.models_file.parent, model["source"]))
                if model.get("tokenizer_source"):
                    model["tokenizer_source"] = str(
                        resolve_path(settings.models_file.parent, model["tokenizer_source"])
                    )
            if model["model_id"] in self.models:
                raise ValueError("Duplicate model ID")
            self.models[model["model_id"]] = model
        self.profiles = {}
        self.calibrations: dict[str, SemanticCalibration] = {}
        # 校准配置是可选文件：缺失时 legacy 流程照常工作，
        # 只有引用它的 semantic profile 会在加载阶段失败（fail fast，且错误明确）。
        calibration_path = settings.calibration_file
        if calibration_path and Path(calibration_path).is_file():
            raw_calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            for ident, item in (raw_calibration.get("versions") or {}).items():
                self.calibrations[str(ident)] = SemanticCalibration.from_dict(dict(item))
        # 按文件名排序加载，保证加载顺序稳定（进而让错误信息稳定）
        for path in sorted(settings.profiles_dir.glob("*.json")):
            profile = ResolvedPipelineConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
            # 净化用的本地 LLM 与预计算映射按"配置目录"解析相对路径，
            # 与向量模型 source 的解析口径一致；解析后写回，运行期不再依赖 cwd。
            if profile.purification is not None:
                purification = profile.purification
                resolved = {}
                if purification.model_path:
                    resolved["model_path"] = str(resolve_path(settings.models_file.parent, purification.model_path))
                if purification.cache_file:
                    resolved["cache_file"] = str(resolve_path(settings.models_file.parent, purification.cache_file))
                if resolved:
                    profile = replace(profile, purification=replace(purification, **resolved))
            if profile.profile_id in self.profiles:
                raise ValueError("Duplicate profile ID")
            # profile 引用的模型必须存在，否则启动即失败（fail fast）
            if profile.model_id not in self.models:
                raise ValueError("Unknown model in profile")
            # semantic profile 引用的校准版本必须存在，否则"质量分"将无据可依
            if profile.calibration_id and profile.calibration_id not in self.calibrations:
                raise ValueError("Unknown calibration in profile")
            self.profiles[profile.profile_id] = profile

    def profile(self, ident):
        if ident not in self.profiles:
            raise ClusterError("PROFILE_NOT_FOUND", "Profile not found", 404)
        return self.profiles[ident]

    def model(self, ident):
        if ident not in self.models:
            raise ClusterError("MODEL_NOT_FOUND", "Model not found", 404)
        return self.models[ident]

    def calibration(self, ident):
        """取回校准配置；不存在时返回 None（legacy profile 本来就不需要）。"""

        if not ident:
            return None
        if ident not in self.calibrations:
            raise ClusterError("CALIBRATION_NOT_FOUND", "Calibration not found", 404)
        return self.calibrations[ident]


def model_fingerprint(spec):
    """模型指纹：设备/来源/编码行为参与计算，凭据（api_key_env）绝不参与。

    这一点很重要——指纹会写进产物并用做缓存键，不能因为它随密钥变化而使缓存失效，
    更不能把密钥名本身泄漏到产物里。
    """
    # Device/source/encoding behavior participate; credentials never do.
    return fingerprint({"schema": "model-v1", **{k: v for k, v in spec.items() if k != "api_key_env"}})


def verify_local_model(spec):
    """校验本地模型目录：目录存在、清单非空、所有文件校验和一致。

    目的有二：确保模型文件完整（避免加载到半截文件），以及确保"用的就是记录在案的
    那个模型"（可复现性）。tokenizer 若在独立目录，则递归做同样的校验。
    """
    directory = Path(spec["source"])
    checksums = spec.get("file_checksums", {})
    if not directory.is_dir() or not checksums:
        raise ClusterError("MODEL_UNAVAILABLE", "Local model must have a validated file manifest", 503)
    # 清单里声明的必需文件必须都有校验和记录
    for relative in spec.get("required_files", []):
        if relative not in checksums:
            raise ClusterError("MODEL_UNAVAILABLE", "Required model file is missing from manifest", 503)
    for relative, expected in checksums.items():
        path = (directory / relative).resolve()
        # 三重校验：不得越出模型目录（防路径穿越）、必须是文件、哈希必须一致
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ClusterError("MODEL_UNAVAILABLE", "Local model file checksum mismatch", 503)

    tokenizer = spec.get("tokenizer_source")
    # tokenizer 与模型同目录则无需重复校验，否则递归校验 tokenizer 目录
    if tokenizer and Path(tokenizer).resolve() != directory.resolve():
        verify_local_model(
            {
                "source": tokenizer,
                "file_checksums": spec.get("tokenizer_checksums", {}),
                "required_files": spec.get("tokenizer_required_files", []),
            }
        )
