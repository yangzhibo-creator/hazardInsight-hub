"""净化器注册表：实例复用 + 降级链 + 可用性如实报告。

三条设计约束（全部来自实测）：

1. **同一进程只装载一份 Qwen**（陷阱 P3）。第二个实例会申请约 51 GiB 并直接
   CUDA OOM，因此这里按 ``(后端, 模型路径, 批大小, max_new_tokens, 护栏)`` 缓存实例。
2. **降级不能中断服务**（交付物 A2.5）。Qwen 权重缺失 / 加载失败时自动改用
   规则净化，流程照跑，但状态里必须写明"已降级 + 原因"，绝不假装一切正常。
3. **可用性探测不能有副作用**。``probe()`` 只做文件系统与依赖检查，不加载权重；
   现场在 ``/health`` 与 profile 列表里看到的就是它给出的结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BasePurifier, IdentityPurifier, Purifier
from .cache import CachedPurifier
from .guard import GuardedPurifier
from .rules import RulePurifier
from ..types import PURIFICATION_BACKENDS

#: 需要本地 LLM 的后端。
_LOCAL_LLM_BACKENDS = frozenset({"qwen", "auto"})
#: 全部合法后端。与 profile 构造期校验共用同一份白名单（types.PURIFICATION_BACKENDS）。
BACKENDS = frozenset(PURIFICATION_BACKENDS)


@dataclass(frozen=True)
class PurificationStatus:
    """一次净化配置在当前环境下的真实状态。"""

    enabled: bool
    requested_backend: str
    effective_backend: str
    degraded: bool
    reason: str | None
    guarded: bool
    model_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "requested_backend": self.requested_backend,
            "effective_backend": self.effective_backend,
            "degraded": self.degraded,
            "reason": self.reason,
            "guarded": self.guarded,
            "model_path": self.model_path,
        }

    @property
    def warning(self) -> str | None:
        """降级时给出的稳定告警码，供结果 warnings 与 /health 使用。"""

        return "PURIFIER_DEGRADED_TO_RULE" if self.degraded else None


def _model_available(model_path: str | None) -> bool:
    """本地 LLM 权重是否就位：只看目录是否存在且非空。

    刻意不做逐文件校验和——Qwen 有 18 个分片、约 55 GB，现场每次健康检查都
    重算哈希会把 300 秒的启动预算吃光。校验和留给离线介质核对清单（md5/sha256）。
    """

    if not model_path:
        return False
    path = Path(model_path)
    if not path.is_dir():
        return False
    return any(child.is_file() for child in path.iterdir())


def _cache_available(cache_file: str | None) -> bool:
    return bool(cache_file) and Path(cache_file).is_file()


def probe(spec) -> PurificationStatus:
    """不加载任何权重，推断该配置会落到哪个后端。"""

    if not spec.enabled:
        return PurificationStatus(
            enabled=False,
            requested_backend=spec.backend,
            effective_backend="identity",
            degraded=False,
            reason="purification_disabled",
            guarded=False,
            model_path=spec.model_path,
        )
    backend = spec.backend
    if backend == "rule":
        return PurificationStatus(True, backend, "rule", False, None, spec.guard, spec.model_path)
    if backend == "cache":
        if _cache_available(spec.cache_file):
            return PurificationStatus(True, backend, "cache", False, None, spec.guard, spec.model_path)
        return PurificationStatus(True, backend, "rule", True, "purification_cache_missing", spec.guard, spec.model_path)
    # auto：优先查表（零成本），其次本地 LLM，最后规则
    if backend == "auto" and _cache_available(spec.cache_file):
        return PurificationStatus(True, backend, "cache", False, None, spec.guard, spec.model_path)
    if backend in _LOCAL_LLM_BACKENDS and _model_available(spec.model_path):
        return PurificationStatus(True, backend, "qwen", False, None, spec.guard, spec.model_path)
    return PurificationStatus(True, backend, "rule", True, "purification_model_missing", spec.guard, spec.model_path)


class PurifierRegistry:
    """按配置缓存净化器实例；Qwen 只装载一次。"""

    def __init__(self) -> None:
        self._instances: dict[tuple, Purifier] = {}
        self._status: dict[tuple, PurificationStatus] = {}
        self._tried: dict[tuple, str] = {}

    @staticmethod
    def _key(spec) -> tuple:
        return (spec.backend, spec.model_path, spec.batch_size, spec.max_new_tokens, bool(spec.guard), spec.cache_file)

    def get(self, spec) -> tuple[Purifier, PurificationStatus]:
        """取回（可能被缓存的）净化器及其真实状态。"""

        key = self._key(spec)
        if key in self._instances:
            return self._instances[key], self._status[key]

        purifier, status = self._build(spec)
        self._instances[key] = purifier
        self._status[key] = status
        return purifier, status

    def status(self, spec) -> PurificationStatus:
        """已构建则返回真实状态，否则返回探测结果（无副作用）。"""

        key = self._key(spec)
        return self._status.get(key) or probe(spec)

    def _build(self, spec) -> tuple[Purifier, PurificationStatus]:
        if not spec.enabled:
            return IdentityPurifier(), probe(spec)

        backend = spec.backend
        # 1) 查表（auto 也优先，因为它零成本且与离线产物一致）
        if backend in ("cache", "auto") and _cache_available(spec.cache_file):
            try:
                return self._wrap(CachedPurifier.from_file(spec.cache_file), spec), PurificationStatus(
                    True, backend, "cache", False, None, spec.guard, spec.model_path
                )
            except Exception as exc:  # noqa: BLE001 - 缓存损坏不应让服务不可用
                self._tried[self._key(spec)] = f"cache_load_failed:{type(exc).__name__}"
        # 2) 规则后端
        if backend == "rule":
            return self._wrap(RulePurifier(), spec), PurificationStatus(True, backend, "rule", False, None, spec.guard, spec.model_path)
        # 3) 本地 LLM
        if backend in _LOCAL_LLM_BACKENDS:
            try:
                from .qwen import QwenPurifier

                primary = QwenPurifier(
                    spec.model_path,
                    device=spec.device,
                    batch_size=spec.batch_size,
                    max_new_tokens=spec.max_new_tokens,
                ).load()
                return self._wrap(primary, spec), PurificationStatus(True, backend, "qwen", False, None, spec.guard, spec.model_path)
            except Exception as exc:  # noqa: BLE001 - 权重缺失/显存不足都要降级而不是崩溃
                self._tried[self._key(spec)] = f"qwen_load_failed:{type(exc).__name__}"
        # 4) 兜底：规则净化
        reason = self._tried.get(self._key(spec)) or "purification_model_missing"
        return self._wrap(RulePurifier(), spec), PurificationStatus(True, backend, "rule", True, reason, spec.guard, spec.model_path)

    @staticmethod
    def _wrap(purifier: BasePurifier, spec) -> Purifier:
        """按需套上极性护栏（护栏对任何主净化器都是纯增量安全网）。"""

        if spec.guard and not isinstance(purifier, GuardedPurifier):
            return GuardedPurifier(purifier, fallback=RulePurifier())
        return purifier

    def release(self, spec=None) -> None:
        """释放实例与显存；``spec=None`` 时释放全部。"""

        if spec is None:
            self._instances.clear()
            self._status.clear()
        else:
            key = self._key(spec)
            self._instances.pop(key, None)
            self._status.pop(key, None)
        import gc

        gc.collect()
        try:  # pragma: no cover - 无 GPU 环境下 torch 可能不存在
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 释放失败不影响正确性
            pass


__all__ = ["BACKENDS", "PurificationStatus", "PurifierRegistry", "probe"]
