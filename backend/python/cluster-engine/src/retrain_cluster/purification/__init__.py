"""SPEAR 阶段 1：输入层语义净化。

对外只暴露四类对象：

* 契约：:class:`Purifier`（任何实现都满足它）；
* 实现：:class:`RulePurifier` / :class:`QwenPurifier` / :class:`CachedPurifier`；
* 安全网：:class:`GuardedPurifier` 与 :func:`polarity_preserved`（陷阱 P1）；
* 装配：:class:`PurifierRegistry` 与 :func:`probe`（实例复用 + 降级链 + 状态报告）。

净化**只发生在编码之前**，且只做剥离：它不改变缺陷类别，也不做任何标注。
"""

from .base import BasePurifier, IdentityPurifier, Purifier
from .cache import CachedPurifier
from .guard import GuardedPurifier, POLARITY_MARKERS, polarity_preserved
from .prompts import FEWSHOT, NOISE_SPEC, SYSTEM_PROMPT, build_prompt
from .qwen import QwenPurifier
from .registry import BACKENDS, PurificationStatus, PurifierRegistry, probe
from .rules import RulePurifier

__all__ = [
    "BACKENDS",
    "BasePurifier",
    "CachedPurifier",
    "FEWSHOT",
    "GuardedPurifier",
    "IdentityPurifier",
    "NOISE_SPEC",
    "POLARITY_MARKERS",
    "PurificationStatus",
    "Purifier",
    "PurifierRegistry",
    "QwenPurifier",
    "RulePurifier",
    "SYSTEM_PROMPT",
    "build_prompt",
    "polarity_preserved",
    "probe",
]
