"""语义净化的统一接口（SPEAR 阶段 1）。

净化器的职责边界必须写死在接口上：**输入一段原始偏差文本，输出只保留
"什么对象 + 出现了什么缺陷/偏差"的浓缩文本**。它不判断缺陷类别，也不产生
任何新的事实——因此它的输出可以被逐条对照、被正则兜底替换、被护栏校验。

为什么单独抽一层而不是直接调 Qwen：

* 现场演示需要"Qwen 不可用也能跑完"，所以必须有同构的规则兜底实现；
* 极性护栏是一个装饰器（``GuardedPurifier``），它只依赖这个接口；
* 预计算映射（``CachedPurifier``）同样只依赖这个接口。

三者可以任意组合，正因为它们都只实现这一个方法。
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Purifier(Protocol):
    """净化器契约：批量进、批量出，顺序与长度必须严格对应。"""

    #: 后端标识，会写进结果与健康检查（``qwen`` / ``rule`` / ``cache`` / ``identity``）。
    name: str

    def purify(self, texts: Sequence[str]) -> list[str]:
        """净化一批文本，返回与输入等长的列表。"""

    def purify_one(self, text: str) -> str:
        """单条净化的便捷方法。"""


class BasePurifier:
    """提供 ``purify_one`` 与默认 ``batch_size`` 的公共实现。

    Protocol 只约束"长什么样"，这里提供一份可复用的默认行为，避免三个实现
    各写一遍 ``purify_one`` 而在边界上产生差异（例如是否 str() 强转）。
    """

    name = "base"
    #: 批量推理的默认批大小；Qwen 侧可覆盖。
    batch_size = 32

    def purify(self, texts: Sequence[str]) -> list[str]:  # pragma: no cover - 抽象
        raise NotImplementedError

    def purify_one(self, text: str) -> str:
        return self.purify([text])[0]


class IdentityPurifier(BasePurifier):
    """原样返回（净化关闭时的空实现）。

    刻意不叫 ``NullPurifier``：它不是"没有净化器"，而是一个**明确的恒等变换**，
    名字要能说明"这一步被执行了，只是什么都没改"。
    """

    name = "identity"

    def purify(self, texts: Sequence[str]) -> list[str]:
        return [str(text) for text in texts]


__all__ = ["BasePurifier", "IdentityPurifier", "Purifier"]
