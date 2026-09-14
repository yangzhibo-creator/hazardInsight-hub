"""极性保真护栏（SPEAR 阶段 1 的强制性校验，对应急救陷阱 P1）。

为什么必须有这一层：缺陷语义的核心是**极性**——"不一致 / 未清理 / 不符合 /
超出 / 漏绑"。一旦净化把否定词丢掉，缺陷就会被读成"正常"。实测中
Qwen3.8-27B 在 ``焊机二维码上名称与系统不一致`` 这类样本上会复现该错误，
且提示词层面无法根治。

因此这里用一条**确定性**规则兜底：源文本含极性/缺陷标记时，净化结果也必须
至少含一个；未通过者整条改用 ``fallback``（默认正则净化）。正则净化只做删除、
不做改写，按构造不可能反转语义——以"略微更啰嗦"换取"绝不出现语义反转"。
"""

from __future__ import annotations

import logging
from typing import Sequence

from .base import BasePurifier, Purifier
from .rules import RulePurifier

log = logging.getLogger("retrain_cluster.purification.guard")

#: 缺陷/极性标记。命中任一即认为该条文本"带有缺陷含义"。
#: 刻意保持与参考实现一致：新增标记会让更多样本被判为"必须保留极性"，
#: 从而更频繁地走规则兜底（更保守，但会牺牲一点净化质量）。
POLARITY_MARKERS: tuple[str, ...] = (
    "不", "未", "无", "非", "缺", "漏", "超", "误", "错", "坏", "损", "变形",
    "失效", "松动", "偏", "倾斜", "堵", "裂", "锈", "脏", "乱", "少", "多",
)


def polarity_preserved(source: str, purified: str) -> bool:
    """源文本含极性/缺陷标记时，净化结果也必须含至少一个。

    源文本本来就没有极性词时视为通过——此时"净化结果没有极性词"是正确的。
    """

    if any(marker in source for marker in POLARITY_MARKERS):
        return any(marker in purified for marker in POLARITY_MARKERS)
    return True


class GuardedPurifier(BasePurifier):
    """带极性护栏的净化器：主净化器输出逐条校验，不合格的走兜底。

    ``last_guard_count`` / ``last_total`` 记录最近一批的拦截比例，会写进结果与
    日志，便于现场说明"护栏真的在工作"（裸 Qwen 1/5 失误 → 加护栏 0/5）。
    """

    name = "guarded"

    def __init__(self, primary: Purifier, fallback: Purifier | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or RulePurifier()
        self.last_guard_count = 0
        self.last_total = 0

    @property
    def batch_size(self) -> int:
        # 批量大小透传给主净化器；护栏本身是逐条的，不引入额外批处理语义。
        return int(getattr(self.primary, "batch_size", 32))

    def purify(self, texts: Sequence[str]) -> list[str]:
        sources = [str(text) for text in texts]
        outputs = list(self.primary.purify(sources))
        if len(outputs) != len(sources):
            # 长度不一致会让"第 i 条结果对应第 i 条输入"这一前提失效，
            # 与其静默错位，不如整批退回规则净化（仍然保证不反转语义）。
            log.warning("主净化器返回 %d 条，输入 %d 条，整批改用 %s", len(outputs), len(sources), self.fallback.name)
            outputs = list(self.fallback.purify(sources))
            self.last_guard_count = len(sources)
            self.last_total = len(sources)
            return outputs

        bad = [index for index, (source, output) in enumerate(zip(sources, outputs)) if not polarity_preserved(source, output)]
        self.last_guard_count = len(bad)
        self.last_total = len(sources)
        if bad:
            log.warning(
                "极性保真校验拦下 %d/%d 条（主净化器 %s），改用 %s 兜底",
                len(bad), len(sources), self.primary.name, self.fallback.name,
            )
            fixed = self.fallback.purify([sources[index] for index in bad])
            for index, fallback_text in zip(bad, fixed):
                outputs[index] = fallback_text
        return outputs


__all__ = ["GuardedPurifier", "POLARITY_MARKERS", "polarity_preserved"]
