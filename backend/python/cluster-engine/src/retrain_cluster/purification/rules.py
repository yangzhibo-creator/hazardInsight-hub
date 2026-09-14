"""规则净化：不依赖任何模型的正则兜底（最后一道防线）。

它的定位**不是**复现论文的净化质量，而是保证两件事：

1. **绝不反转语义**：只做删除与首尾裁剪，从不改写词句；
2. **绝不中断演示**：没有 torch / 没有显存 / 权重缺失时也能把流程跑完。

正因为"只删不改"，它才可以被安全地用作极性护栏的兜底实现。
"""

from __future__ import annotations

import re
from typing import Sequence

from .base import BasePurifier

#: 行首日期：``2025.11.03`` / ``2025-11-3`` / ``2025年11月3日`` / ``20250810``。
_RE_DATE_HEAD = re.compile(
    r"^\s*(?:\d{4}\s*[.\-年/]\s*\d{1,2}\s*[.\-月/]\s*\d{1,2}\s*日?|\d{8}|\d{4}\s*年)"
    r"\s*[，,、:：]?\s*"
)
#: 句尾处置性尾巴："已通知班组打磨处理"、"已要求更换二维码标签" 等。
_RE_TRAILING_DISPOSITION = re.compile(
    r"[；;，,。.、]?\s*(?:已|现)?\s*(?:通知|要求|责令|督促|安排|联系)[^。；;]{0,40}?"
    r"(?:整改|处理|清理|打磨|补做|完善|核实|回复|更换|更新|修正|重新)[^。；;]{0,10}[。.；;]?\s*$"
)
#: 句尾完成态："已处理"、"已整改完成"、"已闭环"。
_RE_TRAILING_DONE = re.compile(r"[；;，,。.、]?\s*(?:已处理(?:完成|完毕)?|已整改(?:完成)?|已完成|已闭环)\s*[。.]?\s*$")
#: 句首过程性引导语："在B9库检查过程中发现"、"巡检发现" 等。
_RE_LEAD_PROCESS = re.compile(
    r"^\s*(?:在)?[^，,。；;]{0,24}?"
    r"(?:巡检|检查|巡查|验收|审核|查看|抽查|复核|测读|核查)(?:过程中|时|中)?"
    r"(?:发现|发现:|发现，|有|存在)?\s*[，,、:：]?\s*"
)
#: 引号/书名号包起来的规范条文整段引用。
_RE_QUOTED_SPEC = re.compile(r"[“\"《][^”\"》]{6,}[”\"》]")
#: 公司名/项目部。
_RE_COMPANY = re.compile(r"(?:中交[一二三四五六七八九十]?航局?|中铁[一二三四五六七八九十]?局|项目部|有限公司|分公司)")
#: 工程/设备/材料编号：必须含数字（如 5BFX20ST0073 / BSB2818VB / 1C16 / ER308L）。
#: 刻意不剥离纯字母缩写（如 BTA），与预置知识库的风格保持一致。
#: 用 ``(?<![A-Za-z0-9])`` 而不是 ``\b``——中文字符在 Python re 中算 word 字符，``\b`` 会失效。
_RE_CODE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[0-9]+[A-Z][A-Z0-9\-]*|[A-Z]{2,}[0-9][A-Z0-9\-]*)"
    r"(?![A-Za-z0-9])"
)
#: 行首部位/段落编号（如 "1307 墙施工缝…" 里的 1307）。
_RE_LEAD_SEGMENT = re.compile(r"^\s*[0-9]+(?:[.\-][0-9]+)*\s*(?=[\u4e00-\u9fa5])")
#: 括号里的编号/图号/报告号/批号。
_RE_PAREN_CODE = re.compile(r"[（(]\s*(?:编号|图号|报告号|批号)\s*[:：]?[^)）]{0,30}[)）]")
_RE_WS = re.compile(r"\s+")


class RulePurifier(BasePurifier):
    """正则净化：能力有限，但确定、离线、可解释。"""

    name = "rule"

    def purify(self, texts: Sequence[str]) -> list[str]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> str:
        # 先去掉所有空白（含换行）：工程文本里的换行不是语义边界
        value = _RE_WS.sub("", str(text).replace("\n", ""))
        value = _RE_DATE_HEAD.sub("", value)
        value = _RE_LEAD_SEGMENT.sub("", value)
        value = _RE_QUOTED_SPEC.sub("", value)
        value = _RE_PAREN_CODE.sub("", value)
        value = _RE_COMPANY.sub("", value)
        value = _RE_LEAD_PROCESS.sub("", value)
        # 处置性尾巴可能层层嵌套（"已通知…并已处理"），最多迭代 3 次到不动点
        for _ in range(3):
            before = value
            value = _RE_TRAILING_DISPOSITION.sub("", value)
            value = _RE_TRAILING_DONE.sub("", value)
            if value == before:
                break
        value = _RE_CODE.sub("", value)
        value = re.sub(r"^[，,、。；;:：\-\s]+", "", value)
        value = re.sub(r"[，,、。；;:：\-\s]+$", "", value)
        value = _RE_WS.sub("", value)
        # 兜底：规则把整条删空时，宁可退回截断原文，也不能返回空串
        # （空文本会让后续嵌入与聚类失去意义）
        return value or _RE_WS.sub("", str(text))[:20]


__all__ = ["RulePurifier"]
