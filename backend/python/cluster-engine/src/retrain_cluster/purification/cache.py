"""预计算净化映射（把 2 万条的净化耗时压到 0）。

现场 70 分钟预算里，Qwen 逐条净化 2 万条约需 1.7–2.8 小时，不可接受。
因此演示用的是"离线净化一次、现场查表"：``purify_map.json`` 是
``原文 -> 浓缩文本`` 的确定性映射，未命中的少量文本才交给兜底净化器。

一个容易踩的坑：**查表结果同样要过极性护栏**。映射文件可能是更早、
更弱的净化器产出的，不能因为"它是预计算的"就免检——护栏加在
:class:`~retrain_cluster.purification.guard.GuardedPurifier` 上，与后端无关。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from .base import BasePurifier, Purifier
from .rules import RulePurifier

log = logging.getLogger("retrain_cluster.purification.cache")


class CachedPurifier(BasePurifier):
    """查表净化；未命中交给 ``fallback``（默认规则净化）并回填内存映射。"""

    name = "cache"

    def __init__(self, mapping: Mapping[str, str] | None = None, fallback: Purifier | None = None) -> None:
        self.mapping: dict[str, str] = dict(mapping or {})
        self.fallback = fallback or RulePurifier()
        #: 最近一批的未命中条数，供结果如实报告"降级使用了多少规则兜底"。
        self.last_miss_count = 0

    @classmethod
    def from_file(cls, path: str | Path, fallback: Purifier | None = None) -> "CachedPurifier":
        """从 JSON 文件加载映射；文件缺失或损坏时抛出，由 registry 决定是否降级。"""

        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("purification cache must be a JSON object of raw->condensed strings")
        log.info("loaded purification cache: %s (%d entries)", target.name, len(payload))
        return cls({str(key): str(value) for key, value in payload.items()}, fallback)

    def purify(self, texts: Sequence[str]) -> list[str]:
        sources = [str(text) for text in texts]
        missing = [text for text in sources if text not in self.mapping]
        self.last_miss_count = len(missing)
        if missing:
            log.info("purification cache miss %d/%d, delegating to %s", len(missing), len(sources), self.fallback.name)
            filled = self.fallback.purify(missing)
            for text, condensed in zip(missing, filled):
                self.mapping[text] = condensed
        # 空串不是合法净化结果（会让嵌入失去意义），退回原文
        return [self.mapping.get(text) or text for text in sources]

    def save(self, path: str | Path) -> None:
        """把（含回填后的）映射写回磁盘，供下一次运行复现。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.mapping, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["CachedPurifier"]
