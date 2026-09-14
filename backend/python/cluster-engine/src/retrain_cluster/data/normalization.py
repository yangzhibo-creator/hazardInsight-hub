"""文本规范化、无效识别与精确去重。

semantic-v1 的第一步不是"编码"，而是把原始请求整理成**唯一有效文本**的集合。
这一步决定了三件事，因此必须独立成层、且行为版本化：

* **无效文本要单独成桶**：空白与纯标点无法表达主题，把它们当噪声会让
  "未归类"里混进两类完全不同的东西，前端无法解释。这里显式识别为 invalid；
* **精确去重**：同一句话重复上千次会让质心被重复样本拖走。相同文本只参与
  一次聚类，但展开回结果时每一份都必须独立可回溯（用 occurrence 记原始位置）；
* **确定性**：唯一文本按内容键排序，而不是按输入顺序。否则打乱输入就会
  改变"哪条文本落到哪个簇"，可复现性无从谈起。

``NORMALIZATION_VERSION`` 随产物落盘；改规则必须改版本，否则新旧结论不可比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from ..artifacts.fingerprints import fingerprint

__all__ = [
    "NORMALIZATION_VERSION",
    "Corpus",
    "InvalidEntry",
    "Occurrence",
    "UniqueEntry",
    "build_corpus",
    "expand_labels",
]

#: 规范化口径版本；任何影响 embedding_text 的改动都必须递增。
NORMALIZATION_VERSION = "zh-shorttext-v1"

#: 判定"纯标点"的字符集合（中英文句子标点与常见符号）。
_PUNCTUATION = frozenset(
    "，。；：、！？（）()[]{},.;:!?\"'“”‘’·-—_/\\|<>@#$%^&*+=~`～…—"
)


@dataclass
class Occurrence:
    """唯一文本的一次原始出现。"""

    index: int  # 在请求 items 中的位置，用于展开回全量结果
    sample_id: str
    metadata: dict


@dataclass
class UniqueEntry:
    """一条参与聚类的唯一文本。"""

    key: str  # 内容键：稳定、可排序、可作为缓存前缀
    text: str  # 规范化后的文本（对外展示与摘要使用）
    embedding_text: str  # 实际送入编码器的文本（可能被 max_chars 截断）
    primary_id: str  # 首次出现的 sample_id，代表作与展示点用它
    frequency: int
    occurrences: list[Occurrence] = field(default_factory=list)


@dataclass
class InvalidEntry:
    """被识别为无效的原始输入（空白/纯标点）。"""

    index: int
    sample_id: str
    text: str
    reason: str


@dataclass
class Corpus:
    """一次请求规范化后的完整视图。"""

    total: int
    unique: list[UniqueEntry]
    invalid: list[InvalidEntry]
    long_text_count: int
    duplicate_count: int

    @property
    def unique_count(self) -> int:
        return len(self.unique)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid)

    def sample_key_fingerprint(self) -> str:
        """唯一文本键集合的指纹，写进产物便于比对两次输入是否同源。"""

        return fingerprint([entry.key for entry in self.unique])


def normalize_text(text) -> str:
    """折叠空白并去掉首尾空白；非字符串一律视为空。"""

    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def _is_punctuation_only(text: str) -> bool:
    return bool(text) and all(char in _PUNCTUATION or char.isspace() for char in text)


def build_corpus(items, max_chars: int = 4000, long_text_chars: int = 512) -> Corpus:
    """把请求样本整理成 ``Corpus``（唯一有效文本 + 无效项 + 统计）。"""

    unique: dict[str, UniqueEntry] = {}
    invalid: list[InvalidEntry] = []
    long_count = 0
    total = len(items)

    for index, item in enumerate(items):
        raw_text = item.get("text") if isinstance(item, dict) else None
        sample_id = str(item.get("id")) if isinstance(item, dict) and item.get("id") is not None else f"row-{index + 1}"
        metadata = dict(item.get("metadata") or {}) if isinstance(item, dict) else {}
        normalized = normalize_text(raw_text)
        if not normalized:
            invalid.append(InvalidEntry(index=index, sample_id=sample_id, text="", reason="blank"))
            continue
        if _is_punctuation_only(normalized):
            invalid.append(
                InvalidEntry(index=index, sample_id=sample_id, text=normalized, reason="punctuation_only")
            )
            continue

        embedding_text = normalized
        if max_chars and len(embedding_text) > int(max_chars):
            embedding_text = embedding_text[: int(max_chars)]
            normalized = embedding_text
        if long_text_chars and len(embedding_text) > int(long_text_chars):
            long_count += 1

        key = hashlib.sha256(f"{NORMALIZATION_VERSION}\x1f{embedding_text}".encode("utf-8")).hexdigest()
        entry = unique.get(key)
        if entry is None:
            unique[key] = UniqueEntry(
                key=key,
                text=normalized,
                embedding_text=embedding_text,
                primary_id=sample_id,
                frequency=1,
                occurrences=[Occurrence(index=index, sample_id=sample_id, metadata=metadata)],
            )
        else:
            entry.frequency += 1
            entry.occurrences.append(Occurrence(index=index, sample_id=sample_id, metadata=metadata))

    # 按内容键排序，保证与输入顺序无关的确定性
    ordered = [unique[key] for key in sorted(unique)]
    duplicate_count = total - len(ordered) - len(invalid)
    return Corpus(
        total=total,
        unique=ordered,
        invalid=invalid,
        long_text_count=long_count,
        duplicate_count=duplicate_count,
    )


def expand_labels(corpus: Corpus, unique_labels):
    """把唯一文本上的标签展开回全部原始样本；无效项固定为 ``-1``。"""

    labels = np.full(int(corpus.total), -1, dtype=np.int32)
    for unique_index, entry in enumerate(corpus.unique):
        label = int(unique_labels[unique_index])
        for occurrence in entry.occurrences:
            labels[occurrence.index] = label
    return labels
