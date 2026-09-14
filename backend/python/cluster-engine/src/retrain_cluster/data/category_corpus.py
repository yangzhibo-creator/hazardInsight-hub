"""分类语料的解析与词表构建。

``datas/rules/规则.txt`` 每行是"标记 + 若干级分类 + 隐患描述"，格式并不严格：
分类级数从 1 到 4 不等，描述里可能含空格、括号与数字，还混着两种**不该入库**的行：

* 占位叶子行：整行只有分类路径，末级是 ``其他``/``其它``——它不是描述，
  若当成描述会产出上百条同文本、不同标签的样本，直接污染聚类；
* 分类定义行：末段是用逗号连接分类路径，用来声明分类层次，本身没有语义。

分类级数无法只看一行确定，因此先统计"位置 2/3 上反复出现的 token"得到词表，
再用 ``looks_like_category`` 对只出现一次的稀有分类做**漏词修正**——只出现一次
的分类名进不了词表，但确实是分类位，必须在描述前把它退回分类。数字量值
（``15m``）虽然短而无句读，却因以数字开头被排除，留在描述里。

切割必须可复现，因此切分由 id 的哈希决定，并保证两侧都有样本。
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path

from ..artifacts.fingerprints import file_hash

__all__ = [
    "LEVEL1",
    "PLACEHOLDER_LEAVES",
    "build_vocabulary",
    "describe_corpus",
    "load_category_corpus",
    "load_field_corpus",
    "looks_like_category",
    "parse_category_table",
    "split_of",
]

#: 一级分类白名单（与规则文件一致）。一级分类是封闭集合，不参与词表统计。
LEVEL1 = (
    "基础管理",
    "现场作业",
    "现场施工机械、器具",
    "现场安全与应急设施",
    "现场环境和条件",
    "人员行为（非作业过程）",
)

#: 占位叶子：出现即代表"该分类下没有具体描述"，不能作为文本入库。
PLACEHOLDER_LEAVES = frozenset({"其他", "其它", "其它项", "其他项"})

#: 分类名的最大字符数；超过它基本可以断定是描述而不是分类。
MAX_CATEGORY_CHARS = 12

#: 最大分类级数（与真实数据的 1~4 级一致）。
MAX_DEPTH = 4

#: 分类名里不应出现的标点（出现即视为描述）。
_CATEGORY_FORBIDDEN = frozenset("，。；：、！？（）()[]{},.;:!?\"'“”‘’·-—_/\\|<>@#$%^&*+=~`～…—")

#: 词表默认最小重复次数。
DEFAULT_VOCAB_MINIMUM = 2

#: 切分比例：约 30% 进 holdout，其余进 calibration。
_HOLDOUT_RATIO = 0.3


def _is_marker(token: str) -> bool:
    """判断行首标记（``a``/``b``/``c``/``d``）——单个 ASCII 字母。"""

    return len(token) == 1 and token.isascii() and token.isalpha()


def build_vocabulary(tokenized, minimum: int = DEFAULT_VOCAB_MINIMUM) -> dict:
    """统计位置 2/3 上重复出现的 token。

    位置 1 是一级分类（封闭集合），因此被刻意忽略；重复次数不足的 token
    不进词表——它可能是描述里的偶然词，要靠 ``looks_like_category`` 单独裁决。
    """

    counts = {2: Counter(), 3: Counter()}
    for tokens in tokenized:
        for position in (2, 3):
            if position < len(tokens):
                counts[position][tokens[position]] += 1
    return {
        position: {token for token, count in counts[position].items() if count >= int(minimum)}
        for position in (2, 3)
    }


def looks_like_category(token) -> bool:
    """启发式判断一个 token 是否像分类名。

    规则：非空、不以数字开头、长度不超过 12、且不含句读/标点。
    ``15m``（数字开头）会被排除，长句子（含标点或超长）同样被排除。
    """

    if not isinstance(token, str):
        return False
    candidate = token.strip()
    if not candidate:
        return False
    if candidate[0].isdigit():
        return False
    if len(candidate) > MAX_CATEGORY_CHARS:
        return False
    return not any(char in _CATEGORY_FORBIDDEN for char in candidate)


def _looks_like_definition(text: str) -> bool:
    """末段是逗号连接的分类路径 => 分类定义行。"""

    if "," not in text:
        return False
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        return False
    return all(looks_like_category(part) for part in parts)


def parse_category_table(lines, minimum: int = DEFAULT_VOCAB_MINIMUM) -> tuple[list[dict], Counter]:
    """把规则行解析成 ``(records, dropped)``。

    ``records`` 每项含 ``categories``（分类列表）与 ``text``（描述）；
    ``dropped`` 记录被丢弃的原因计数，便于核对解析没有悄悄吃掉数据。
    """

    tokenized = [str(line).split() for line in lines]
    vocabulary = build_vocabulary(tokenized, minimum=minimum)
    records: list[dict] = []
    dropped: Counter = Counter()

    for tokens in tokenized:
        if not tokens:
            continue
        has_marker = _is_marker(tokens[0])
        body = tokens[1:] if has_marker else tokens
        if not body:
            continue
        offset = 1 if has_marker else 0

        categories = [body[0]]
        position = 1
        # 至少留一个 token 作为描述：因此循环止于 len(body)-1
        while position < len(body) - 1 and len(categories) < MAX_DEPTH:
            token = body[position]
            known = token in vocabulary.get(position + offset, ())
            if known or looks_like_category(token):
                categories.append(token)
                position += 1
            else:
                break

        text = " ".join(body[position:]).strip()
        if not text:
            dropped["空行"] += 1
            continue
        if text in PLACEHOLDER_LEAVES:
            dropped["占位叶子"] += 1
            continue
        if _looks_like_definition(text):
            dropped["分类定义行"] += 1
            continue
        records.append({"categories": categories, "text": text})

    return records, dropped


def split_of(ident) -> str:
    """由标识哈希决定 calibration / holdout，保证可复现且两侧都有样本。"""

    digest = hashlib.sha256(f"category-split-v1:{ident}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "holdout" if value < _HOLDOUT_RATIO else "calibration"


def _labels_for(categories: list[str]) -> dict:
    """按分类深度生成标签；深度不足时降级为 None，而不是编造。"""

    return {
        "l1": categories[0],
        "l2": "/".join(categories[:2]),
        "l3": "/".join(categories[:3]) if len(categories) >= 3 else None,
    }


def _corpus(name, items, *, label_field="l2", source=None, dropped=None) -> dict:
    corpus = {
        "name": name,
        "label_field": label_field,
        "label_fields": ["l1", "l2", "l3"],
        "items": items,
        "dropped": dict(dropped or {}),
    }
    if source is not None:
        corpus["source"] = str(source)
        try:
            corpus["source_fingerprint"] = file_hash(source)
        except OSError:
            corpus["source_fingerprint"] = None
    return corpus


def load_category_corpus(path) -> dict:
    """读取规则文件并构建带真值标签的语料。"""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records, dropped = parse_category_table(lines)
    items = []
    for index, record in enumerate(records):
        ident = f"R{index:05d}"
        categories = record["categories"]
        items.append(
            {
                "id": ident,
                "text": record["text"],
                "labels": _labels_for(categories),
                "depth": len(categories),
                "split": split_of(ident),
            }
        )
    return _corpus("rule-categories", items, source=path, dropped=dropped)


def load_field_corpus(path, *, text_column="隐患描述", label_column="隐患分类", id_column="隐患单号") -> dict:
    """读取现场抽样 CSV（仅用一级/二级分类作为粗粒度真值）。

    保留这个入口是为了让校准脚本能复核"换到真实长文本上没退化"。它不参与
    阈值扫描，因此这里只做最小可用的字段映射。
    """

    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines()))
    items = []
    for index, row in enumerate(rows):
        text = str(row.get(text_column) or "").strip()
        if not text:
            continue
        raw_label = str(row.get(label_column) or "").strip()
        categories = [part for part in raw_label.split("/") if part] or ["未分类"]
        ident = str(row.get(id_column) or f"F{index:05d}")
        items.append(
            {
                "id": ident,
                "text": text,
                "labels": _labels_for(categories),
                "depth": len(categories),
                "split": split_of(ident),
            }
        )
    return _corpus("field-samples", items, source=path)


def describe_corpus(corpus: dict) -> str:
    """给出一句可读的语料摘要（供构建脚本打印）。"""

    field = corpus.get("label_field") or "l2"
    distribution = Counter(
        item.get("labels", {}).get(field) for item in corpus.get("items", []) if item.get("labels", {}).get(field)
    )
    splits = Counter(item.get("split") for item in corpus.get("items", []))
    return (
        f"{corpus.get('name')}: {len(corpus.get('items', []))} 条；"
        f"{field} 主题 {len(distribution)} 个；"
        f"calibration={splits.get('calibration', 0)} holdout={splits.get('holdout', 0)}"
    )
