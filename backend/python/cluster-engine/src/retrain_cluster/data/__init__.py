"""数据层：准入校验、历史数据集读取、文本规范化与分类语料解析。

这些模块只依赖 numpy 与标准库，不触碰模型/网络，因此可以被最底层的
``types``/``config`` 安全引用而不引入重依赖。包级符号在这里显式再导出，
让调用方有唯一的导入入口（``from retrain_cluster.data import validate_items``）。
"""

from __future__ import annotations

from .category_corpus import (
    LEVEL1,
    PLACEHOLDER_LEAVES,
    build_vocabulary,
    describe_corpus,
    load_category_corpus,
    load_field_corpus,
    looks_like_category,
    parse_category_table,
    split_of,
)
from .loaders import load_knowledge_texts, load_labels, load_reference
from .normalization import (
    NORMALIZATION_VERSION,
    Corpus,
    InvalidEntry,
    Occurrence,
    UniqueEntry,
    build_corpus,
    expand_labels,
)
from .validation import validate_items, validate_matrix

__all__ = [
    "LEVEL1",
    "NORMALIZATION_VERSION",
    "PLACEHOLDER_LEAVES",
    "Corpus",
    "InvalidEntry",
    "Occurrence",
    "UniqueEntry",
    "build_corpus",
    "build_vocabulary",
    "describe_corpus",
    "expand_labels",
    "load_category_corpus",
    "load_field_corpus",
    "load_knowledge_texts",
    "load_labels",
    "load_reference",
    "looks_like_category",
    "parse_category_table",
    "split_of",
    "validate_items",
    "validate_matrix",
]
