"""产物层：向量缓存、运行产物与内容指纹。

这里用惰性导出（PEP 562）而不是在 ``__init__`` 顶部写一堆 ``import``，
原因是导入图存在一条天然的回边：``artifacts.cache`` 需要 ``data.validation``，
而 ``data.loaders`` 又需要 ``..types``，``types`` 反过来要 ``artifacts.fingerprints``。
如果 ``artifacts/__init__`` 在导入 ``fingerprints`` 时顺带把 ``cache`` 拉进来，
就会在 ``types`` 尚未初始化完成时回到 ``data``，形成循环导入。

惰性导出让"先导入哪个子模块"不再重要：包级符号在真正被访问时才解析。
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EmbeddingCache",
    "RunStore",
    "SemanticRunStore",
    "TextEmbeddingCache",
    "array_hash",
    "atomic_json",
    "build_key",
    "file_hash",
    "fingerprint",
    "jsonable",
]

#: 包级符号 -> 定义它的子模块。
_EXPORTS = {
    "fingerprint": ".fingerprints",
    "file_hash": ".fingerprints",
    "array_hash": ".fingerprints",
    "jsonable": ".fingerprints",
    "EmbeddingCache": ".cache",
    "RunStore": ".runs",
    "atomic_json": ".runs",
    "TextEmbeddingCache": ".text_cache",
    "build_key": ".text_cache",
    "SemanticRunStore": ".semantic_runs",
}


def __getattr__(name: str):
    """按需解析包级符号；未知名字保持标准的 ``AttributeError`` 语义。"""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # 解析一次后缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(__all__)
