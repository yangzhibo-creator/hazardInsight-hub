"""内容指纹：把"这次算的是什么"压缩成可比较、可落盘的稳定字符串。

为什么需要这一层：

* **可复现**：产物里记录配置/模型/数据的指纹，才能回答"同样的结论是不是
  同样的输入算出来的"；
* **缓存键**：向量缓存必须以内容为键。用对象地址或 `repr` 会因进程/顺序漂移，
  用内容哈希才能跨进程复用；
* **形状敏感**：``array_hash`` 必须把 dtype 与 shape 一起纳入，否则
  ``(2, 2)`` 与 ``(1, 4)``、``float64`` 与 ``float32`` 会算出同一个值——
  缓存一旦把它们当成同一份数据，读回来就是错的。

本模块只依赖标准库与 numpy，因此可以被任何层安全导入（包括 ``types`` 这种
最底层的数据契约），不会把重量级依赖带进导入图。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["array_hash", "file_hash", "fingerprint", "jsonable"]

#: 读文件哈希时的块大小。避免把大模型权重整体读进内存。
_CHUNK = 1024 * 1024


def jsonable(obj: Any) -> Any:
    """把任意嵌套对象转换成 ``json.dumps`` 能直接吃下的纯 Python 结构。

    numpy 标量与数组是这里唯一真正需要处理的类型：结果对象里到处是
    ``int64`` / ``float32`` / ``ndarray``，它们不是 JSON 原生类型。
    其它无法识别的对象交给 ``default=str`` 兜底，保证"记录产物"这件事
    永远不会因为一个冷门类型而彻底失败。
    """

    if isinstance(obj, np.ndarray):
        return [jsonable(item) for item in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        # 集合本身无序，先按稳定键排序，保证同一集合每次得到相同的列表
        return sorted((jsonable(item) for item in obj), key=lambda item: repr(item))
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return jsonable(asdict(obj))
    return obj


def fingerprint(obj: Any) -> str:
    """对象的规范 JSON 指纹（sha256 十六进制）。

    规范化的关键是 ``sort_keys=True`` 加最紧凑的分隔符：同一份字典无论键的
    插入顺序如何，都得到同一个指纹。这一点对"配置指纹"尤其重要——否则
    改一次代码里的字段顺序就会让全部缓存失效。
    """

    payload = json.dumps(
        jsonable(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """文件内容的 sha256，分块读取以避免占用大内存。"""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    """数组指纹：**同时**包含 dtype、shape 与原始字节。

    只哈希字节是不够的：``(2, 2)`` 与 ``(1, 4)`` 的字节完全相同，
    ``float64`` 与 ``float32`` 也可能在截断后碰撞。把 dtype 与 shape
    写进摘要前缀，才能让"形状/精度不同 = 不同指纹"成为硬约束。
    """

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()
