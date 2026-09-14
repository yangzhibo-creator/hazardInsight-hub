"""文本级向量缓存（semantic-v1 路径）。

与 legacy 的**批级**缓存（``artifacts.cache``）并存、互不影响，原因很直接：

* 批级键是"整批文本 + 模型"，只要批次里改一条，整批缓存作废；
  10 万条语料新增 1 万条时要重算全部，代价不可接受；
* semantic-v1 需要的是**按条**复用：同一个文本在不同批次、不同运行里
  都应有同一个键，因此键由"规范化文本 + 预处理版本 + 模型指纹 + 运行环境指纹"
  四元组决定。

运行环境指纹包含设备/精度/依赖版本：CPU 与 GPU 的浮点结果不保证逐位一致，
混用会让"这个键到底对应哪个向量"无法解释，因此宁可不命中也不要错命中。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .fingerprints import array_hash, fingerprint
from ..errors import ClusterError

__all__ = ["TextEmbeddingCache", "build_key"]

#: 缓存 schema；序列化口径变化时递增。
_CACHE_SCHEMA = "text-embedding-cache-v1"


def build_key(normalized_text, preprocessing_version, model_fingerprint, runtime_fingerprint) -> str:
    """文本级缓存键的规范定义。

    四项分别回答：算的是哪条文本、用什么预处理口径、用哪个模型、在什么环境下。
    任何一项变化都必须让键变化，否则就会出现"换了模型却复用旧向量"的静默错误。
    """

    return fingerprint(
        {
            "schema": _CACHE_SCHEMA,
            "text": "" if normalized_text is None else str(normalized_text),
            "preprocessing": str(preprocessing_version),
            "model": str(model_fingerprint),
            "runtime": str(runtime_fingerprint),
        }
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class TextEmbeddingCache:
    """按文本键存放单条向量的目录缓存。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    # ------------------------------------------------------------------ 读

    def _load_one(self, key: str):
        meta_path = self.directory / f"{key}.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            file_name = meta["file"]
            expected = meta.get("sha256")
        except (KeyError, TypeError, ValueError, OSError):
            return None
        array_path = self.directory / file_name
        if not array_path.is_file():
            return None
        try:
            values = np.load(array_path, allow_pickle=False)
        except (ValueError, OSError):
            raise ClusterError("CACHE_INTEGRITY", "Text embedding cache integrity check failed", 500) from None
        if expected is not None and array_hash(values) != expected:
            raise ClusterError("CACHE_INTEGRITY", "Text embedding cache integrity check failed", 500)
        return values

    def lookup(self, keys):
        """批量查询：返回 ``(命中字典, 未命中键列表)``。

        命中率由这里的查询结果直接决定，调用方不会在后续步骤里二次推导，
        因此上报的 hit/miss 与真实读盘行为一致。
        """

        hits: dict = {}
        misses: list = []
        for key in keys:
            values = self._load_one(key)
            if values is None:
                misses.append(key)
            else:
                hits[key] = values
        return hits, misses

    # ------------------------------------------------------------------ 写

    def commit(self, mapping, dimension=None) -> None:
        """提交一批 ``{key: 向量}``；写入前按需校验维度。"""

        for key, values in mapping.items():
            array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
            if dimension is not None and (array.ndim < 1 or array.shape[-1] != int(dimension)):
                raise ClusterError("INVALID_EMBEDDINGS", "Text embedding dimension does not match the model", 500)
            file_name = f"{key}.npy"
            _write_atomic(self.directory / file_name, _npy_bytes(array))
            meta = {
                "schema": _CACHE_SCHEMA,
                "key": key,
                "file": file_name,
                "sha256": array_hash(array),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
            _write_atomic(self.directory / f"{key}.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
