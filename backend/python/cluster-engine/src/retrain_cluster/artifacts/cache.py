"""批级向量缓存（legacy 路径）。

一次 legacy 聚类会按同一批文本请求同一批向量；没有缓存时，重跑、调参与
API 重试都会重复付出（可能计费的）推理成本。这里的键是"文本序列 + 模型指纹"
的规范哈希，命中即可跳过编码。

工程上必须守住两条：

* **文本边界不能丢**：``["ab", "c"]`` 与 ``["a", "bc"]`` 的拼接是同一串字符，
  但它们是两份不同的输入。因此键是对 JSON 数组取指纹，而不是对拼接后的长串取；
* **读回来的必须就是当初写下去的那份**：缓存损坏（半截文件、被改动、尺寸不符）
  时宁可当作未命中或直接报错，也不能把错向量喂给聚类——那会静默污染结果。
  因此每条缓存都带内容哈希，读取时先校验再返回。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .fingerprints import array_hash, fingerprint
from ..errors import ClusterError

__all__ = ["EmbeddingCache"]

#: 缓存 schema；改序列化方式时递增，旧文件自然不再被当成有效缓存。
_CACHE_SCHEMA = "embedding-cache-v1"


def _write_atomic(path: Path, payload: bytes) -> None:
    """先写同目录临时文件再 ``os.replace``，保证读者永远看不到半截文件。"""

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


class EmbeddingCache:
    """基于目录的批级向量缓存：``<key>.json``（元信息） + ``<key>.npy``（向量）。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    # ------------------------------------------------------------------ 键

    def key(self, texts, model_hash) -> str:
        """由文本序列与模型指纹生成稳定键。

        文本以**列表**形式进入指纹，因此 ``["ab", "c"]`` 与 ``["a", "bc"]``
        必然不同——这正是"边界敏感"的含义。
        """

        return fingerprint(
            {
                "schema": _CACHE_SCHEMA,
                "model": str(model_hash),
                "texts": ["" if text is None else str(text) for text in texts],
            }
        )

    # ------------------------------------------------------------------ 路径

    def _meta_path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _array_path(self, file_name: str) -> Path:
        return self.directory / file_name

    # ------------------------------------------------------------------ 写

    def _store(self, key: str, values: np.ndarray, *, verified: bool, origin: str | None = None) -> None:
        array = np.ascontiguousarray(np.asarray(values))
        file_name = f"{key}.npy"
        # npy 先落盘、元信息最后落盘：看到元信息才算这套缓存完整
        _write_atomic(self._array_path(file_name), _npy_bytes(array))
        meta = {
            "schema": _CACHE_SCHEMA,
            "key": key,
            "file": file_name,
            "verified": bool(verified),
            "sha256": array_hash(array),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
        if origin is not None:
            meta["origin"] = str(origin)
        _write_atomic(self._meta_path(key), json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

    def save(self, key, values) -> None:
        """保存一份经过本地计算的向量（可信）。"""

        self._store(key, values, verified=True)

    def import_legacy(self, key, path) -> None:
        """导入历史遗留的 ``.npy`` 向量，并标记为**未核验**。

        历史向量没有伴随元信息，无法证明它对应哪个模型/哪次预处理，因此
        默认读取时必须被拒绝；只有调用方显式 ``allow_unverified=True`` 才放行。
        """

        source = Path(path)
        if not source.is_file():
            raise ClusterError("CACHE_SOURCE_MISSING", "Legacy embedding file does not exist", 404)
        try:
            values = np.load(source, allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ClusterError("CACHE_INVALID", "Legacy embedding file is unreadable", 422) from exc
        self._store(key, values, verified=False, origin=str(source))

    # ------------------------------------------------------------------ 读

    def load(self, key, rows=None, dimension=None, allow_unverified=False):
        """读取缓存。

        返回 ``None`` 表示"没有可用的缓存"（缺失、半截、未核验）；
        返回数组表示命中且通过完整性校验；哈希不符则抛 ``ClusterError``，
        因为"文件在那儿但内容不对"属于损坏，不能静默当成未命中。
        """

        meta_path = self._meta_path(key)
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            file_name = meta["file"]
            expected = meta.get("sha256")
            verified = bool(meta.get("verified", False))
        except (KeyError, TypeError, ValueError, OSError):
            return None
        if not verified and not allow_unverified:
            return None
        array_path = self._array_path(file_name)
        if not array_path.is_file():
            # 元信息在、向量文件不在 => 半截缓存，按"未命中"处理
            return None
        try:
            values = np.load(array_path, allow_pickle=False)
        except (ValueError, OSError):
            # 文件在但解不开，说明内容已被破坏：这是完整性失败，不能静默当未命中
            raise ClusterError("CACHE_INTEGRITY", "Embedding cache integrity check failed", 500) from None
        if expected is not None and array_hash(values) != expected:
            raise ClusterError("CACHE_INTEGRITY", "Embedding cache integrity check failed", 500)
        if rows is not None and values.shape[0] != rows:
            raise ClusterError("CACHE_INTEGRITY", "Embedding cache integrity check failed: row count mismatch", 500)
        if dimension is not None and (values.ndim != 2 or values.shape[1] != dimension):
            raise ClusterError("CACHE_INTEGRITY", "Embedding cache integrity check failed: dimension mismatch", 500)
        return values


def _npy_bytes(array: np.ndarray) -> bytes:
    """把数组序列化成 npy 字节流（不落临时文件，便于原子写）。"""

    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()
