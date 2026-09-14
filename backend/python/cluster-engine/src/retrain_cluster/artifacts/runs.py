"""运行产物（run）的原子落盘与读取。

一次聚类/实验的结论必须"要么完整可见、要么完全不可见"，否则审计时会把
半份产物当成结论。因此这里做了两件事：

* ``atomic_json``：先写同目录临时文件、``fsync``、再 ``os.replace``。读者
  永远看到的是替换完成的整份文件，不会读到写一半的 JSON；同时拒绝
  ``NaN`` / ``Infinity``——它们不是合法 JSON，写进去只会让下游解析失败；
* ``RunStore``：一个 run 目录里先写 ``result.json``、最后写 ``manifest.json``。
  **manifest 是提交标记**：只有它存在且 ``status=completed`` 时结果才可读。
  中途被杀只会留下没有 manifest 的目录，读取方据此判定"未完成"。

读取时用结果文件的哈希反查 manifest 里记录的 ``result_sha256``，
任何改动（哪怕只是把标签改掉）都会被抓成错误而不是静默返回错数据。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

from .fingerprints import file_hash, jsonable
from ..errors import ClusterError

__all__ = ["RunStore", "atomic_json"]

#: run_id 直接参与路径拼接，必须白名单校验，杜绝路径穿越。
_RUN_ID = re.compile(r"run_[0-9a-f]{32}")


def atomic_json(path, obj) -> None:
    """原子写 JSON：创建父目录、拒绝非有限数、不留临时文件。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False 会在遇到 NaN/Inf 时抛 ValueError；此时还没创建临时文件
    payload = json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=".pending-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)


class RunStore:
    """目录化的 run 仓库：``<root>/<run_id>/{result.json,manifest.json}``。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    # ------------------------------------------------------------------ 路径

    def _path(self, run_id: str) -> Path:
        """校验 run_id 格式；非法值不碰文件系统，直接判"未找到"。"""

        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ClusterError("RUN_NOT_FOUND", "Result not found", 404)
        return self.directory / run_id

    # ------------------------------------------------------------------ 写

    def save(self, result, manifest) -> None:
        """保存结果与清单；拒绝覆盖已存在的 run。"""

        run_id = result.get("run_id") or manifest.get("run_id")
        out = self._path(run_id)
        # exist_ok=False 是"永不覆盖"的硬保证：重跑同一 run_id 会显式失败，
        # 而不是把旧结论悄悄替换成新结论
        out.mkdir(parents=True, exist_ok=False)
        result_path = out / "result.json"
        atomic_json(result_path, result)
        committed = {
            "schema_version": 1,
            **dict(manifest),
            "run_id": run_id,
            "status": "completed",
            # 结果文件的内容哈希；读取时用它判断是否被改动
            "result_sha256": file_hash(result_path),
        }
        # manifest 最后写：它的出现即"这份产物已提交"
        atomic_json(out / "manifest.json", committed)

    # ------------------------------------------------------------------ 读

    def get(self, run_id):
        """读回结果；缺失/未完成/被篡改分别报"未找到"或"完整性失败"。"""

        path = self._path(run_id)
        if not path.is_dir():
            raise ClusterError("RUN_NOT_FOUND", "Result not found", 404)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ClusterError("RUN_NOT_FOUND", "Result not found", 404)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ClusterError("RUN_INTEGRITY", "Result integrity check failed", 500) from None
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            raise ClusterError("RUN_INTEGRITY", "Result integrity check failed", 500)
        result_path = path / "result.json"
        if not result_path.is_file() or file_hash(result_path) != manifest.get("result_sha256"):
            raise ClusterError("RUN_INTEGRITY", "Result integrity check failed", 500)
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ClusterError("RUN_INTEGRITY", "Result integrity check failed", 500) from None
