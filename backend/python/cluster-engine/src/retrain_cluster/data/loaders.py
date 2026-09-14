"""历史数据集的读取口径（legacy 实验与 CLI 使用）。

真实输入文件不随仓库分发，但"文件长什么样、读出来是什么"必须被固定下来，
否则同一份配置在不同机器上会得到不同的样本顺序/标签，复现实验就失去意义：

* ``reference.json``：文本 -> one-hot 标签向量。字典的插入顺序就是样本顺序；
* ``labels.json``：标签名 -> 文本列表。一条文本可能出现在多个标签下，
  因此 ID 按行生成、天然唯一；
* ``database.txt``：每行一条知识文本，**保留结尾空行**——历史文件以换行
  结尾，用 ``splitlines`` 会悄悄吃掉最后一条空记录，令行数对不上。

读取时同时记录来源文件的内容指纹，产物据此可追溯"这批数据是哪一版"。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..artifacts.fingerprints import file_hash, fingerprint
from ..errors import ClusterError
from ..types import TextDataset

__all__ = ["load_knowledge_texts", "load_labels", "load_reference"]


def _read_json(path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClusterError("DATASET_NOT_FOUND", "Dataset file is missing", 404) from exc
    except ValueError as exc:
        raise ClusterError("DATASET_INVALID", "Dataset file is not valid JSON", 422) from exc


def load_reference(path) -> TextDataset:
    """读取 ``reference.json``：文本 -> one-hot 标签。"""

    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ClusterError("DATASET_INVALID", "Reference dataset must be a non-empty JSON object", 422)
    texts = list(payload.keys())
    labels = np.array([int(np.argmax(np.asarray(vector))) for vector in payload.values()], dtype=np.int32)
    source_fingerprint = file_hash(path)
    return TextDataset(
        dataset_id=fingerprint({"schema": "reference-v1", "source": source_fingerprint}),
        sample_ids=[f"reference-{index:05d}" for index in range(len(texts))],
        texts=texts,
        labels=labels,
        source_fingerprint=source_fingerprint,
    )


def load_labels(path) -> TextDataset:
    """读取 ``labels.json``：标签名 -> 文本列表，按出现顺序展平。"""

    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ClusterError("DATASET_INVALID", "Labels dataset must be a non-empty JSON object", 422)
    texts: list[str] = []
    labels: list[int] = []
    label_names: dict[int, str] = {}
    for index, (name, group) in enumerate(payload.items()):
        label_names[index] = str(name)
        for text in group or []:
            texts.append(text)
            labels.append(index)
    source_fingerprint = file_hash(path)
    return TextDataset(
        dataset_id=fingerprint({"schema": "labels-v1", "source": source_fingerprint}),
        sample_ids=[f"labels-{index:05d}" for index in range(len(texts))],
        texts=texts,
        labels=np.array(labels, dtype=np.int32),
        label_names=label_names,
        source_fingerprint=source_fingerprint,
    )


def load_knowledge_texts(path) -> list[str]:
    """读取 ``database.txt``；用 ``split("\\n")`` 保留结尾空行。"""

    try:
        return Path(path).read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        raise ClusterError("DATASET_NOT_FOUND", "Knowledge text file is missing", 404) from exc
