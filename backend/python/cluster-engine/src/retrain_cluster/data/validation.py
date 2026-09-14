"""输入准入校验。

为什么要单独成层：聚类算法对"一行一个样本"的矩阵有硬性假设，一旦形状、
类型或取值越界，报错会发生在很深的 numpy/sklearn 内部，信息既难懂又难定位。
在边界先把非法输入收敛成带错误码的 ``ClusterError``，可以让 API/CLI 直接给出
可读的 4xx，而不是 500。

这里同时服务两条路径：

* ``validate_items``：语义/legacy 的原始文本请求；
* ``validate_matrix``：编码之后、进算法之前的向量矩阵（也用于知识库构建）。
"""

from __future__ import annotations

import numpy as np

from ..errors import ClusterError

__all__ = ["validate_items", "validate_matrix"]


def validate_matrix(matrix, rows=None, dimension=None):
    """校验并规范化一个二维数值矩阵。

    通过时返回 ``np.asarray`` 的结果（调用方可直接复用，避免重复转换）；
    失败时抛 ``ClusterError``，消息里分别点明"不是有限数值矩阵"、
    "行数不符"或"维度不符"，便于测试与运维快速定位。
    """

    array = np.asarray(matrix)
    # 数值类型用 dtype.kind 判定：'f' 浮点、'i' 有符号、'u' 无符号。
    # 字符串/对象数组必须被拒绝，否则后续算法会在更深处抛出难以理解的错误。
    if array.ndim != 2 or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ClusterError("INVALID_MATRIX", "Expected a finite numeric matrix", 422)
    if rows is not None and array.shape[0] != int(rows):
        raise ClusterError("INVALID_SAMPLE_COUNT", "Matrix row count does not match the sample count", 422)
    if dimension is not None and array.shape[1] != int(dimension):
        raise ClusterError("INVALID_DIMENSION", "Matrix dimension does not match the expected dimension", 422)
    return array


def validate_items(items, minimum=2, maximum=None):
    """校验文本样本列表：数量、ID 唯一且存在、文本非空白。

    成功返回 ``None``——本函数只负责"准入"，不负责转换数据。
    """

    count = len(items) if items is not None else 0
    if count < int(minimum):
        raise ClusterError("INVALID_SAMPLE_COUNT", f"Sample count must be at least {int(minimum)}", 422)
    if maximum is not None and count > int(maximum):
        raise ClusterError("INVALID_SAMPLE_COUNT", f"Sample count must not exceed {int(maximum)}", 422)

    identifiers = []
    for item in items:
        identifier = item.get("id") if isinstance(item, dict) else None
        # ID 是结果可回溯到原始样本的唯一凭据；缺失或重复都会让"哪条是哪条"失真
        if identifier is None or identifier == "":
            raise ClusterError("INVALID_SAMPLE_ID", "Sample IDs must be present and unique", 422)
        identifiers.append(str(identifier))
    if len(set(identifiers)) != len(identifiers):
        raise ClusterError("INVALID_SAMPLE_ID", "Sample IDs must be present and unique", 422)

    for item in items:
        text = item.get("text") if isinstance(item, dict) else None
        # 必须是真正的字符串：None、数字或纯空白都无法构成一条可聚类的文本
        if not isinstance(text, str) or not text.strip():
            raise ClusterError("INVALID_TEXT", "Blank text is not allowed", 422)
    return None
