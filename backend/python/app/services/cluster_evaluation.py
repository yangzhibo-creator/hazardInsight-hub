"""聚类外部指标的网关侧评估。

为什么放在网关而不是引擎：这 6 个指标（ARI / VM / FMS / AMI / HS / CS）需要
**真实类别标签**，而标签来自调用方数据集的某一列（例如 ``category``），与聚类
算法本身无关。把标签塞进 cluster-engine 的请求契约会污染"聚类"这件事的输入；
网关手里已经有逐条 metadata 与引擎返回的逐条 cluster_id，在同一位置对齐两边
即可，引擎契约保持不变。

口径与历史实验完全一致：直接复用 cluster-engine 的 ``QbEvaluator``（外部指标
只在非噪声样本上计算，噪声比例另行记录）。**不在这里重写公式**，否则现场数字
会与论文表格对不上。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: 真值列的候选字段名（按优先级）。全条命中才认，避免"部分标注"悄悄改变口径。
GROUND_TRUTH_FIELDS: tuple[str, ...] = (
    "category",
    "label",
    "true_label",
    "target",
    "class",
    "类别",
    "标签",
)

#: 对外输出（以及差值比较）使用的指标键。与离线基准 comparison 的键保持一致，
#: 前端因此可以用同一套渲染逻辑处理"本次运行"与"归档基准"。
METRIC_KEYS: tuple[str, ...] = (
    "ari",
    "vm",
    "fms",
    "ami",
    "hs",
    "cs",
    "score",
    "nClusters",
    "noiseRatio",
)


def extract_ground_truth(
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[str] | None, str | None]:
    """从逐条 metadata 里取出真值标签。

    返回 ``(labels, field)``；只有当**每一条**样本都带同一个非空标签时才认定
    可用，否则返回 ``(None, None)``。宁可不算，也不要拿"部分标注 + 大量空值"
    算出一组误导人的指标。
    """

    if not metadata:
        return None, None
    # 字段名大小写不敏感：上传的 CSV 可能写成 Category / CATEGORY
    actual_by_lower: dict[str, str] = {}
    for record in metadata:
        for key in record:
            actual_by_lower.setdefault(str(key).strip().lower(), str(key))
    for field in GROUND_TRUTH_FIELDS:
        actual = actual_by_lower.get(field)
        if actual is None:
            continue
        values = [str(record.get(actual) or "").strip() for record in metadata]
        if all(values):
            return values, actual
    return None, None


def _normalize(raw: Mapping[str, Any]) -> dict[str, Any]:
    """把 QbEvaluator 的 snake_case 结果统一成 camelCase 展示口径。

    这里用白名单收口：只放行已知指标，避免 `n_noise` 这类内部字段混进对外契约
    （噪声绝对条数由 `noiseRatio` + 样本总数即可推出，不必重复下发）。
    """

    out: dict[str, Any] = {}
    for key in ("ari", "nmi", "vm", "fms", "ami", "hs", "cs", "score"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = round(float(value), 6)
    n_clusters = raw.get("n_clusters")
    if isinstance(n_clusters, (int, float)) and not isinstance(n_clusters, bool):
        out["nClusters"] = int(n_clusters)
    noise_ratio = raw.get("noise_ratio")
    if isinstance(noise_ratio, (int, float)) and not isinstance(noise_ratio, bool):
        out["noiseRatio"] = round(float(noise_ratio), 6)
    return out


def evaluate_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """计算 6 项外部指标（复用 cluster-engine 的评估器，口径与论文一致）。"""

    import numpy as np
    from retrain_cluster.evaluation.metrics import QbEvaluator

    raw = QbEvaluator()(np.asarray(list(true_labels)), np.asarray(list(predicted_labels)))
    return _normalize(dict(raw))


def compare_metrics(
    base: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, float]:
    """``target - base`` 的逐项差值 + ARI 相对增益。

    方向与离线基准一致：``base`` 是基线（净化关 / nr0），``target`` 是本次主
    运行（净化开 / 完整框架）。
    """

    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        left, right = base.get(key), target.get(key)
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[key] = round(float(right) - float(left), 4)
    base_ari, target_ari = base.get("ari"), target.get("ari")
    if isinstance(base_ari, (int, float)) and isinstance(target_ari, (int, float)) and base_ari:
        out["ariRelative"] = round((float(target_ari) - float(base_ari)) / float(base_ari), 4)
    return out
