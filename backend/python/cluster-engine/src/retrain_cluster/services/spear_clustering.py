"""SpearStrategy：SPEAR 三阶段流水线的编排层（交付物 A 的核心）。

```
原始文本
  → ① 语义净化（LLM 剥离四类噪声；可关闭；带极性护栏；失败自动降级）
  → ② 表示增强（efinal = β·e_input + (1−β)·mean(top-k 浓缩锚点)）
  → ③ 簇划分（复用 legacy 的算法注册表，实测口径为 Ward 凝聚层次聚类）
```

三条必须守住的工程约束：

1. **关掉净化必须与改动前逐位一致**。做法不是"再实现一遍 legacy"，
   而是让净化关闭时把 ``encoded_texts`` 原样设为原始文本，之后复用**同一个**
   嵌入缓存、同一个检索器、同一个 ``FeaturePipeline`` 与同一个算法函数。
   计算路径完全相同，因此标签按构造逐位相等（验收标准 4）。
2. **净化只发生在编码前**，且只做剥离。净化器不参与任何类别判定。
3. **降级要如实报告**。规则兜底、护栏拦截、缓存未命中都会被写进结果的
   ``purification`` 与 ``warnings``，绝不静默替换。
"""

from __future__ import annotations

from dataclasses import asdict
import time
import uuid

import numpy as np

from ..artifacts.cache import EmbeddingCache
from ..artifacts.fingerprints import array_hash, fingerprint
from ..artifacts.runs import RunStore
from ..clustering.registry import WARNINGS, validate_params
from ..config import Catalog, model_fingerprint
from ..data.validation import validate_items, validate_matrix
from ..errors import ClusterError
from ..features.pipeline import FeaturePipeline
from ..logging import event
from ..purification import PurifierRegistry, probe
from .strategies import SPEAR_VERSION, StrategyMeta

#: 结果里携带的净化前后对照条数上限。演示只需要一屏，
#: 全量对照会让 2 万条请求的结果体膨胀到不可接受。
MAX_PURIFY_SAMPLES = 50

#: 与 legacy 一致的"近零范数"兜底：净化后若全为空向量，说明输入或模型有问题。
BAD_VECTOR_ABSOLUTE_LIMIT = 20


class SpearStrategy:
    """spear-v1 的策略实现（净化 + 可选检索增强 + legacy 算法）。"""

    def __init__(
        self,
        *,
        profile,
        settings,
        catalog: Catalog,
        encoders,
        embedder=None,
        retriever_for=None,
        purifiers: PurifierRegistry | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self.catalog = catalog
        self.encoders = encoders
        # 注入的 embedder/retriever_for 来自 ClusteringService：复用它的缓存与实例，
        # 这是"关掉净化后逐位一致"的结构性保证，而不是靠测试碰巧通过。
        self.embedder = embedder
        self.retriever_for = retriever_for
        self.purifiers = purifiers or PurifierRegistry()
        self.runs = RunStore(settings.artifacts_dir / "runs")
        self._fallback_cache = EmbeddingCache(settings.artifacts_dir / "embeddings")
        status = self.purifiers.status(profile.purification)
        self.meta = StrategyMeta(
            implementation_version=SPEAR_VERSION,
            algorithms=(profile.algorithm,),
            supports_single_item=False,
            supports_cache_only=False,
            calibration_version=None,
            features={
                "version": profile.features.version,
                "n_results": profile.features.n_results,
                "beta": profile.features.beta,
                "purification": status.as_dict(),
            },
        )

    # ------------------------------------------------------------------ 向量
    def _embeddings(self, texts, model_id):
        """取向量：优先用注入的 embedder，否则退回本地缓存的等价实现。"""

        if self.embedder is not None:
            return self.embedder(texts, model_id)
        spec = self.catalog.model(model_id)
        model_hash = model_fingerprint(spec)
        key = self._fallback_cache.key(texts, model_hash)
        values = self._fallback_cache.load(key, rows=len(texts), dimension=spec["dimension"])
        hit = values is not None
        if not hit:
            values = self.encoders.get(spec, model_hash).encode(texts)
            values = validate_matrix(values, rows=len(texts), dimension=spec["dimension"])
            self._fallback_cache.save(key, values)
        return values, model_hash, hit

    # ------------------------------------------------------------------ 净化
    def _purify(self, items):
        """执行阶段 1；返回（送入编码的文本, 报告, 告警）。"""

        spec = self.profile.purification
        warnings: list[str] = []
        if not spec.enabled:
            status = probe(spec)
            return [item["text"] for item in items], {
                "enabled": False,
                "backend": status.effective_backend,
                "requested_backend": status.requested_backend,
                "degraded": False,
                "reason": status.reason,
                "guarded": False,
                "guard_hits": 0,
                "total": 0,
                "samples": [],
                "elapsed_ms": 0.0,
            }, warnings

        purifier, status = self.purifiers.get(spec)
        started = time.monotonic()
        raw_texts = [item["text"] for item in items]
        purified = list(purifier.purify(raw_texts))
        if len(purified) != len(raw_texts):
            raise ClusterError("PURIFICATION_FAILED", "Purifier returned a mismatched batch", 500)
        # 空结果会让嵌入失去意义：宁可退回原文，也不能把一条记录变成空串
        encoded = [value if value and value.strip() else raw for value, raw in zip(purified, raw_texts)]
        elapsed = round((time.monotonic() - started) * 1000, 3)

        guard_hits = int(getattr(purifier, "last_guard_count", 0))
        total = int(getattr(purifier, "last_total", len(raw_texts)))
        if status.degraded:
            warnings.append("PURIFIER_DEGRADED_TO_RULE")
        if spec.guard and guard_hits:
            warnings.append("POLARITY_GUARD_TRIGGERED")

        samples = [
            {"id": item["id"], "raw": raw, "condensed": value}
            for item, raw, value in list(zip(items, raw_texts, encoded))[:MAX_PURIFY_SAMPLES]
        ]
        report = {
            "enabled": True,
            "backend": status.effective_backend,
            "requested_backend": status.requested_backend,
            "degraded": status.degraded,
            "reason": status.reason,
            "guarded": bool(spec.guard),
            "guard_hits": guard_hits,
            "total": total,
            "samples": samples,
            "elapsed_ms": elapsed,
        }
        return encoded, report, warnings

    # ------------------------------------------------------------------ 执行
    def run(self, items, *, enforce_api_limits: bool = False) -> dict:
        started = time.monotonic()
        profile = self.profile
        if profile.purification is None:
            raise ClusterError("INVALID_PROFILE", "spear-v1 profile requires a purification spec", 422)

        limit = min(self.settings.max_samples, profile.max_samples) if enforce_api_limits else profile.max_samples
        validate_items(items, maximum=limit)
        clusterer = validate_params(profile.algorithm, profile.algorithm_params, profile.backend)
        warnings = list(WARNINGS.get(profile.algorithm, []))

        # ① 语义净化
        encoded_texts, report, purify_warnings = self._purify(items)
        warnings.extend(code for code in purify_warnings if code not in warnings)

        spec = self.catalog.model(profile.model_id)
        model_hash = model_fingerprint(spec)
        # ② 表示增强：先取（净化后）输入向量，再按 profile 决定是否检索增强
        retriever = self.retriever_for(profile, model_hash) if self.retriever_for else None
        values, model_hash, hit = self._embeddings(encoded_texts, profile.model_id)
        transformed = FeaturePipeline(retriever, self.settings.query_batch).transform(values, profile.features)

        # ③ 簇划分
        pred = np.asarray(clusterer(transformed, **profile.algorithm_params))
        if pred.shape != (len(items),) or pred.dtype.kind not in "iu":
            raise ClusterError("INVALID_CLUSTER_OUTPUT", "Algorithm returned invalid labels", 500)

        if not spec.get("revision") or spec.get("revision") == "unversioned":
            warnings.append("REMOTE_MODEL_REVISION_UNPINNED")
        if profile.source_result:
            warnings.append("HISTORICAL_EMBEDDINGS_NOT_VERIFIED")
        if profile.features.pca_dim:
            warnings.append("LEGACY_SEPARATE_REDUCTION")

        result = {
            "run_id": "run_" + uuid.uuid4().hex,
            "profile_id": profile.profile_id,
            "algorithm": profile.algorithm,
            "implementation_version": SPEAR_VERSION,
            "n_samples": len(items),
            "n_clusters": len(set(pred) - {-1}),
            "n_noise": int((pred == -1).sum()),
            "assignments": [{"id": item["id"], "cluster_id": int(label)} for item, label in zip(items, pred)],
            "warnings": warnings,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "purification": report,
        }
        # 产物清单必须记清"净化到底用没用、用了谁、拦了多少条"，
        # 否则事后无法解释同一份输入为何两次结果不同（降级/护栏都会改变输入文本）。
        manifest = {
            "schema_version": 2,
            "run_id": result["run_id"],
            "implementation_version": SPEAR_VERSION,
            "config": asdict(profile),
            "profile_fingerprint": profile.fingerprint,
            "model": spec,
            "model_fingerprint": model_hash,
            "input_fingerprint": fingerprint([{"id": item.get("id"), "text": item.get("text")} for item in items]),
            "purified_fingerprint": fingerprint(encoded_texts) if report["enabled"] else None,
            "embedding_fingerprint": array_hash(values),
            "feature_fingerprint": array_hash(transformed),
            "cache_hit": hit,
            "purification": {key: value for key, value in report.items() if key != "samples"},
            "backend": profile.backend,
            "knowledge_base": getattr(retriever, "manifest", None),
            "baseline_kind": "spear-v1-current-environment",
            "seed": None,
        }
        self.runs.save(result, manifest)
        event(
            "completed",
            run_id=result["run_id"],
            algorithm=profile.algorithm,
            n_samples=len(items),
            elapsed_ms=result["elapsed_ms"],
            purification_backend=report["backend"],
            purify_samples=len(report["samples"]),
        )
        return result


__all__ = ["BAD_VECTOR_ABSOLUTE_LIMIT", "MAX_PURIFY_SAMPLES", "SpearStrategy"]
