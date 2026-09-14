# 上游缺失子包的重建说明（`artifacts/` 与 `data/`）

> 这份说明针对一个**上游仓库缺陷**，不是 SPEAR 功能的一部分，但它是所有功能的
> 前置条件：没有它，`import retrain_cluster` 直接失败。

## 问题

上游 `main`（`daf268a v4-demo`）里，
`src/retrain_cluster/data/` 与 `src/retrain_cluster/artifacts/` 两个子包**从未被提交**，
但代码到处在 import 它们：

| 调用方 | 依赖 |
|---|---|
| `config.py` | `artifacts.fingerprints.{fingerprint,file_hash}` |
| `services/clustering.py` | `artifacts.cache.EmbeddingCache`、`artifacts.runs.RunStore`、`artifacts.text_cache.TextEmbeddingCache`、`data.validation.{validate_items,validate_matrix}` |
| `services/semantic_clustering.py` | `artifacts.text_cache.{TextEmbeddingCache,build_key}`、`data.normalization.{NORMALIZATION_VERSION,build_corpus,expand_labels}` |
| `api/app.py`、`retrieval/*`、`cli.py`、`features/reduction.py`、`services/experiments.py` | 同族的指纹/缓存/清单/加载器 |

根因是 `backend/python/cluster-engine/.gitignore` 里两条**无前导斜杠**的规则：

```
data/
artifacts/
```

它们匹配任意层级的同名目录，于是把源码子包也一起忽略了。已在 `.gitignore` 末尾
用 `!src/retrain_cluster/data/` 与 `!src/retrain_cluster/artifacts/` 收回源码目录；
运行期目录（`cluster-engine/data/`、`cluster-engine/artifacts/`）仍然被忽略。

## 重建原则

这些文件是**按现有测试与调用点反推重建**的，不是从别处拷贝：

* 可观察行为以仓库里已提交的测试为准（`tests/unit/test_artifacts_runs.py`、
  `test_data_cache.py`、`test_semantic_runs.py`、`test_validation.py`、
  `test_category_corpus.py`、`test_semantic_pipeline.py` 等）；
* 公共 API 以调用点的实际用法为准（函数名、参数名、返回结构、异常码与错误消息）；
* 保持既有风格：领域异常统一走 `ClusterError(code, message, status)`，
  每个包声明 `__all__`，子包可独立导入（`tests/unit/test_package_api.py` 会在
  全新解释器里逐个 `import` 来暴露循环依赖）。

因此 `artifacts/__init__.py` 使用 PEP-562 惰性导出，避免
`types → artifacts.fingerprints → cache → data.validation → loaders → types` 的真实环。

## 必须知道的重建边界

* `data/normalization.py` 的规范化口径（无效文本判定、长文本阈值、去重键）来自测试
  固定的性质，**不等于**原始实现逐字复刻；如果上游提供原文件，应当以原文件为准并
  重跑 `tests/unit/test_semantic_pipeline.py`。
* `data/category_corpus.py` 的启发式（占位叶子、分类定义行、稀有分类回退）由
  `tests/unit/test_category_corpus.py` 钉住，真实数据不变量（`datas/rules/规则.txt`）
  也已通过。
* 重建细节与未解决项见 `src/retrain_cluster/data/RECONSTRUCTION-NOTES.md`。

## 验证

```bash
cd backend/python/cluster-engine && python -m pytest
# 276 passed, 1 skipped, 4 deselected

cd backend/python && python -m pytest
# 95 passed, 9 deselected
```

网关套件通过即说明：`retrain_cluster` 能被完整导入、`Catalog` 能加载仓库里的
全部真实 profile、`SemanticRunStore` 能支撑作业明细的分页与完整性校验。
