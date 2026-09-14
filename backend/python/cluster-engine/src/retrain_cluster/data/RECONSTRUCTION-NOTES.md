# 重建说明（RECONSTRUCTION-NOTES）

本文件记录重建 `retrain_cluster.artifacts` / `retrain_cluster.data` 时发现的、
**位于这两个包之外**、因此无法在本任务范围内修复的测试失败。两个包本身的
契约测试（`test_artifacts_runs.py`、`test_data_cache.py`、`test_semantic_runs.py`、
`test_validation.py`、`test_boundaries.py`、`test_category_corpus.py`、
`test_semantic_pipeline.py`、`test_package_api.py` 等）全部通过。

## 环境缺少可选依赖（与本任务无关）

`pyproject.toml` 把这些标注为 optional：`hdbscan` / `chinese-whispers` /
`networkx` / `umap-learn`，但当前 `/userdata/venvs/hub` 未安装：

* `tests/regression/test_golden_labels.py` 在模块顶层
  `from retrain_cluster.clustering import chinese_whispers`，后者第 11 行
  `import networkx`，于是**收集阶段**即报 `ModuleNotFoundError: networkx`，
  导致 `pytest -q` 整体中断（1 error）。该模块的 fixture 虽用
  `pytest.importorskip("chinese_whispers")`，但为时已晚——顶层导入先失败了。
* `tests/api/test_api.py::test_all_ten_algorithms_both_modes_via_api` 中
  `hdbscan` / `chinese_whispers` 两组参数各失败 2 次，错误为
  `ALGORITHM_UNAVAILABLE: Algorithm dependencies are not installed`。

修复需要安装可选依赖，或把 `clustering/chinese_whispers.py` 的 `networkx`
导入改为惰性——两者都在本任务允许修改的两个包目录之外。

## 并发编辑中的文件（另一名工程师）

* `tests/unit/test_semantic_pipeline.py::test_capability_table_only_claims_registered_algorithms`
  失败：`strategies.py` 新增了 `SPEAR_VERSION="spear-v1"` 并让
  `list_strategy_versions()` 返回三个版本，而测试仍断言恰为
  `{legacy-v1, semantic-v1}`。
* `tests/api/test_schemas_routes.py::test_response_serialization_roundtrip`
  失败：`ClusteringResponse` 新增了 `purification` 字段，`model_dump()` 多出
  `{"purification": None}`。
* `tests/unit/test_purification.py`、`tests/integration/test_spear_integration.py`
  失败：需要本地模型目录（`/models/Qwen3.8-27B` 等），当前环境没有。

这些失败都源自 `src/retrain_cluster/services/strategies.py`、`api/schemas.py`、
`purification/*`、`services/spear_clustering.py` 等并发改动，不在本任务范围内。
