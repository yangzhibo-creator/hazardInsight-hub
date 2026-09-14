# SPEAR 全量验收证据（预演机 GPU 实测）

> 本文件记录一次**真机、全量、可复现**的验收运行，用来回答"这套东西到底跑出什么数"。
> 所有数字都来自实际运行，不接受"应该可以"。原始日志与产物见文末路径。

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | 预演机，2× NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB/卡） |
| 计算卡 | `CUDA_VISIBLE_DEVICES=1`（物理 GPU 1；GPU 0 被无关服务占用约 54 GiB） |
| 解释器 | `/root/spear-venv/bin/python`，Python 3.11.16（venv 叠加在 conda `comp` 上） |
| 关键版本 | torch 2.8.0+cu129、transformers 5.17.0、sentence-transformers 6.0.1、chromadb 1.5.9、scikit-learn 1.9.1、numpy 2.4.6、optuna 5.0.0、hdbscan 0.8.44 |
| 向量模型 | `bge-large-zh-v1.5`，10 个必需文件 SHA-256 与 `configs/models.toml` **逐一相同**（无需重建清单） |
| 净化模型 | `Qwen3.8-27B`，常驻约 81.7 GiB；缺少 `causal_conv1d` / `flash-linear-attention`，走参考实现回退（约 3.93 条/秒） |
| 数据 | 测试集 20,198 条 / 353 类（`labels.json`，仅用于最终评估）；知识库来源 `database.txt` |

## 2. 知识库

```bash
python -m retrain_cluster build-kb --model-id bge-large-zh-v1.5 \
  --id bge-large-raw-lines-v1 --input data/database.txt
```

* manifest `count` = **18,170**；打开后的 collection `count()` = **18,170**（一致）
* `processing=raw-lines-v1`、`metric=l2`、`dimension=1024`、`status=completed`、`records_sha256` 重新计算后一致
* 读取口径与历史一致：`load_knowledge_texts` 共 18,170 行 / 17,927 个不同字符串；首条为 `'- 焊接电流书写错误'`（`- ` 前缀保留），末条为空串（尾部换行）

## 3. 全量 20,198 条结果（引擎自带 `QbEvaluator`，全部 0 噪声）

| profile | 净化 | ARI | VM | 预测簇数 |
|---|---|---|---|---|
| `bge-large-agglomerative-nr0-legacy-v1` | – | 0.154483 | 0.590283 | 271 |
| `bge-large-agglomerative-nr1-legacy-v1` | – | 0.275355 | 0.681373 | 203 |
| `spear_purified_retrieval` | **关** | 0.275355 | 0.681373 | 203 |
| `spear_purified` | 关 | 0.154483 | 0.590283 | 271 |
| **`spear_purified_retrieval`** | **开** | **0.246522** | 0.649224 | 260 |
| `spear_purified` | 开 | 0.211021 | 0.628592 | 294 |

分阶段耗时（秒，嵌入 / 知识库打开 / 检索+融合 / 聚类）：

* nr0：353.96 / 0 / 0.02 / 108.91
* nr1：0.09 / 2.19 / 11.99 / 96.20
* `spear_purified_retrieval` 净化关：0.10 / 2.18 / 12.07 / 98.01
* `spear_purified_retrieval` 净化开：0.09 / 2.23 / 11.91 / 97.78

## 4. "关掉净化逐位一致"在全量上成立

| 对比 | 不同标签数 |
|---|---|
| nr1 legacy ↔ `spear_purified_retrieval` 净化关 | **0 / 20,198** |
| nr0 legacy ↔ `spear_purified` 净化关 | **0 / 20,198** |

（`np.array_equal` 为真；不是近似、不是指标相同，而是逐条标签相同。）

## 5. 极性护栏在真模型上生效

* 10 条实探：裸 Qwen 违规 1 条，经护栏后 0 条；全量运行护栏命中 **1,471 / 20,198（7.28%）**，并触发 `POLARITY_GUARD_TRIGGERED`。
* 原始复现（陷阱 P1，逐字）：
  * 输入：`2025.11.03 在工艺评定车间巡检发现焊机二维码上名称与系统不一致，已要求更换二维码标签`
  * 裸 Qwen：`焊机二维码名称与系统一致`
  * 护栏后：`焊机二维码上名称与系统不一致`
* 净化开启的全量运行报告 `backend=cache`（命中预计算映射）：这是 `backend=auto` 的既定生产路径；映射由 Qwen 离线生成，抽样核对与在线 Qwen 0/64 差异，且护栏在读取时同样生效。

## 6. 与验收目标的冲突（必须明确，不许粉饰）

任务书要求完整框架 ARI 落在 **0.26–0.29**。净化**开启**的实测是 **0.2465，未达区间**。

原因排查（均未改配置）：

* 归档的 0.2705 来自 `full_result.json`，其中 `"enable_purify": false`——它是**净化关**的 nr1 数字，净化开的数值此前从未被测过；
* 护栏不是原因：关掉护栏为 0.2451（更差）；
* 截断高度不是原因：在保存的融合矩阵上重切 t∈[2.0, 4.0]，ARI 区间 0.2068–0.2477，最高 0.2477，任何切法都到不了 0.26；
* 真实原因：净化让**纯向量路径**变好（0.1545→0.2110），却让**检索增强路径**变差（0.2754→0.2465）。

按任务书"不要重新调参、发现冲突先报告"的要求，此处**没有**为净化后输入重新搜索截断高度/β/k；结论是：**当前实现与配置下，净化开的完整框架达不到 0.26–0.29**，0.2465 就是它在本环境的真实值。

## 7. 测试

| 命令 | 结果 |
|---|---|
| `pytest`（默认，排除 integration） | `276 passed, 1 skipped, 4 deselected`（开发机，3.12 venv） |
| `pytest -m integration`（配有 bge + Qwen） | `4 passed`（预演机；含跨用例显存回收，否则会因陷阱 P3 在同进程第二次装载 Qwen 时 OOM） |
| 网关 `pytest` | `95 passed, 9 deselected` |

集成测试的两处工程加固（本次一并修）：

1. Qwen 净化用例不再硬编码 `/models/Qwen3.8-27B`，改为从 `configs/app.toml` 的
   `spear_purified.purification.model_path` 解析，目录不存在即 skip，并允许
   `SPEAR_LLM_MODEL_PATH` 覆盖；
2. `tests/integration/conftest.py` 在每个集成用例后 `gc.collect()` +
   `torch.cuda.empty_cache()`，让"Qwen 只能装一份"（陷阱 P3）在测试进程内也成立。

## 8. 原始证据

* 汇总报告（工作区）：`spear-acceptance-report.md`
* 原始产物：`spear-evidence/`（每次运行的 JSON/pred.npy、`integration.log`、
  `kb_build.json`、`polarity_probe.log`、`precompute.log`、`threshold_scan.json`）
* 服务器：仓库 `/root/spear-hub`，运行脚本 `/root/spear-run/`
* 服务器上 497/497 个被跟踪文件 SHA-256 在运行前后一致——**没有改动任何 tracked 文件**；
  只新增了 gitignore 覆盖的输入与产物（`cluster-engine/data/`、`models/`、`artifacts/`）
