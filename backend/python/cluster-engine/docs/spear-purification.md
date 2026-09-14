# SPEAR：语义净化 + 表示增强 + 凝聚层次聚类

本文说明本仓库新增的 **`spear-v1`** 实现版本：数据口径、配置方式、护栏与降级链、
以及"关掉净化后逐位一致"这条硬约束是怎么被结构性保证的。

> 术语：SPEAR 的三个阶段是
> ① **语义净化**（输入层剥离四类噪声）→ ② **表示增强**（检索增强融合）→
> ③ **簇划分**（Ward 凝聚层次聚类）。本实现里 ②③ 复用既有 `legacy-v1` 的
> `FeaturePipeline` 与算法注册表，只有 ① 是新增能力。

## 1. 它与 `legacy-v1` / `semantic-v1` 的关系

| 实现版本 | 输入层 | 表示层 | 算法 | 典型 profile |
|---|---|---|---|---|
| `legacy-v1` | 原文 | `β·X + (1−β)·mean(top-k 邻居)`（可关） | 11 种注册算法 | `bge-large-agglomerative-nr0/nr1-legacy-v1` |
| `semantic-v1` | 规范化 + 精确去重 | L2 + 确定性 PCA + Auto-K | `semantic_auto_kmeans` | `bge-large-semantic-auto-kmeans-v1` |
| **`spear-v1`** | **语义净化**（可开关、带护栏、可降级） | **复用 `legacy-v1` 的 FeaturePipeline** | **复用 legacy 算法集合** | `spear_purified`、`spear_purified_retrieval` |

分派按 `implementation_version` 查表（`services/strategies.py`），不按算法名猜测：
"新算法 + 旧版本"这类组合在 `Catalog` 加载阶段就会以 `INVALID_PROFILE` 失败。

## 2. 配置

```jsonc
{
  "profile_id": "spear_purified_retrieval",
  "algorithm": "agglomerative",
  "model_id": "bge-large-zh-v1.5",
  "algorithm_params": { "distance_threshold": 2.6535528326328843 },
  "features": { "beta": 0.4973823453680069, "n_results": 6, "pca_dim": 0, "version": "spear-v1" },
  "purification": {
    "enabled": true,           // 总开关：false 时退化到改动前的行为
    "backend": "auto",         // auto | qwen | rule | cache
    "model_path": "../models/Qwen3.8-27B",
    "cache_file": "../data/purify_map.json",
    "device": "cuda",
    "batch_size": 32,
    "max_new_tokens": 64,
    "guard": true              // 极性保真护栏（默认必须开）
  },
  "knowledge_base_id": "bge-large-raw-lines-v1",
  "implementation_version": "spear-v1"
}
```

* `backend=auto` 的优先级：**预计算映射 → 本地 Qwen → 规则净化**。
  现场不重跑 Qwen 逐条净化（2 万条约 1.7–2.8 小时），用离线映射查表；
  未命中的少量文本才走兜底。
* `model_path` / `cache_file` 支持相对路径，按 `models.toml` 所在目录解析成绝对路径。
* 新增两个 profile：`spear_purified`（净化 + 纯向量）与
  `spear_purified_retrieval`（净化 + 检索增强，即完整框架）。

### 请求级开关

`POST /api/v1/clusterings` 额外接受一个可选字段（`extra=forbid` 契约的一部分）：

```json
{ "profile_id": "spear_purified", "items": [...], "purification": { "enabled": false } }
```

只覆盖显式给出的字段；legacy / semantic profile 收到覆盖会返回 `INVALID_PROFILE`
（静默忽略会让调用方以为自己关掉了净化）。

## 3. 极性保真护栏（陷阱 P1）

实测：Qwen3.8-27B 会把 `焊机二维码上名称与系统不一致` 净化成
`焊机二维码名称与系统一致`——否定词丢失、缺陷含义反转；即使提示词写了硬约束并
给出完全相同的少样本示例仍会复现。

因此输出侧有一道**确定性校验**（`purification/guard.py`）：

1. 源文本含极性/缺陷标记（`不 未 无 非 缺 漏 超 误 错 坏 损 变形 失效 松动 偏
   倾斜 堵 裂 锈 脏 乱 少 多`）时，净化结果也必须至少含一个；
2. 未通过的整条改用 `RulePurifier` —— 正则净化**只删不改**，按构造不可能反转语义；
3. 拦截条数写进结果（`purification.guard_hits`）并触发
   `POLARITY_GUARD_TRIGGERED` 告警，现场可核对"护栏真的在工作"。

## 4. 降级链与可用性报告

| 情形 | 行为 | 报告 |
|---|---|---|
| `enabled=false` | 恒等变换，走改动前路径 | `backend=identity` |
| Qwen 权重缺失 / 加载失败 / 显存不足 | 自动改用规则净化，服务照常 | `degraded=true`、`reason`、告警 `PURIFIER_DEGRADED_TO_RULE` |
| 预计算映射缺失 | 回落到 Qwen 或规则 | `degraded`（对应原因） |
| 映射文件损坏 | 回落到 Qwen 或规则 | 同上 |

净化是**软依赖**：它不参与 profile 的 `available` 判定（那会让缺模型时整个 profile
不可用），但会出现在 `/api/v1/profiles`、网关 `/api/clustering/profiles` 与
每次运行结果的 `purification` 字段里。

实例复用：`PurifierRegistry` 按配置缓存净化器。Qwen3.8-27B 常驻约 75 GiB 显存且
**只能装一份**（第二个实例会直接 CUDA OOM，陷阱 P3），因此同一进程内不会重复加载；
需要腾显存时调用 `release()`。

## 5. "关掉净化逐位一致"是怎么保证的

不是在 SPEAR 里重写一遍 legacy，而是让净化关闭时：

1. `encoded_texts` 原样设为原始文本；
2. 复用**同一个** `ClusteringService.embeddings`（同一嵌入缓存、同一编码器实例）；
3. 复用**同一个** `ChromaRetriever` 与 `FeaturePipeline`；
4. 复用**同一个** `validate_params` 返回的算法函数。

计算路径完全相同，因此标签按构造逐位相等。`tests/unit/test_spear_strategy.py` 用
`==` 全等断言钉住了 nr0 与 nr1 两条路径。

## 6. 验证

```bash
# 引擎（默认不需要 GPU / 模型；integration 用例显式排除）
cd backend/python/cluster-engine
python -m pytest                     # 276 passed, 1 skipped, 4 deselected
python -m pytest -m integration      # 需要真实本地模型

# 网关（替身引擎，不需要模型）
cd backend/python
python -m pytest                     # 95 passed, 9 deselected
```

验收标准 4 的证据在 `test_spear_strategy.py`：

* `test_purification_off_is_bit_identical_to_the_legacy_path`
* `test_retrieval_off_is_bit_identical_to_the_legacy_retrieval_path`

全量 20,198 条已在预演机 GPU 上实跑，逐位一致在全量上成立（nr1 vs 净化关 0/20,198
条不同），知识库 18,170 条，集成测试通过。**但净化开启的完整框架 ARI 实测为 0.2465，
低于任务书要求的 0.26–0.29**——归档的 0.2705 实际是"净化关"的数字。完整数据、根因
排查与"未擅自调参"的说明见
[`spear-acceptance-evidence.md`](spear-acceptance-evidence.md)；界面用的归档基准见
`backend/python/app/data/offline_baseline.json`（行名已显式区分净化开/关）。

## 7. 现场部署

离网单机部署见 [`deploy/onsite-h20/README.md`](../../../../deploy/onsite-h20/README.md)
（**现场定制，不是上游功能**）。
