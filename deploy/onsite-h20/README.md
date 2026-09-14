# deploy/onsite-h20 —— 离网现场演示部署（现场定制，**不是上游功能**）

> **提交边界声明**：本目录是为「人工智能能力现状摸底测评」现场演示做的部署定制，
> 与 `cluster-engine` 的上游能力无关。合并回上游时请**整个目录排除**，
> 或只合并聚类相关的 `backend/python/cluster-engine/**`、`web/pages/Clustering*.tsx`、
> `shared/clustering.ts` 等文件。详见根目录 `README.md` 的提交说明。

## 它是什么

把本仓库的**聚类网关（`backend/python/app`）+ 聚类引擎（`backend/python/cluster-engine`）
+ 前端聚类页面（Vite 构建产物）**装进一个镜像，在完全离网的 NVIDIA H20 单机上
用一条命令起服务，用于现场实机演示 SPEAR 三阶段流水线：

```
原始隐患/偏差文本
  → ① 语义净化（本地 Qwen3.8-27B，可开关，带极性护栏，失败自动降级为正则）
  → ② 表示增强（efinal = β·e_input + (1−β)·mean(top-k 浓缩锚点)）
  → ③ 簇划分（Ward 凝聚层次聚类）
```

现场端口 **6006**（需向平台报备），容器内由 `app_entry.py` 同时提供：

| 路径 | 用途 |
|---|---|
| `/` | 前端页面（Vite `dist` 产物） |
| `/api/health` | 健康检查（Dockerfile 的 `HEALTHCHECK` 与启动脚本都轮询它） |
| `/api/clustering/profiles` | profile 列表，含净化的可用性/降级状态 |
| `/api/clustering/run` | 同步聚类（含净化前后对照） |
| `/api/clustering/baseline` | 离线基准结果（实时失败时的兜底展示） |

## 目录内容

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 现场镜像（`FROM h20-inference:v1.0`，离线装依赖） |
| `Dockerfile.structural` | **结构验证专用**：`python:3.11-slim` + 联网装依赖，**不是交付路径** |
| `app_entry.py` | API + 前端静态资源组合入口 |
| `build_image.sh` | 构建 → 容器自检 `/api/health` → `docker save` → 打印 sha256 |
| `start_container.sh` | 载入镜像 → 起容器 → 轮询 `/api/health`（≤300s）→ 打印访问地址；失败打印日志 |
| `make_wheels.sh` | 下载 cp311/manylinux 离线 wheel（排除 torch） |
| `requirements.lock.txt` | 离线依赖锁定清单（115 项 + `jieba`；**不含 torch**） |
| `assets/` | 构建资产入口（模型/知识库/净化映射），见 `assets/README.md` |
| `web-dist/` | 前端产物；仓库里是占位页，正式构建时由 `build_image.sh` 覆盖 |

## 交付步骤（在构建机上）

```bash
# 0) 前提：已 docker load 平台提供的基础镜像 h20-inference:v1.0
docker images | grep h20-inference

# 1) 准备离线 wheel（只需一次；平台标签已放宽到 manylinux_2_28）
bash deploy/onsite-h20/make_wheels.sh

# 2) 准备资产：把 bge 权重与预置知识库放进 assets/（见 assets/README.md）
#    cp -a /path/to/bge-large-zh-v1.5      deploy/onsite-h20/assets/models/
#    cp -a /path/to/bge-large-raw-lines-v1 deploy/onsite-h20/assets/knowledge_bases/

# 3) 构建 + 自检 + 导出
bash deploy/onsite-h20/build_image.sh
# 产物：spear-onsite-h20.tar + spear-onsite-h20.tar.sha256
```

## 现场启动（H20 服务器，约 2–5 分钟）

```bash
# 介质：spear-onsite-h20.tar(+.sha256) + Qwen3.8-27B 权重目录
sha256sum -c spear-onsite-h20.tar.sha256          # 安审核对
QWEN_HOST_DIR=/data/models/Qwen3.8-27B \
  bash deploy/onsite-h20/start_container.sh --tar spear-onsite-h20.tar
# 无 Qwen 权重时的降级演示：
#   bash deploy/onsite-h20/start_container.sh --tar spear-onsite-h20.tar --no-purify
```

脚本会：载入 tar → 起容器 →（若给了 Qwen 目录，只读挂载到
`/app/cluster-engine/models/Qwen3.8-27B`）→ 轮询 `/api/health` 最多 300 秒 →
打印访问地址与健康检查 JSON。任一步失败都会直接打印 `docker logs --tail 80`。

## 环境变量（容器内）

| 变量 | 默认 | 说明 |
|---|---|---|
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | 强制离线，杜绝任何外网请求 |
| `EMBED_MODEL_PATH` | `/app/cluster-engine/models/bge-large-zh-v1.5` | 本地向量模型（进镜像） |
| `LLM_MODEL_PATH` | `/app/cluster-engine/models/Qwen3.8-27B` | 净化 LLM（**卷挂载**，不进镜像） |
| `ENABLE_PURIFY` | `1` | 净化总开关；`0` 时走规则净化 |
| `DEVICE` | `cuda` | 设备；结构验证镜像里为 `cpu` |
| `PORT` | `6006` | 服务端口 |
| `SERVE_WEB` | `1` | `0` 时只提供 API（排查前端问题时用） |

## 结构验证（不需要真基础镜像/真权重）

```bash
bash deploy/onsite-h20/build_image.sh --structural
# 等价于：
#   docker build -f deploy/onsite-h20/Dockerfile.structural -t spear-onsite-structural .
#   docker run --rm -p 16006:6006 spear-onsite-structural
#   curl -s http://127.0.0.1:16006/api/health
```

结构验证镜像**故意不装** torch / chromadb / sentence-transformers，用来验证
"依赖缺失时服务仍能起来并如实报告 degraded"，而不是假装一切正常。
它不能替代现场镜像的验收：GPU 推理、真实净化、真实知识库都必须在
`h20-inference:v1.0` 上验证。

实测（`python:3.11-slim` 结构验证镜像，2026-09-14）：

```
docker build -f deploy/onsite-h20/Dockerfile.structural -t spear-onsite-structural .   # OK
docker run -d -p 16006:6006 spear-onsite-structural
GET /api/health            -> 200（status=degraded：本机没有向量模型，如实报告）
GET /                      -> 200（前端占位页）
GET /api/clustering/baseline -> 200（comparison.ariRelative=0.7703）
profiles: spear_purified / spear_purified_retrieval 的 purification
          = {effectiveBackend: rule, degraded: true, reason: purification_model_missing}
```

注意：为了装下上传接口，运行期依赖里必须有 `python-multipart`（FastAPI 在注册
表单路由时就会检查它，缺它整个应用 import 失败）。锁定清单里已包含。

## 已知缺口（交付前必须补齐，不要当成"已经好了"）

1. **仓库里没有大文件**：`assets/models/bge-large-zh-v1.5`（约 1.3 GB）、
   `assets/knowledge_bases/bge-large-raw-lines-v1/`、`wheels/` 都不在 git 里，
   必须从交付介质拷入（`assets/README.md` 有说明）。
2. **`pytz` / `tzdata` 未列入锁定清单**：本清单的抓取环境里 `pandas` 声明了这两个依赖。
   网关运行期不使用 pandas（`tabular_reader.py` 直接解析 OOXML，XLSX 不依赖 openpyxl），
   因此 `--no-deps` 安装后不影响聚类。若后续要在镜像里跑分析脚本，请先补这两个 pin。
3. **离线基准结果是归档值，且必须区分净化开/关**：`backend/python/app/data/offline_baseline.json`
   里是预演机 GPU 全量实测（20,198 条口径）。行名显式区分
   `nr1_legacy_purification_off`（0.2754）与
   `spear_purified_retrieval_purification_on`（0.2465）——此前把 0.2705 挂在
   "完整框架"名下是标注错误（该数字来自 `enable_purify: false` 的运行）。
   **净化开启的 0.2465 未达到任务书要求的 0.26–0.29**，界面与文档都如实展示；
   详见 `backend/python/cluster-engine/docs/spear-acceptance-evidence.md`。
4. **现场必须跑全量**：Ward 截断高度与样本量强相关（陷阱 P2），抽样运行的指标不具可比性。

## 时间预算（对应现场 70 分钟）

| 阶段 | 预期 |
|---|---|
| `docker load` + 起容器 + `/api/health` | 2–5 分钟（tar 不含 55 GB 权重） |
| 引擎就绪（模型清单校验） | 首次 10–30 秒 |
| 净化前后对照（`/api/clustering/run` 小批量，开启净化） | 首条需等 Qwen 装载约 20 秒，之后约 0.3–0.5 秒/条 |
| 无净化全量 20,198 条端到端 | 约 214 秒（嵌入 10s + 知识库检索 12s + 聚类 2×105s） |
