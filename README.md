# 核电工程隐患智能识别与定级系统（Demo）

**Nuclear Power Engineering Hazard Intelligent Identification & Grading System**

面向核电工程建设现场的 **AI 辅助** 安全隐患识别与 A/B/C/D 智能定级演示系统。上传工程现场图片后，系统通过多模态大模型识别隐患，结合「通用法规 / 核电专用标准 / 企业内部文件 / 历史相似案例」四类知识源，输出可追溯的隐患分析结果、定级依据与整改建议，并支持在原图上标注隐患区域。

> ⚠️ 本系统为 **AI 辅助判断** 工具，页面持续提示：**最终结果应由专业安全人员结合有效受控文件复核确认。**

---

## 一、核心能力

- **多模态图像识别**：0~10 条隐患候选，自动判断「未发现明确安全隐患」；
- **四类知识源 RAG**：通用标准、核电专用标准、企业文件、历史案例（keyword 检索，预留 `KnowledgeRetriever` 接口）；
- **Agent Runtime 显式工作流**：Preflight → Vision → RAG → Rules → Context → Reasoning → Validation → Result；
- **真实流式进度**：工作台通过 SSE 接收 run / step / tool / retrieval / cache / fallback 事件，不再使用前端假进度；
- **可观测与可追踪**：每次分析生成 conversation / session / task / run / trace ID，结果固化 workflow / prompt / rule / knowledge 版本与缓存、耗时指标；
- **Agent 上下文 Token 可观测**：按 Session → Agent → Turn → LLM Step 保存请求前不可变 Context Snapshot，并把六类 item 估算与 Provider actual usage、缓存、耗时和累计 Ledger 分开展示；支持当前构成、Step/Turn 趋势、stable-ID Diff 与原文明细；
- **A/B/C/D 隐患智能定级**：输出 `等级 + 定级理由 + 严重度/可能性/风险值`，风险值 = 严重度 × 可能性 辅助计算，机器风险规则守卫模型输出；
- **法规依据展示**：每一条引用均展示文件、编号、条款、原文与引用原因；
- **历史相似案例 TOP3**：相似度可视化 + 历史定级 + 历史整改措施；
- **原图 bbox 标注**：等级配色方框、编号标签、点击联动、隐藏标注、导出「隐患说明图」PNG；
- **目录导航（三级）**：一级目录为 `隐患发现 · 隐患管理 · 隐患知识库 · 偏差聚类 · 设置`；二级分组沿用原有功能划分（如「隐患管理」下的隐患识别 / 历史记录 / 隐患测试 / 隐患防控），三级为具体页面。绑定单个页面的分组渲染为直接跳转，不再多套一层；
- **聚类分析（聚类展示 / 聚类测试 / 偏差数据库）**：对隐患文本做无监督聚类，输出簇结构、关键词、代表样本与二维（PCA）散点分布；算法与向量化由独立的 Python 聚类服务（`cluster-engine`，11 种算法 / 20 个 profile / 本地 BGE 中文向量模型）提供，前端不做任何算法近似；「聚类测试」页可查看服务就绪状态、算法与 profile 可用性，并对同一批数据做多算法横向对比；「偏差数据库」页可查看/生成检索增强用的语义知识库，并在聚类时选择使用哪一个；
- **人工干预**：修改定级、编辑隐患描述、编辑整改建议（立即/整改/预防三类）；
- **隐患发现**：固定摄像头抓拍识别 + 具身智能机器人自动巡检（多点位画面分析、进度与结果汇总），一键转入隐患台账；
- **隐患防控**：隐患台账闭环管理（排查-登记-整改-复查-销号），支持来源、等级、状态流转、责任人/时限与台账导出；
- **定级规则库**：内置 A/B/C/D 分级规则 + 行业判定规则 **TXT / CSV 双格式导入**；导入规则会按正确的 a/b/c 业务 taxonomy 参与匹配与上下文注入，星标规则触发人工复核，不会被误映射为风险等级；
- **知识图谱**：定级规则**圆球（径向）知识图谱**——根球在圆心，子节点按层级分布同心环；**点击圆球展开 / 收回**对应子节点，带 +/− 标记、搜索自动展开命中路径并高亮、节点详情、滚轮缩放 / 全部展开与收回、**导出 PNG**；
- **结果管理**：保存到 localStorage、刷新恢复、隐患记录页、导出 HTML 报告（可打印 PDF）。

## 二、技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 19 + TypeScript + Vite 6 + 自研组件 / CSS 设计系统（桌面优先，蓝白企业风） |
| 后端 | Node.js + Express 5 + Multer（图片上传）+ Zod（Schema 校验） |
| 聚类服务 | Python 3.12 + FastAPI + scikit-learn / hdbscan / sentence-transformers（本地 BGE 向量模型）+ Pydantic（契约校验），算法层为内置的 `cluster-engine` |
| 模型 | 抽象 Multimodal Provider，默认 `OpenAI 兼容` 视觉接口（纯 fetch，无供应商 SDK） |
| 知识库 | 本地 JSON + 轻量关键词/类别加权检索（预留 ES / Milvus / pgvector 替换点） |

## 三、快速开始

### 3.1 环境要求

| 依赖 | 版本 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| Node.js | ≥ 20.19.0 | 必需 | 见 `package.json` 的 `engines`；低于此版本 Vite 6 / React 19 可能启动失败 |
| npm | 随 Node | 必需 | 依赖锁文件为 `package-lock.json` |
| Python | 3.12 | 可选 | **仅聚类分析需要**，不跑聚类可以不装 |
| 本地向量模型 | bge-large-zh-v1.5（约 1.3 GB） | 可选 | 仅聚类需要，**未随仓库分发**，需自行下载（见 3.4） |

> 聚类分析是**可选增强**：Python 服务未启动时其余功能完全不受影响，聚类页面会明确提示「服务未就绪」。

### 3.2 安装

```bash
npm install
```

### 3.3 启动（开发模式）

| 命令 | 启动内容 | 端口 |
| --- | --- | --- |
| `npm run dev` | Node API + Vite 前端 | 3001 + 5175 |
| `npm run dev:all` | Node API + Vite + Python 聚类服务 | 3001 + 5175 + 8000 |
| `npm run dev:linux` | Linux 开发启动全部服务，前端支持局域网访问 | 3001 + 5175 + 8000 |
| `npm run dev:api` | 仅 Node API | 3001 |
| `npm run dev:web` | 仅 Vite 前端 | 5175 |
| `npm run dev:py` | 仅 Python 聚类服务 | 8000 |

日常开发一条命令即可：

```bash
npm run dev
```

- 前端入口：<http://localhost:5175>
- API：<http://localhost:3001>

> **务必通过 5175 访问前端**。开发模式下 Express 不托管 SPA（直接访问 `http://localhost:3001/` 会返回 404），Vite 才是页面入口；只有生产模式才由 Express 托管 `dist`。

打开页面后，先在「模型配置」中配置真实多模态模型，然后：

1. 点击「使用演示样例」载入任意一张内置演示图，或上传本地 JPG/JPEG/PNG/WebP 图片；
2. 点击「**AI 智能识别**」；
3. 查看识别结果、绘制标注、修改等级与整改建议、导出报告。

#### Linux 源码启动

需要 Node.js ≥20.19.0、Python 3.12。以下命令均在项目根目录执行；首次部署安装依赖：

```bash
npm ci
python3.12 -m venv backend/python/.venv
(cd backend/python && .venv/bin/python -m pip install -r requirements.txt)
```

Linux 需要自己的虚拟环境，不能复用 Windows 的 `.venv/Scripts`。聚类使用的本地模型还需按 3.4 节准备，启动命令不会自动下载模型。

```bash
# 开发模式：前端 + Node API + Python 聚类服务
npm run dev:linux
# 浏览器访问 http://localhost:5175 或 http://<Linux主机IP>:5175
```

开发前端监听 `0.0.0.0:5175`，Python 默认监听 `127.0.0.1:8000`，浏览器通过前端代理调用聚类接口。按 Ctrl+C 停止整组服务；已有服务占用这些端口时，先停止原有启动命令。

```bash
# 生产模式：先构建，再启动静态页面/API 和 Python 聚类服务
npm run build
npm run start:linux
# 浏览器访问 http://<Linux主机IP>:3001
```

两种命令均以前台进程运行。若使用独立 Python 环境，可指定解释器，例如 `PY_CLUSTER_PYTHON=/opt/hazard-venv/bin/python npm run dev:linux`；也支持 PATH 中的命令名，如 `python3.12`。未设置时自动选择 `backend/python/.venv/bin/python`，无需手动 activate。

已有完整离线 Docker 演示包的 Linux 机器，可在演示包目录执行 `bash start.sh`；其镜像归档需由 `deploy/demo-v4/package.ps1` 打包生成，仅克隆源码不包含该归档。

### 3.4 启用聚类分析（可选，独立 Python 服务）

聚类分析依赖独立的 Python 服务，未启动时**不影响**其余功能，聚类页面会明确提示「服务未就绪」。

> 下面三步中的 `cd` 均以**项目根目录**为起点。

**第 1 步：准备 Python 环境**（Python 3.12）

```bash
cd backend/python
python -m venv .venv

# Windows
.venv/Scripts/pip install -r requirements.txt
# Linux / macOS
.venv/bin/pip install -r requirements.txt
```

用 `uv` 会快很多（本项目开发时用的就是它）：

```bash
cd backend/python
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS 换成 .venv/bin/python
```

> 如果环境装不上 torch，可以只装纯向量所需的依赖：
> `pip install -e "./cluster-engine[api,algorithms]"`
> 此时检索增强（nr1）与需要本地模型的 profile 会不可用，`/api/clustering/profiles` 会逐条标出原因。

**第 2 步：下载本地向量模型**

模型约 1.3 GB，**未随仓库分发**（已在 `.gitignore` 中排除），需要拉到 `backend/python/cluster-engine/models/bge-large-zh-v1.5/`。在 `backend/python/cluster-engine` 目录下、激活 venv 后执行：

```bash
cd backend/python/cluster-engine
python -m retrain_cluster download-model \
  --source BAAI/bge-large-zh-v1.5 --revision main \
  --destination models/bge-large-zh-v1.5
```

引擎会用 SHA-256 校验模型完整性，清单内联在 `configs/models.toml` 的 `file_checksums` 字段。若下载后校验不通过，重新生成清单并替换该字段：

```bash
python -m retrain_cluster model-manifest \
  --directory models/bge-large-zh-v1.5 \
  --output models/bge-large-zh-v1.5/manifest.json
```

> 缺少本地模型时接口返回 `MODEL_UNAVAILABLE`，`/api/clustering/health` 与「聚类测试」页会显示模型未就绪。`download-model` 拒绝覆盖已存在的目录，重下需先手动清理。

**第 3 步：启动**

```bash
cd ../../..           # backend/python/cluster-engine → 项目根目录（三层）
npm run dev:all       # Node API + Vite + Python 聚类服务
```

也可以只启动聚类服务：`npm run dev:py`（默认 `http://127.0.0.1:8000`）。

首次启动会校验本地向量模型，需要数十秒；期间「聚类测试」页的状态会显示为不可用，点「重新校验」即可。

### 3.5 生产模式与 Docker

```bash
npm run build     # 类型检查 + 前端构建到 dist/
npm start         # NODE_ENV=production，Express 同时托管 API 与静态页面
# 访问 http://localhost:3001
```

生产模式下 Express 会把 `/api/clustering/*` 转发到 Python 聚类服务（见 `server/http/clustering.routes.ts`），因此部署时同样需要单独运行该服务；目标地址可用 `PY_CLUSTER_URL` 覆盖。

也可以直接用仓库根目录的 `Dockerfile` 构建单容器镜像（多阶段：构建前端 → 运行时用 `tsx` 执行 TS 服务端源码）：

```bash
docker build -t hazard-insight-hub .
docker run -d -p 3001:3001 \
  -e MULTIMODAL_API_BASE_URL=https://api.deepseek.com/v1 \
  -e MULTIMODAL_API_KEY=<your-key> \
  -e MULTIMODAL_MODEL=deepseek-v4-flash-vision-exp \
  hazard-insight-hub
# 访问 http://localhost:3001
```

> 镜像**只包含 Node 侧**（EXPOSE 3001），不含 Python 聚类服务与 1.3 GB 向量模型；需要聚类功能时得另外部署 `backend/python`，并通过 `PY_CLUSTER_URL` 指向它。
> 运行时用 `tsx` 直接执行 TS 源码，所以镜像里保留了 devDependencies，体积偏大；若要精简需先把服务端也预编译成 JS。

### 3.6 npm 脚本总览

**开发与构建**

| 脚本 | 说明 |
| --- | --- |
| `npm run dev` | Node API(3001) + Vite(5175) |
| `npm run dev:all` | 追加 Python 聚类服务(8000) |
| `npm run dev:linux` | Linux 开发：三服务一起启动，Vite 监听 0.0.0.0:5175 |
| `npm run dev:api` | 仅 Node API（`tsx server/index.ts`） |
| `npm run dev:web` | 仅 Vite 前端 |
| `npm run dev:py` | 仅 Python 聚类服务 |
| `npm run build` | `tsc --noEmit` + `vite build` → `dist/` |
| `npm start` | 生产模式启动（`NODE_ENV=production`） |
| `npm run start:linux` | Linux 生产：已构建页面/API + Python 聚类服务（先运行 build） |
| `npm run preview` | Vite 预览已构建产物（不经过 Express） |

**检查与测试**

| 脚本 | 说明 |
| --- | --- |
| `npm run typecheck` | 前端 + Node 侧类型检查（`tsc --noEmit`） |
| `npm test` | Node 端全部测试（backend + frontend，`tsx --test`） |
| `npm run test:grading` | 仅定级测试（`test/backend/grading.test.ts`） |
| `npm run test:hazard` | 仅隐患测试（`test/backend/hazard-test.test.ts`） |
| `npm run test:py` | Python 单元测试（`pytest`，integration 默认跳过） |
| `npm run test:py -- -m integration` | Python 端到端（需本地向量模型，会真实跑一次聚类） |
| `npm run evaluate:grading` | 定级文本阶段基准评测 |

### 3.7 辅助脚本

**`scripts/clustering/py.mjs`** —— Python 解释器探测与统一入口。npm 脚本里没法同时写对 Windows 的 `.venv/Scripts/python.exe` 和 Linux/macOS 的 `.venv/bin/python`，所以由它做一次探测，让上面几个 `dev:py` / `test:py` 保持跨平台：

```bash
node scripts/clustering/py.mjs serve    # 启动 FastAPI（uvicorn，端口取 PY_CLUSTER_PORT，默认 8000）
node scripts/clustering/py.mjs test     # 运行 pytest
node scripts/clustering/py.mjs <args>   # 其余参数原样透传给该解释器
```

解释器查找顺序：`PY_CLUSTER_PYTHON` 环境变量（绝对路径或 PATH 命令名）→ 当前平台的项目虚拟环境（Windows 为 `backend/python/.venv/Scripts/python.exe`，Linux/macOS 为 `backend/python/.venv/bin/python`）→ PATH 中的 `python3`。显式指定的解释器不可用时会直接报错，不静默换环境。

**`scripts/grading/`** —— 定级质量评估工具链。多为一次性/离线分析脚本，依赖 `.tmp/grading-evaluation/` 下的中间产物（该目录已 gitignore），需要先跑前置步骤才会有数据：

| 脚本 | 说明 |
| --- | --- |
| `evaluate.ts` | 文本阶段基准评测。用法 `tsx scripts/grading/evaluate.ts before\|after [limit] [concurrency]`；标签与复核后字段不进入 prompt，改生产 prompt 前必须先冻结 before 一次 |
| `summarize.ts` | 汇总评测批次结果，与冻结基线做对比 |
| `probe.ts` | 探测当前 provider 连通性并打印一次 chat 调用结果 |
| `baseline/grade.ts` | 冻结的数值定级基线，**仅作评测基准，生产代码不引用** |
| `profile_data.py` | 只读抽取 BIFF `.xls` 台账文本（需 `xlrd`） |
| `xls_image_extract.py` | 从 BIFF8 `.xls` 按 ClientAnchor 建立「行号 → 内嵌图片」映射 |
| `export_test_set.py` | 分层抽样生成 `datas/test/隐患抽样_抽样50条.xlsx`（B 10 / C 20 / D 20，全部带图） |
| `analyze_quality.py` | 标签一致性审计（近重复隐患描述检测） |
| `inspect_drawings.py` | 检查工作簿内嵌绘图结构 |

## 四、环境变量

复制 `.env.example` 为 `.env` 并配置真实模型（未配置时无法进行识别）：

```env
# OpenAI 兼容多模态接口
MULTIMODAL_API_BASE_URL=https://api.example.com/v1
MULTIMODAL_API_KEY=
MULTIMODAL_MODEL=vision-model-name

# 可选
PORT=3001
MULTIMODAL_JSON_MODE=0        # 置 1 时请求 response_format=json_object（部分供应商支持）
MULTIMODAL_TIMEOUT_MS=120000  # 请求超时
MULTIMODAL_CONTEXT_WINDOW=    # 可选；模型 Context Window，未知留空
MULTIMODAL_INPUT_PRICE_PER_MILLION=   # 可选；实际输入每百万 Token 价格
MULTIMODAL_OUTPUT_PRICE_PER_MILLION=  # 可选；实际输出每百万 Token 价格
MULTIMODAL_CACHE_READ_PRICE_PER_MILLION=  # 可选；缓存读取价格
MULTIMODAL_CACHE_WRITE_PRICE_PER_MILLION= # 可选；缓存写入价格
MULTIMODAL_PRICE_CURRENCY=CNY
```

> **界面直填换模型，无需改文件**：打开「模型配置」，直接填写 API 地址、API Key 与模型名称并保存，立即生效；配置持久化到项目根 `.runtime-model.json`（已 gitignore）。支持测试连接、切换配置及恢复 `.env` 默认。
>
> **颜色主题**：顶栏「主题」按钮或「系统设置 → 颜色主题」切换 5 套 Apple 风格主题（经典蓝 / 海洋青 / 石墨灰 / 能量绿 / 深空深色），选择持久化到浏览器。

### 聚类服务相关变量

聚类服务有自己的一层配置（`backend/python`，前缀 `HAZARD_`，可用 `.env` 或环境变量覆盖），Node 侧只需一个转发地址：

| 变量 | 位置 | 默认 | 说明 |
| --- | --- | --- | --- |
| `PY_CLUSTER_URL` | Node 进程 | `http://127.0.0.1:8000` | Express 生产模式下转发 `/api/clustering/*` 的目标地址 |
| `VITE_PY_CLUSTER_TARGET` | 构建期 | `http://127.0.0.1:8000` | 开发模式 Vite 代理目标 |
| `PY_CLUSTER_PORT` / `PY_CLUSTER_HOST` | `npm run dev:py` | `8000` / `127.0.0.1` | Python 服务监听地址 |
| `HAZARD_TIMEOUT_SECONDS` | Python | `120` | 单次聚类的墙钟超时，超过返回 504 |
| `HAZARD_UPLOAD_MAX_BYTES` | Python | `10485760` | 上传数据文件大小上限 |
| `HAZARD_DEFAULT_PROFILE_ID` | Python | 空 | 指定默认 profile；留空则由网关自动挑选 |
| `HAZARD_CORS_ORIGINS` | Python | `5175 / 3001` 本地来源 | 直连（不经代理）调试时的允许来源，逗号分隔 |

> 前端**不硬编码**任何 Python 地址：开发走 Vite 代理、生产走 Express 转发，两条路径都只暴露同源 `/api/clustering/*`。

### GitHub OAuth 2.0 登录

1. 在 GitHub **Settings → Developer settings → OAuth Apps → New OAuth App** 创建应用；
2. 开发环境填写 Homepage URL `http://localhost:5175`，Authorization callback URL `http://localhost:5175/api/auth/github/callback`；
3. 在 `.env` 配置以下值并重启服务：

```env
GITHUB_OAUTH_CLIENT_ID=你的Client ID
GITHUB_OAUTH_CLIENT_SECRET=你的Client Secret
SESSION_SECRET=至少32字符的高强度随机字符串
APP_BASE_URL=http://localhost:5175
```

配置生效后，未登录访问会停留在独立登录页；登录成功后才会挂载工作台。除 `/api/auth/*` 外的业务 API 同样要求有效会话，不能通过绕过前端直接调用。服务端使用 Authorization Code + PKCE，登录只申请 `read:user`；Client Secret 和用户 access token 不会发送到前端。生产环境请把 Homepage、callback 与 `APP_BASE_URL` 一并改为 HTTPS 正式域名。

## 五、运行模式

| 模式 | 触发条件 | 行为 |
| --- | --- | --- |
| **AI 模式** | auto 下 `MULTIMODAL_API_KEY` 已配置，或显式选择 AI | 真实调用多模态模型完成 Agent 工作流 |

系统仅支持真实模型识别。未配置模型时返回配置错误；视觉模型调用失败时报告错误，不再返回内置演示识别结果。无 Key 的内网网关可在「模型配置」中显式选择 AI 模式。

> 多模态 Provider 已抽象为 `MultimodalModelProvider`（`vision / chat / healthCheck`），位于 `server/providers/`，可扩展任意 OpenAI 兼容或自研视觉服务。

## 六、项目结构

```text
hazardInsight-hub/
├── shared/                  # 前后端共享：类型 / 分级定义 / 报告 HTML 生成
│   ├── types.ts
│   ├── agent-protocol.ts
│   ├── grading-meta.ts
│   ├── clustering.ts        # 聚类接口契约（与 Python Pydantic 逐字段对齐）
│   └── report.ts
├── server/
│   ├── index.ts             # Express API（analyze/standards/cases/export/model/rules 等）
│   ├── agent/               # Agent Runtime：状态机、Context/Cache/Memory、Tools、Trace、Hazard Workflow
│   ├── pipeline.ts          # 旧调用兼容门面，内部转 Agent Runtime
│   ├── lib/                 # .env 加载 / Zod Schema / 系统 Prompt
│   ├── http/                # 路由层；clustering.routes.ts 为 Python 聚类服务的反向代理
│   ├── retrieval/           # 标准与历史案例检索（KnowledgeRetriever）
│   ├── grading/             # A/B/C/D 定级规则与护栏
│   ├── rules/               # 判定规则 TXT/CSV 解析与持久化
│   ├── providers/           # 多模态 Provider：openai-compatible + 工厂
│   └── data/                # 知识库 JSON + 规则演示样本（sample_rules.txt / sample_rules.csv）
├── backend/python/          # Python 聚类服务（独立进程，非主项目运行时依赖）
│   ├── app/                 # 网关：契约(schemas) / 服务(services) / 路由(api) / 预处理(processors)
│   ├── cluster-engine/      # 聚类引擎：11 种算法、20 个 profile、本地 BGE 向量模型、Optuna 调参
│   ├── samples/             # 内置示例数据（由 scripts/build_sample_dataset.py 从真实隐患抽样生成）
│   ├── scripts/             # 离线脚本（示例数据构建等）
│   └── tests/               # pytest（单元测试 + 默认跳过的 integration 端到端）
├── web/
│   ├── pages/               # 智能识别 / 隐患发现 / 隐患防控 / 定级规则 / 知识图谱 / 知识库 / 记录 / 设置 / 聚类展示 / 聚类测试 / 偏差数据库
│   ├── components/          # ImagePanel、ImageAnnotator、HazardCard、EvidenceDrawer、ClusterScatter…
│   ├── lib/                 # API 客户端、localStorage（记录/台账/主题）、发现设备源、图谱布局、图片工具、hash 路由
│   ├── api/                 # 领域接口客户端（clustering.ts 等，统一拆封 success/data/error）
│   └── styles.css           # 设计系统 + 多主题
├── scripts/
│   ├── clustering/py.mjs    # 跨平台 Python 启动/测试入口（解释器探测）
│   └── grading/             # 定级质量评估工具链（基准评测、样本生成、标签审计）
├── datas/                   # 隐患抽样数据与演示图片
├── test/                    # Node 端测试（backend / frontend）
├── docs/                    # 架构设计与实现说明
├── index.html
├── vite.config.ts
├── tsconfig.json
└── .env.example
```

## 七、核心接口

| 接口 | 说明 |
| --- | --- |
| `POST /api/analyze` | `multipart/form-data`：`image=<file>`（可选 `scenario=<id>`）→ `AnalysisResult` |
| `POST /api/agent/runs/stream` | `multipart/form-data` 创建 Agent run，以 SSE 返回真实 workflow 事件与最终结果 |
| `GET /api/agent/runs/:runId` / `GET /api/agent/runs/:runId/events` | 查询 run 状态/结果与补取事件 |
| `GET /api/agent/traces/:traceId` | 查询本进程内 trace、step、tool、retrieval、cache 与 fallback 记录（Demo） |
| `GET /api/agent/runs/:runId/token-usage` | 查询单 Run 的聚合及每一次 LLM Call usage |
| `GET /api/agent/sessions/:sessionId/token-usage` / `GET /api/agent/conversations/:conversationId/token-usage` | Session / Conversation 累计 Token Usage |
| `GET /api/agent/token-usage?conversationId=&sessionId=` | Token 统计页面所需三级总览 |
| `GET /api/agent/context-observability?conversationId=&sessionId=` | Agent / Turn / LLM Step 上下文统计总览（不含大段 item 原文） |
| `GET /api/agent/context-observability/steps/:stepId` | 按需读取历史 Step 的完整不可变 Context Snapshot |
| `GET /api/agent/context-observability/export/:sessionId` | 导出可解析 Session Context Log |
| `GET /api/standards?type=` | 标准知识库列表（general / nuclear / enterprise） |
| `GET /api/cases?q=&category=&grade=` | 历史案例检索 |
| `POST /api/export` | 接收 `{ analysis }`，返回自包含 HTML 报告 |
| `GET /api/model` / `POST /api/model-config` / `POST /api/model-config/reset` | 模型信息 / 保存运行期配置 / 恢复 .env 默认 |
| `GET /api/rules` / `POST /api/rules/import` / `POST /api/rules/import-demo` / `POST /api/rules/import-demo-csv` / `POST /api/rules/clear` | 判定规则库查询 / TXT·CSV 导入 / 载入 TXT 演示样本 / 载入 CSV 演示样本 / 清空 |
| `GET /api/health-check` | Provider 连通性测试 |
| `GET /api/clustering/health` | 聚类服务健康检查：统一包裹，`data.engine` 报告引擎与本地向量模型就绪状态（顶层另有 `status` 供探针读取） |
| `GET /api/clustering/profiles` | 可用 profile 列表（20 个 profile 的算法、模型、样本上限与不可用原因）；`?refresh=true` 强制重新校验 |
| `GET /api/clustering/algorithms` | 引擎暴露的聚类算法及其依赖可用性 |
| `GET /api/clustering/sample` | 内置示例数据（真实核电工程隐患抽样 50 条） |
| `POST /api/clustering/datasets` | `multipart/form-data`：`file=<CSV/XLSX/JSON/TXT>` → 解析为聚类样本（自动识别文本列、ID 去重、文本清洗） |
| `POST /api/clustering/run` | 执行一次聚类 → 统计 + 簇摘要（关键词/代表样本/元数据分布）+ 逐条归属 + 二维 PCA 坐标；`options.knowledgeBaseId` 可覆盖检索增强 profile 的知识库 |
| `GET /api/clustering/knowledge-bases` | 偏差数据库列表（条数、维度、生成方式、创建时间、大小与是否有条目快照） |
| `GET /api/clustering/knowledge-bases/{id}` | 知识库详情 + 分页条目（`?offset=&limit=`）；`entriesSource` 说明条目从快照/语料/历史来源读到，或不可用 |
| `POST /api/clustering/knowledge-bases` | `multipart/form-data`：`file=<TXT/CSV/XLSX/JSON>`、`name`、`mode=purified\|raw`、可选 `ident`/`model_id` → 提交异步生成任务，返回 `jobId` |
| `GET /api/clustering/knowledge-bases/jobs` / `GET /api/clustering/knowledge-bases/jobs/{jobId}` | 知识库生成任务列表 / 轮询进度（`stage`、`processed/total`、`terminal`） |
| `GET /api/health` | Python 聚类服务的探针（与 `/api/clustering/health` 同源路由，便于规范与前端代理各取所需） |

## 八、API 数据结构要点

- 图片识别结果：结构化 JSON（`AnalysisResult`），每条隐患包含
  `title / category / description / confidence / bbox(0~1 归一化) / evidence / possibleConsequence / grade / gradeReason / severityScore / probabilityScore / riskScore / standardReferences / historicalReferences / rectification / manualReviewRequired`；
- v2 结果新增可选 `ruleReferences` 与 `agentMeta`，旧浏览器记录仍可兼容；
- bbox 使用 **0~1 归一化坐标**，前端按原图比例绘制方框；
- 模型输出一律经 **Zod 校验 + 容错**，无效数据不得进入前端；
- 全部拟引用法规/条款/案例均为**从知识库检索的真实条目**，推理层只允许按 key 引用，禁止编造。

## 九、数据声明

> **项目内法规、企业制度与历史案例均属于演示 Mock 数据，不代表任何真实核电项目的正式受控文件。**
> 生产使用前必须替换为经过审核且有效的企业法规标准库，并接入真实历史案例库、对 AI 定级建立专业人员复核与签字流程。

所有页面与导出报告均带有上述声明提示。

## 十、聚类分析模块

### 1. 架构与职责边界

聚类能力**不在 TypeScript 里重写**，而是原样复用已有的 Python 聚类引擎，由一层很薄的 FastAPI 网关把它包装成本项目的接口契约：

```text
web/pages/Clustering*.tsx
  → web/api/clustering.ts            （同源 /api/clustering/*，统一拆封 success/data/error）
    → Vite 代理（开发） / Express 转发（生产）
      → backend/python/app           （FastAPI 网关：契约校验、文件解析、结果加工）
        → backend/python/cluster-engine（算法、向量化、检索增强、降维 —— 唯一算法来源）
```

| 层 | 做什么 | 不做什么 |
| --- | --- | --- |
| 前端页面 / 客户端 | 取数、渲染、交互、错误提示 | 不复现任何聚类或向量化逻辑 |
| FastAPI 网关 | 契约校验、上传解析、调用引擎、把结果加工成「统计 / 簇摘要 / 二维坐标」 | 不实现算法、不调超参 |
| cluster-engine | 向量化、特征融合、11 种聚类算法、Optuna 调参、产物落盘 | 不关心本项目的页面与文案 |

二维坐标（PCA）与簇摘要（代表样本 / c-TF-IDF 关键词 / 元数据分布）明确标注为**展示辅助**，不参与聚类决策 —— 簇划分与噪声判定完全以引擎输出为准。

### 2. 算法与配置档

引擎内置 11 种聚类算法，其中 10 种通过 API 暴露（`mean_shift` 仅 CLI）：

`agglomerative`、`hdbscan`、`radbscan`、`optics`、`affinity_propagation`、`dbscan`、`chinese_whispers`、`birch`、`leader`、`canopy`（+ CLI-only `mean_shift`）。

每个算法各有「纯向量（nr0）」与「检索增强（nr1，先用 ChromaDB 检索近邻再融合特征）」两类 profile，共 20 个。前端可选「自动选择」，由网关按「纯向量优先 + 算法偏好顺序」挑一个当前可执行的 profile；也可显式指定 `algorithm` 或 `profileId`。**参数由 profile 决定**，网关不虚构 `n_clusters` 之类的超参。

### 3. 两个页面的分工

| 页面 | 路由 | 面向 | 内容 |
| --- | --- | --- | --- |
| 聚类展示 | `/clustering/overview` | 演示 / 汇报 | 一键对示例或上传数据聚类；统计卡、二维散点图（可点击图例高亮簇）、簇卡片（关键词 / 元数据分布 / 代表样本）、样本归属明细与详情抽屉 |
| 聚类测试 | `/clustering/test` | 工程验证 | 服务与引擎就绪状态、算法与 profile 可用性清单、手工输入小批量文本做冒烟测试、单次运行逐条结果 + CSV 导出、**多算法横向对比**（簇数 / 噪声 / 耗时 / 向量缓存） |
| 偏差数据库 | `/clustering/knowledge-base` | 知识库运维 | 查看现有检索知识库与条目、上传语料由大模型净化（或原文）生成新库并轮询进度；聚类页据此选择本次使用的库 |

两页共享 `web/lib/useClusterEngine.ts`（只读元信息），各自独立持有执行状态；簇颜色由 `web/lib/clusterColor.ts` 按 `cluster_id` 稳定分配，保证散点图、图例、卡片、表格同色。

### 4. 偏差数据库（检索增强知识库）

检索增强（nr1 / `spear_purified_retrieval`）依赖一个**语义知识库**：把一批偏差文本编码成向量写入 ChromaDB，聚类时按近邻检索补偿表示。「偏差数据库」页面（`/clustering/knowledge-base`）把它的生命周期搬到界面上：

| 能力 | 说明 |
| --- | --- |
| 查看现有库 | 列出 `cluster-engine/artifacts/knowledge_bases/<id>/` 下的全部库（条数、维度、生成方式、创建时间、大小），并分页查看库内条目 |
| 上传语料生成 | `TXT`（每行一条）/ `CSV` / `XLSX` / `JSON`；`purified` 模式由本地 Qwen 逐条语义净化后入库，`raw` 模式原文直接入库；提交后异步执行、页面轮询进度 |
| 聚类时选择 | 「聚类展示 / 聚类测试」页在所选 profile 为检索增强时出现「偏差数据库」下拉，把 `knowledgeBaseId` 随请求下发，覆盖 profile 的默认库 |

工程约定：

- 知识库本体写在 **cluster-engine** 的 `artifacts/knowledge_bases/<id>/`（`manifest.json` + `index/` + `entries.jsonl`）；网关只存生成任务记录（`backend/python/artifacts/knowledge_bases/`）；
- 构建时额外落 `entries.jsonl`（写入索引的真实文本）与 `corpus.txt`（上传原文）旁路快照 —— ChromaDB 只存向量，没有快照就只能看到条目数、看不到内容；历史库若没有快照，页面会**如实说明**"只能看元信息"；
- 知识库标识必须匹配 `[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}`；中文名称写进清单的 `display_name`，标识按名称 slug 或时间戳自动分配；
- 生成是**异步作业**：Qwen 逐条净化两万条语料需要一到三小时，提交立刻返回 `jobId`，中途刷新页面仍能看到进度；服务重启会把遗留的"生成中"标记为失败，而不是让它永远转圈；
- 同一时刻只允许一个生成任务（GPU/显存约束），重复提交返回 429；
- 选择的知识库条目数少于 profile 声明的近邻数 `k` 时（如演示用小语料），本次请求会把 `k` 下调为条目数，并在结果告警里写明 —— 不静默改变检索口径；
- 纯向量 profile 不吃知识库：传入 `knowledgeBaseId` 会被显式拒绝（`INVALID_PROFILE`），而不是静默忽略。

### 5. 接口约定

- 所有接口返回统一包裹 `{ success, data, error }`；失败时 `error` 含 `code`（如 `UNSUPPORTED_FILE`、`PROFILE_UNAVAILABLE`、`SERVICE_BUSY`、`CLUSTERING_TIMEOUT`）与面向用户的中文 `message`，前端只需一套错误分支；
- Python 内部字段为 snake_case，对外统一序列化为 camelCase（Pydantic `alias_generator`），与 `shared/clustering.ts` 逐字段一致，前端不做任何键名转换；
- 网关为**单飞**模式：同一时刻只允许一个聚类请求进入引擎，其余请求返回 429，避免 CPU/内存被打满；
- 数据上限：单次 ≤ 30 万条样本、单条文本 ≤ 4000 字符、上传文件 ≤ 200 MB；聚类墙钟超时默认 3600 秒。
  - 注意：**放开个数限制不等于所有算法都能跑满**。层次聚类（agglomerative / affinity_propagation）是 O(n²) 时空复杂度，30 万条会内存溢出；实际能吃满量级的是 dbscan / hdbscan / birch / leader / canopy 等线性或近线性算法。

### 6. 运行与测试

环境准备、启动命令与全部 npm 脚本见「**三、快速开始**」的 3.4 / 3.6 小节，聚类相关的常用命令：

```bash
npm run dev:py    # 仅启动 Python 聚类服务
npm run test:py   # 运行 Python 单元测试（integration 默认跳过）
npm run test:py -- -m integration   # 端到端（需本地向量模型，会真实跑一次聚类）
```

`backend/python/tests/` 覆盖：契约（camelCase 双向兼容、强校验、自由字典键名不被改写）、文本清洗、CSV/JSON/TXT/XLSX 解析（含 GBK、BOM、损坏文件）、簇摘要（分组、置信度归一、代表样本、元数据分布、长度不一致报错）、示例数据与上传解析、以及端到端接口。

### 7. 来源与数据声明

- 聚类算法层来自既有的 Python 聚类工程（`retrain-cluster`），迁移为本项目的 `backend/python/cluster-engine`，**未重写算法**；网关与其契约层为本次新增；
- 内置示例数据由 `backend/python/scripts/build_sample_dataset.py` 从本仓库的隐患抽样 CSV 生成（50 条，含隐患级别 / 分类 / 排查类型 / 作业区域等业务字段），仅用于演示；
- 聚类结果为**无监督的统计分组**，簇标签（`类别 0`、`类别 1`…）不代表任何业务语义或定级结论，需人工解读。

## 十一、Roadmap（可选增强）

- 替换 `KnowledgeRetriever` 为 BM25 / 向量检索（Elasticsearch、Milvus、pgvector）；
- 接入企业受控标准库与版本有效性校验；
- 隐患记录对接整改闭环系统与审批流；
- 多图对比、视频抽帧识别等。

完整的现状审计、Agent Runtime/Workflow、Context Cache、Memory/RAG/Rules/Tools、SSE、可观测性与迁移设计见 [`docs/架构设计/Agent运行时与工作流架构设计.md`](docs/架构设计/Agent运行时与工作流架构设计.md)；更多文档见 [`docs/README.md`](docs/README.md)。

---

*演示工程 · 仅供技术交流与产品原型演示使用*
