# =============================================================================
# deploy/onsite-h20 —— 离网现场演示镜像（**现场部署定制，不是上游功能**）
#
# 目标机：NVIDIA H20 / x86_64 / Ubuntu 22.04 / CUDA 12.9 / Docker 29.1.3
# 基础镜像：由平台提供 h20-inference:v1.0（PyTorch 2.8.0 + Python 3.11.15）
#
# 这个文件只服务于"现场离线单机演示"，与 cluster-engine 的上游能力无关。
# 上游合并时应当整目录排除（见 README.md 的提交边界说明）。
#
# 构建（在有网/内网的构建机上，见 build_image.sh）：
#   docker build -f deploy/onsite-h20/Dockerfile -t spear-onsite-h20:v1.0 .
#
# 想先用替代基础镜像验证结构而不需要真权重时，用 Dockerfile.structural。
# =============================================================================
ARG BASE_IMAGE=h20-inference:v1.0
FROM ${BASE_IMAGE}

# 离线强制：容器内绝不访问 HuggingFace / PyPI。必须在 import transformers 之前生效。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app/cluster-engine/src:/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf_home \
    TOKENIZERS_PARALLELISM=false \
    MODELS_DIR=/app/cluster-engine/models \
    EMBED_MODEL_PATH=/app/cluster-engine/models/bge-large-zh-v1.5 \
    LLM_MODEL_PATH=/app/cluster-engine/models/Qwen3.8-27B \
    OUT_DIR=/app/output \
    PORT=6006 \
    DEVICE=cuda

WORKDIR /app

# -----------------------------------------------------------------------------
# 1) 离线 Python 依赖
#
# 用 --no-deps：锁定清单本身就是完整闭包，且基础镜像已提供 torch；
# 若允许 pip 解析依赖，它会因为 --no-index 下找不到 torch 而直接失败（陷阱 P5）。
# -----------------------------------------------------------------------------
COPY deploy/onsite-h20/requirements.lock.txt /tmp/requirements.lock.txt
COPY deploy/onsite-h20/wheels/ /tmp/wheels/
RUN pip install --no-index --no-deps --find-links=/tmp/wheels -r /tmp/requirements.lock.txt \
 || pip install --no-index --no-deps --break-system-packages --find-links=/tmp/wheels -r /tmp/requirements.lock.txt \
 && rm -rf /tmp/wheels /tmp/requirements.lock.txt

# -----------------------------------------------------------------------------
# 2) 应用代码：聚类引擎 + 网关 + 组合入口（API + 前端静态资源）
# -----------------------------------------------------------------------------
COPY backend/python/cluster-engine /app/cluster-engine
COPY backend/python/app /app/app
COPY deploy/onsite-h20/app_entry.py /app/app_entry.py

# -----------------------------------------------------------------------------
# 3) 前端静态产物
#
# web-dist/ 在仓库里放了一份占位页面，保证结构验证时 COPY 一定成功；
# 真实构建由 build_image.sh 先用 npm 构建再覆盖（离网镜像内不放 Node）。
# -----------------------------------------------------------------------------
COPY deploy/onsite-h20/web-dist/ /app/web/

# -----------------------------------------------------------------------------
# 4) 资产：bge 向量模型（约 1.3 GB，打进镜像）与预置知识库
#
# Qwen3.8-27B（约 55 GB）**不进镜像**，运行时用 -v 只读挂载（否则 tar 包过大，
# 现场 docker load 会挤占 70 分钟预算）。
#
# assets/ 始终存在（仓库里带 README），真实资产由 build_image.sh 在构建前放入；
# 缺失时构建仍然成功，只是运行期会把对应 profile 报为不可用——这是有意的：
# 结构验证与"缺件构建"都不该因为一个 COPY 失败而整单失败。
# -----------------------------------------------------------------------------
COPY deploy/onsite-h20/assets/ /opt/onsite-assets/
RUN set -eux; \
    if [ -d /opt/onsite-assets/models/bge-large-zh-v1.5 ]; then \
        mkdir -p /app/cluster-engine/models; \
        cp -a /opt/onsite-assets/models/bge-large-zh-v1.5 /app/cluster-engine/models/; \
    fi; \
    if [ -d /opt/onsite-assets/knowledge_bases ]; then \
        mkdir -p /app/cluster-engine/artifacts/knowledge_bases; \
        cp -a /opt/onsite-assets/knowledge_bases/. /app/cluster-engine/artifacts/knowledge_bases/; \
    fi; \
    if [ -f /opt/onsite-assets/purify_map.json ]; then \
        mkdir -p /app/cluster-engine/data; \
        cp -a /opt/onsite-assets/purify_map.json /app/cluster-engine/data/; \
    fi; \
    rm -rf /opt/onsite-assets

# 运行期产物目录（引擎写 run manifest / 网关写作业与数据集）
RUN mkdir -p /app/output /app/artifacts

# HEALTHCHECK 只用 python：基础镜像不保证有 curl/wget
# start-period 给足 300 秒（模型清单校验 + 首轮加载都在这段时间内完成）
HEALTHCHECK --interval=30s --timeout=6s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:6006/api/health', timeout=5).status==200 else 1)"

EXPOSE 6006

CMD ["python", "-m", "uvicorn", "app_entry:app", "--host", "0.0.0.0", "--port", "6006"]
