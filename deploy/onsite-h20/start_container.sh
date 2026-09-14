#!/usr/bin/env bash
# =============================================================================
# 现场一键启动：载入镜像（如需）→ 起容器 → 等 /api/health → 打印访问地址。
# 失败时直接打印容器日志，避免"命令没反应但不知道哪一步卡住"。
#
# 用法：
#   bash deploy/onsite-h20/start_container.sh                       # 默认端口 6006、GPU all
#   bash deploy/onsite-h20/start_container.sh --no-purify           # 不挂 Qwen，走规则净化
#   QWEN_HOST_DIR=/data/models/Qwen3.8-27B bash deploy/onsite-h20/start_container.sh
#
# 选项：--image --name --port --qwen-dir --gpus --tar --no-purify
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-spear-onsite-h20:v1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-spear-onsite}"
HOST_PORT="${HOST_PORT:-6006}"
CONTAINER_PORT=6006
#: Qwen 权重目录（约 55 GB）。挂到与 profile 相对路径一致的容器内路径。
QWEN_HOST_DIR="${QWEN_HOST_DIR:-/data/models/Qwen3.8-27B}"
QWEN_CONTAINER_DIR="/app/cluster-engine/models/Qwen3.8-27B"
GPU_DEVICES="${GPU_DEVICES:-all}"
TAR_PATH="${TAR_PATH:-}"
ENABLE_PURIFY="${ENABLE_PURIFY:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --port) HOST_PORT="$2"; shift 2 ;;
    --qwen-dir) QWEN_HOST_DIR="$2"; shift 2 ;;
    --gpus) GPU_DEVICES="$2"; shift 2 ;;
    --tar) TAR_PATH="$2"; shift 2 ;;
    --no-purify) ENABLE_PURIFY=0; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

command -v docker >/dev/null 2>&1 || { echo "未找到 docker。" >&2; exit 1; }

# ----------------------------------------------------------------------------- 镜像
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if [ -n "${TAR_PATH}" ] && [ -f "${TAR_PATH}" ]; then
    say "载入镜像 ${TAR_PATH}"
    docker load -i "${TAR_PATH}"
  else
    echo "本地没有镜像 ${IMAGE}。请先 docker load -i spear-onsite-h20.tar，或用 --tar 指定 tar 路径。" >&2
    exit 1
  fi
fi

# ----------------------------------------------------------------------------- Qwen 挂载
MOUNTS=()
if [ "${ENABLE_PURIFY}" -eq 1 ]; then
  if [ -d "${QWEN_HOST_DIR}" ]; then
    MOUNTS+=(-v "${QWEN_HOST_DIR}:${QWEN_CONTAINER_DIR}:ro")
  else
    echo "警告：未找到 Qwen 权重目录 ${QWEN_HOST_DIR}，净化将自动降级为规则兜底。" >&2
    echo "      如需现场演示 LLM 净化，请用 --qwen-dir 指向真实目录。" >&2
  fi
fi

# ----------------------------------------------------------------------------- 起容器
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  say "移除同名旧容器 ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  say "启动容器（GPU=${GPU_DEVICES}，端口 ${HOST_PORT}->${CONTAINER_PORT}）"
else
  echo "警告：宿主机未检测到 nvidia-smi；GPU 推理将不可用。" >&2
  say "启动容器（端口 ${HOST_PORT}->${CONTAINER_PORT}）"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus "${GPU_DEVICES}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${MOUNTS[@]}" \
  -e ENABLE_PURIFY="${ENABLE_PURIFY}" \
  -e DEVICE=cuda \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PORT="${CONTAINER_PORT}" \
  -e OUT_DIR=/app/output \
  "${IMAGE}" >/dev/null

# ----------------------------------------------------------------------------- 健康检查（最多 300 秒）
URL="http://127.0.0.1:${HOST_PORT}/api/health"
say "等待服务就绪：${URL}（最多 300 秒）"
READY=0
for i in $(seq 1 150); do
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "${URL}" >/dev/null 2>&1 && { READY=1; }
  else
    python3 - "${HOST_PORT}" <<'PY' >/dev/null 2>&1 && READY=1 || true
import sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=3) as response:
    sys.exit(0 if response.status == 200 else 1)
PY
  fi
  if [ "${READY}" -eq 1 ]; then
    echo "服务就绪（等待 ${i}×2 秒）。"
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" != "true" ]; then
    echo "容器已退出，最近日志：" >&2
    docker logs --tail 80 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  sleep 2
done

if [ "${READY}" -ne 1 ]; then
  echo "服务 300 秒内未就绪，最近日志：" >&2
  docker logs --tail 80 "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

# ----------------------------------------------------------------------------- 汇总
say "就绪"
echo "访问地址   ： http://127.0.0.1:${HOST_PORT}/"
echo "健康检查   ： ${URL}"
echo "聚类 API   ： http://127.0.0.1:${HOST_PORT}/api/clustering/profiles"
echo "离线基准   ： http://127.0.0.1:${HOST_PORT}/api/clustering/baseline"
echo "查看日志   ： docker logs -f ${CONTAINER_NAME}"
echo "停止服务   ： docker rm -f ${CONTAINER_NAME}"
echo
if command -v curl >/dev/null 2>&1; then
  curl -fsS "${URL}" | python3 -m json.tool 2>/dev/null || curl -fsS "${URL}" || true
  echo
fi
