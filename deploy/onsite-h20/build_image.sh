#!/usr/bin/env bash
# =============================================================================
# 构建现场演示镜像并导出 tar（在有网/内网的构建机上执行，不要在 H20 现场执行）。
#
# 用法：
#   bash deploy/onsite-h20/build_image.sh                 # 现场路径（需要真基础镜像 + 资产 + wheels）
#   bash deploy/onsite-h20/build_image.sh --structural    # 结构验证（python:3.11-slim，联网装依赖）
#
# 选项：
#   --assets DIR     从 DIR 读取 models/ knowledge_bases/ purify_map.json（默认 deploy/onsite-h20/assets）
#   --skip-web       不重新构建前端（沿用 web-dist/ 里的现有产物）
#   --no-save        只构建镜像，不导出 tar
#   --base IMAGE     覆盖基础镜像（默认 h20-inference:v1.0）
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
ASSETS_DIR="${HERE}/assets"
IMAGE="${IMAGE:-spear-onsite-h20:v1.0}"
OUT_TAR="${OUT_TAR:-spear-onsite-h20.tar}"
BASE_IMAGE="${BASE_IMAGE:-h20-inference:v1.0}"
STRUCTURAL=0
SKIP_WEB=0
SAVE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --structural) STRUCTURAL=1; IMAGE="${IMAGE_STRUCTURAL:-spear-onsite-structural:local}"; SAVE=0; shift ;;
    --assets) ASSETS_DIR="$2"; shift 2 ;;
    --skip-web) SKIP_WEB=1; shift ;;
    --no-save) SAVE=0; shift ;;
    --base) BASE_IMAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

command -v docker >/dev/null 2>&1 || { echo "未找到 docker，请在装有 Docker 的构建机上执行。" >&2; exit 1; }

# ----------------------------------------------------------------------------- 资产
if [ "${STRUCTURAL}" -eq 0 ]; then
  say "检查构建资产：${ASSETS_DIR}"
  if [ ! -d "${ASSETS_DIR}/models/bge-large-zh-v1.5" ]; then
    echo "缺少 ${ASSETS_DIR}/models/bge-large-zh-v1.5（约 1.3 GB，需进镜像）。" >&2
    echo "请先按 deploy/onsite-h20/assets/README.md 把交付介质里的模型拷进来。" >&2
    exit 1
  fi
  if [ ! -d "${ASSETS_DIR}/knowledge_bases/bge-large-raw-lines-v1" ]; then
    echo "警告：未找到预置知识库 ${ASSETS_DIR}/knowledge_bases/bge-large-raw-lines-v1。" >&2
    echo "      镜像仍可构建，但检索增强 profile 会在运行期报 KNOWLEDGE_BASE_UNAVAILABLE。" >&2
  fi
  if [ ! -d "${HERE}/wheels" ] || [ -z "$(ls -A "${HERE}/wheels" 2>/dev/null)" ]; then
    echo "缺少 ${HERE}/wheels（离线 wheel）。请先执行 make_wheels.sh。" >&2
    exit 1
  fi
  docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || {
    echo "本地不存在基础镜像 ${BASE_IMAGE}。请先 docker load -i h20-inference-v1.0.tar" >&2
    exit 1
  }
fi

# ----------------------------------------------------------------------------- 前端
if [ "${SKIP_WEB}" -eq 0 ] && [ "${STRUCTURAL}" -eq 0 ]; then
  say "构建前端产物（Vite）"
  if command -v npm >/dev/null 2>&1; then
    ( cd "${ROOT}" && npm ci --no-audit --no-fund >/dev/null && npm run build )
    rm -rf "${HERE}/web-dist"
    mkdir -p "${HERE}/web-dist"
    if [ -d "${ROOT}/dist" ]; then
      cp -a "${ROOT}/dist/." "${HERE}/web-dist/"
    else
      echo "npm run build 未产出 dist/，保留占位页面。" >&2
    fi
  else
    echo "未找到 npm，跳过前端构建（沿用 web-dist/ 现有内容）。" >&2
  fi
fi

# ----------------------------------------------------------------------------- 构建
if [ "${STRUCTURAL}" -eq 1 ]; then
  say "结构验证构建（${IMAGE}，基础镜像 python:3.11-slim）"
  docker build --pull -f "${HERE}/Dockerfile.structural" -t "${IMAGE}" "${ROOT}"
else
  say "现场构建（${IMAGE}，基础镜像 ${BASE_IMAGE}）"
  docker build --build-arg BASE_IMAGE="${BASE_IMAGE}" -f "${HERE}/Dockerfile" -t "${IMAGE}" "${ROOT}"
fi
docker images "${IMAGE}"

# ----------------------------------------------------------------------------- 自检
say "启动容器并自检 /api/health"
PORT="${SMOKE_PORT:-16006}"
CID="$(docker run -d -p "${PORT}:6006" "${IMAGE}")"
trap 'docker rm -f "${CID}" >/dev/null 2>&1 || true' EXIT
OK=0
for _ in $(seq 1 60); do
  if python3 - "${PORT}" <<'PY' >/dev/null 2>&1
import sys, urllib.request
port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as response:
    sys.exit(0 if response.status == 200 else 1)
PY
  then OK=1; break; fi
  sleep 2
done
if [ "${OK}" -ne 1 ]; then
  echo "自检失败：/api/health 未在 120 秒内返回 200，最近日志：" >&2
  docker logs --tail 80 "${CID}" >&2 || true
  exit 1
fi
python3 - "${PORT}" <<'PY' || true
import json, sys, urllib.request
port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
PY
docker rm -f "${CID}" >/dev/null
trap - EXIT

# ----------------------------------------------------------------------------- 导出
if [ "${SAVE}" -eq 1 ]; then
  say "导出 ${OUT_TAR} 并计算 sha256"
  docker save -o "${OUT_TAR}" "${IMAGE}"
  ls -lh "${OUT_TAR}"
  ( sha256sum "${OUT_TAR}" 2>/dev/null || shasum -a 256 "${OUT_TAR}" ) | tee "${OUT_TAR}.sha256"
  echo
  echo "现场介质清单：${OUT_TAR} + ${OUT_TAR}.sha256 + Qwen3.8-27B 权重目录 + 本 README"
fi
