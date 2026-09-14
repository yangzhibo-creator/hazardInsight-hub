#!/usr/bin/env bash
# =============================================================================
# 下载离线 wheel（在联网/内网镜像源可达的构建机上执行一次）。
#
# 目标容器：cp311 / x86_64 / glibc 2.35（Ubuntu 22.04）。
#
# 两个必须照做的点：
#   1. **不要下载 torch / torchvision / torchaudio** —— 由基础镜像 h20-inference:v1.0
#      提供（2.8.0+cu129）；重复安装会白占数 GB 并可能破坏已验证的 CUDA 组合。
#   2. **平台标签要放宽到 manylinux_2_28** —— 现场锁定版本里 onnxruntime /
#      scikit-learn / greenlet / hf-xet / bcrypt 只发布了 manylinux_2_24/2_27/2_28
#      标签；只写 manylinux2014_x86_64 会让 pip 报 "No matching distribution"，
#      生成的 wheels/ 缺件，而缺件要到容器内 `pip install --no-index` 才暴露。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${HERE}/wheels}"
REQ_IN="${REQ_IN:-${HERE}/requirements.lock.txt}"
PYBIN="${PYBIN:-python3}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PLATFORM="${PLATFORM:-manylinux_2_28_x86_64}"
EXTRA_PLATFORM="${EXTRA_PLATFORM:-manylinux2014_x86_64}"

mkdir -p "${OUT_DIR}"

# 过滤掉 torch 三件套（锁定清单本来就没有，这里是防御性处理）
FILTERED="$(mktemp)"
trap 'rm -f "${FILTERED}"' EXIT
grep -vEi '^(torch|torchvision|torchaudio)([=<>!~].*)?$' "${REQ_IN}" > "${FILTERED}"

EXPECTED="$(grep -c '==' "${FILTERED}" || true)"
echo "锁定清单条目：${EXPECTED}；下载平台：${PLATFORM} + ${EXTRA_PLATFORM}"

"${PYBIN}" -m pip download \
  --dest "${OUT_DIR}" \
  --platform "${PLATFORM}" \
  --platform "${EXTRA_PLATFORM}" \
  --python-version 311 \
  --implementation cp \
  --only-binary=:all: \
  --index-url "${PIP_INDEX}" \
  -r "${FILTERED}"

ACTUAL="$(find "${OUT_DIR}" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')"
echo "wheel 数量：${ACTUAL}（期望约 ${EXPECTED} 项）"
du -sh "${OUT_DIR}"

if [ "${ACTUAL}" -lt "${EXPECTED}" ]; then
  echo "警告：wheel 数量少于清单条目，可能有个别版本没有 cp311 manylinux 轮子。" >&2
  echo "      请对照上面的 pip 输出确认缺了哪些，并在文档里记录处理方式。" >&2
  exit 1
fi

echo "完成。下一步：bash deploy/onsite-h20/build_image.sh"
