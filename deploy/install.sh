#!/usr/bin/env bash
# 克隆并安装 pnd_sdk_python，以及指定版本的 numpy / onnxruntime / opencv-python。
# 若 cyclonedds 编译安装失败，请参考:
#   https://github.com/pndbotics/pnd_sdk_python
#   设置 CYCLONEDDS_HOME 后重试。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/pnd_sdk_python"
REPO_URL="https://github.com/pndbotics/pnd_sdk_python.git"

NUMPY_SPEC="numpy==1.25.0"
# OpenCV 4.10 系列
OPENCV_SPEC="opencv-python==4.10.0.84"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PIP=(python -m pip)
else
  PIP=(python3 -m pip)
fi

echo "==> 使用 pip: ${PIP[*]}"
"${PIP[@]}" install --upgrade pip

if [[ -d "${DEST}/.git" ]]; then
  echo "==> 已存在仓库，拉取更新: ${DEST}"
  git -C "${DEST}" pull --ff-only
else
  echo "==> 克隆 ${REPO_URL} -> ${DEST}"
  mkdir -p "$(dirname "${DEST}")"
  git clone --depth 1 "${REPO_URL}" "${DEST}"
fi

echo "==> 安装依赖: ${NUMPY_SPEC}, onnxruntime, ${OPENCV_SPEC}, cyclonedds==0.10.2"
"${PIP[@]}" install "${NUMPY_SPEC}" onnxruntime "${OPENCV_SPEC}" "cyclonedds==0.10.2"

echo "==> 以可编辑模式安装 pnd_sdk_python"
"${PIP[@]}" install -e "${DEST}"

echo "==> 完成。"
