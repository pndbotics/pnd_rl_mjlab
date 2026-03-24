#!/bin/bash
# 播放 Adam-SP 跟踪策略（加载本地 checkpoint）
# 用法: ./scripts/play.sh [-g GPU] [-m MOTION_FILE] [-c CHECKPOINT_FILE] [-n NUM_ENVS] [-v]

# 默认值
GPU=0
TASK=Mjlab-Tracking-Flat-Adam-SP
MOTION_FILE=mjlab/motions/adam_sp/dance1_subject2.npz
# 请改为你的实际 run 目录下的 model_xxx.pt，例如: logs/rsl_rl/adam_sp_tracking/2026-02-24_12-00-00/model_5000.pt
CHECKPOINT_FILE=logs/rsl_rl/adam_sp_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
NUM_ENVS=
VIDEO=false

# 解析参数
while getopts "g:m:c:n:vh" opt; do
  case $opt in
    g) GPU="$OPTARG" ;;
    m) MOTION_FILE="$OPTARG" ;;
    c) CHECKPOINT_FILE="$OPTARG" ;;
    n) NUM_ENVS="$OPTARG" ;;
    v) VIDEO=true ;;
    h)
      echo "用法: $0 [-g GPU] [-m MOTION_FILE] [-c CHECKPOINT_FILE] [-n NUM_ENVS] [-v]"
      echo "  -g  显卡 ID (默认: 0)"
      echo "  -m  动作文件路径 (默认: mjlab/motions/adam_sp/dance1_subject2.npz)"
      echo "  -c  checkpoint 文件路径 (默认: 见脚本内 CHECKPOINT_FILE)"
      echo "  -n  环境数量 (可选，不传则用 checkpoint 对应配置)"
      echo "  -v  录制视频到 log 目录"
      exit 0
      ;;
    \?)
      echo "未知选项: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=$GPU

ARGS=(
  scripts/play.py "$TASK"
  --motion_file="${MOTION_FILE}"
  --checkpoint_file="${CHECKPOINT_FILE}"
)
[[ -n "$NUM_ENVS" ]] && ARGS+=(--num_envs="${NUM_ENVS}")
[[ "$VIDEO" == true ]] && ARGS+=(--video)

python "${ARGS[@]}"
