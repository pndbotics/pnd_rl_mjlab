#!/usr/bin/env bash
# 训练脚本：支持通过参数配置 GPU、motion_file、num_envs、logger 等，可后台运行并保存日志
# 用法示例:
#   ./scripts/train_tracking.sh
#   ./scripts/train_tracking.sh -b   # 后台运行，日志写入 logs/rsl_rl/run_logs/
#   ./scripts/train_tracking.sh -o new_dof_pos     # 运行目录和日志均用 new_dof_pos
#   ./scripts/train_tracking.sh -g 0 -m mjlab/motions/adam_sp/dance1_subject2.npz -n 4096 -l wandb

set -e

# 默认参数
TASK="${TASK:-Mjlab-Tracking-Flat-Adam-SP}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
MOTION_FILE="${MOTION_FILE:-mjlab/motions/adam_sp/dance1_subject2.npz}"
NUM_ENVS="${NUM_ENVS:-4096}"
LOGGER="${LOGGER:-wandb}"
RUN_BACKGROUND=false
RUN_NAME=""   # 为空则用时间戳/日期；指定则同时用作运行目录名和后台日志名
EXTRA_ARGS=()

# 解析命令行参数（短选项 + 长选项）
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    -m|--motion_file)
      MOTION_FILE="$2"
      shift 2
      ;;
    -n|--num_envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    -l|--logger)
      LOGGER="$2"
      shift 2
      ;;
    -t|--task)
      TASK="$2"
      shift 2
      ;;
    -b|--background)
      RUN_BACKGROUND=true
      shift
      ;;
    -o|--name|--log_name)
      RUN_NAME="$2"
      shift 2
      ;;
    -h|--help)
      echo "用法: $0 [选项] [额外的 tyro 参数...]"
      echo ""
      echo "选项:"
      echo "  -g, --gpu DEVICES      CUDA 可见设备，如 0 或 0,1 (默认: 0)"
      echo "  -m, --motion_file PATH motion npz 文件路径 (默认: mjlab/motions/adam_sp/dance1_subject2.npz)"
      echo "  -n, --num_envs N       并行环境数量 (默认: 4096)"
      echo "  -l, --logger NAME      日志后端，如 wandb 或 tensorboard (默认: wandb)"
      echo "  -t, --task TASK_ID     任务 ID (默认: Mjlab-Tracking-Flat-Adam-SP)"
      echo "  -b, --background       后台运行，stdout/stderr 写入 logs/rsl_rl/run_logs/"
      echo "  -o, --name NAME        运行名称，同时用于：运行目录 logs/rsl_rl/<exp>/NAME/ 和后台日志 run_logs/NAME.log"
      echo "  -h, --help             显示此帮助"
      echo ""
      echo "其余参数会原样传给 train.py（如 --video --video_length=100 等）。"
      echo ""
      echo "环境变量可覆盖默认值: TASK, CUDA_DEVICES, MOTION_FILE, NUM_ENVS, LOGGER, RUN_NAME"
      exit 0
      ;;
    *)
      # 剩余参数传给 python
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 控制台日志目录：与 train.py 的 log 根目录一致，放在 logs/rsl_rl 下
RUN_LOG_DIR="${REPO_ROOT}/logs/rsl_rl/run_logs"
if [[ -n "$RUN_NAME" ]]; then
  LOG_FILE_NAME="${RUN_NAME}"
  [[ "$LOG_FILE_NAME" == *.log ]] || LOG_FILE_NAME="${LOG_FILE_NAME}.log"
  RUN_LOG_FILE="${RUN_LOG_DIR}/${LOG_FILE_NAME}"
else
  RUN_LOG_FILE="${RUN_LOG_DIR}/train_$(date +%Y-%m-%d_%H-%M-%S).log"
fi

echo "[INFO] TASK=$TASK"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] num_envs=$NUM_ENVS"
echo "[INFO] logger=$LOGGER"
[[ -n "$RUN_NAME" ]] && echo "[INFO] run_name=$RUN_NAME"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[INFO] 额外参数: ${EXTRA_ARGS[*]}"
fi

if [[ "$RUN_BACKGROUND" == true ]]; then
  mkdir -p "$RUN_LOG_DIR"
  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') 后台训练启动 ==="
    echo "[INFO] TASK=$TASK"
    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
    echo "[INFO] motion_file=$MOTION_FILE"
    echo "[INFO] num_envs=$NUM_ENVS"
    echo "[INFO] logger=$LOGGER"
    [[ -n "$RUN_NAME" ]] && echo "[INFO] run_name=$RUN_NAME"
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[INFO] 额外参数: ${EXTRA_ARGS[*]}"
    echo "---"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    export PYTHONUNBUFFERED=1
    RUN_DIR_ARGS=()
    [[ -n "$RUN_NAME" ]] && RUN_DIR_ARGS=(--log-dir-name="$RUN_NAME")
    exec python scripts/train.py "$TASK" \
      --motion_file="$MOTION_FILE" \
      --env.scene.num-envs="$NUM_ENVS" \
      --agent.logger="$LOGGER" \
      "${RUN_DIR_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  } >> "$RUN_LOG_FILE" 2>&1 &
  TRAIN_PID=$!
  echo $TRAIN_PID > "${RUN_LOG_DIR}/.last_train.pid"
  echo "[INFO] 后台运行，日志: $RUN_LOG_FILE"
  echo "[INFO] 查看实时日志: tail -f $RUN_LOG_FILE"
  echo "[INFO] PID: $TRAIN_PID (已保存到 ${RUN_LOG_DIR}/.last_train.pid)"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
RUN_DIR_ARGS=()
[[ -n "$RUN_NAME" ]] && RUN_DIR_ARGS=(--log-dir-name="$RUN_NAME")
exec python scripts/train.py "$TASK" \
  --motion_file="$MOTION_FILE" \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.logger="$LOGGER" \
  "${RUN_DIR_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
