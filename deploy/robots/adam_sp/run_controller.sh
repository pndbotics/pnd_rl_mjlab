#!/bin/bash
# 启动 Adam SP Controller.py 的脚本
# 使用方法: ./run_controller.sh
#
# 环境变量配置（可选，脚本会自动检测）:
#   CONDA_BASE_PATH      - Conda 安装路径 (默认: ~/miniconda3 或 ~/anaconda3)
#   CONDA_ENV_NAME       - Conda 环境名称 (默认: ros2_humble)
#   ROS2_DISTRO          - ROS2 发行版名称 (默认: humble)
#   ROS2_INSTALL_PATH    - ROS2 安装路径 (默认: /opt/ros/${ROS2_DISTRO})
#   PNDROBOT_WS_PATH     - pndrobotstatepub workspace 路径 (可选)
#   PROJECT_ROOT         - 项目根目录 (默认: 脚本所在目录)

set -e  # 遇到错误立即退出

# 获取脚本所在目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        print_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# 检查目录是否存在
check_dir() {
    if [ ! -d "$1" ]; then
        print_error "目录不存在: $1"
        return 1
    fi
    return 0
}

print_info "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 1. 设置 Conda
CONDA_BASE_PATH="${CONDA_BASE_PATH:-}"
if [ -z "$CONDA_BASE_PATH" ]; then
    # 自动检测 conda 安装位置
    if [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE_PATH="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE_PATH="$HOME/anaconda3"
    elif [ -d "/opt/conda" ]; then
        CONDA_BASE_PATH="/opt/conda"
    else
        print_error "无法找到 Conda 安装路径。请设置 CONDA_BASE_PATH 环境变量"
        exit 1
    fi
fi

CONDA_SH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"
if ! check_file "$CONDA_SH"; then
    print_error "Conda 初始化脚本不存在: $CONDA_SH"
    print_info "请设置正确的 CONDA_BASE_PATH 环境变量"
    exit 1
fi

print_info "使用 Conda: $CONDA_BASE_PATH"
source "$CONDA_SH"

# 激活 conda 环境
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ros2_humble}"
print_info "激活 Conda 环境: $CONDA_ENV_NAME"
if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
    print_error "无法激活 Conda 环境: $CONDA_ENV_NAME"
    print_info "请设置正确的 CONDA_ENV_NAME 环境变量，或创建该环境"
    exit 1
fi

# 2. Source ROS2
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
ROS2_INSTALL_PATH="${ROS2_INSTALL_PATH:-/opt/ros/$ROS2_DISTRO}"
ROS2_SETUP="$ROS2_INSTALL_PATH/setup.bash"

if ! check_file "$ROS2_SETUP"; then
    print_error "ROS2 setup 文件不存在: $ROS2_SETUP"
    print_info "请设置正确的 ROS2_INSTALL_PATH 或 ROS2_DISTRO 环境变量"
    exit 1
fi

print_info "Source ROS2 ($ROS2_DISTRO): $ROS2_INSTALL_PATH"
source "$ROS2_SETUP"

# 3. Source pndrobotstatepub (从pnd_deploy_ros2)
PNDROBOT_WS_PATH="${PNDROBOT_WS_PATH:-}"
if [ -z "$PNDROBOT_WS_PATH" ]; then
    # 尝试常见的路径
    COMMON_PATHS=(
        "$HOME/workplace/pnd_deploy_ros2/pndrobotros2/install"
        "$HOME/pnd_deploy_ros2/pndrobotros2/install"
        "$HOME/catkin_ws/install"
        "$HOME/ros2_ws/install"
        "./pnd_deploy_ros2/pndrobotros2/install"
    )
    
    for path in "${COMMON_PATHS[@]}"; do
        if [ -f "$path/setup.bash" ]; then
            PNDROBOT_WS_PATH="$path"
            break
        fi
    done
fi

if [ -n "$PNDROBOT_WS_PATH" ]; then
    PNDROBOT_SETUP="$PNDROBOT_WS_PATH/setup.bash"
    if check_file "$PNDROBOT_SETUP"; then
        print_info "Source pndrobotstatepub: $PNDROBOT_WS_PATH"
        source "$PNDROBOT_SETUP"
    else
        print_warn "pndrobotstatepub setup 文件不存在: $PNDROBOT_SETUP (跳过)"
    fi
else
    print_warn "未找到 pndrobotstatepub workspace，跳过 (可设置 PNDROBOT_WS_PATH 环境变量)"
fi

# 4. 检查 Controller.py 是否存在
if ! check_file "$PROJECT_ROOT/Controller.py"; then
    print_error "Controller.py 不存在于: $PROJECT_ROOT"
    exit 1
fi

# 5. 设置数据收集参数（可选）
# 如果设置了 COLLECT_DURATION，会在指定时间后自动保存数据
# 如果不设置，需要手动按B按钮保存数据
export ENABLE_REF_ROOT_LOGGING="${ENABLE_REF_ROOT_LOGGING:-true}"  # 启用日志收集（默认true）
# export COLLECT_DURATION=10  # 自动保存时间（秒），取消注释以启用

# 6. 运行 Controller
print_info "启动 Adam SP Controller.py..."
python "$PROJECT_ROOT/Controller.py"
