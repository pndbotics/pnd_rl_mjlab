# Adam SP ROS2 部署说明

本目录包含 Adam SP 机器人通过 ROS2 与 `pnd_deploy_ros2` 通信的部署代码。

## 目录结构

```
adam_sp/
├── Controller.py              # ROS2控制器主程序
├── run_controller.sh          # 启动脚本
├── config/
│   ├── ros2.yaml              # ROS2控制器配置
│   └── policy/
│       └── velocity/
│           └── v0/
│               ├── exported/   # 放置训练好的ONNX模型
│               │   ├── policy.onnx
│               │   └── policy.onnx.data
│               └── params/
│                   └── deploy.yaml  # 部署参数配置
└── README.md                   # 本文件
```

## 前置要求

1. **ROS2 Humble** 已安装
2. **Conda 环境** `ros2_humble` 已创建并安装所需依赖
3. **pnd_deploy_ros2** 已编译并可以 source
4. **训练好的模型** 已导出为 ONNX 格式

## 安装步骤

### 1. 安装 ROS2 和依赖

参考 `pnd_deploy_python` 的安装脚本：
- 安装 ROS2 Humble
- 创建 conda 环境 `ros2_humble`
- 安装 Python 依赖（numpy, scipy, onnxruntime 等）

### 2. 配置 pnd_deploy_ros2 路径

设置环境变量（可选，脚本会自动检测）：
```bash
export PNDROBOT_WS_PATH=/path/to/pnd_deploy_ros2/pndrobotros2/install
```

### 3. 准备模型文件

将训练好的模型文件放置到：
```
config/policy/velocity/v0/exported/
├── policy.onnx
└── policy.onnx.data
```

### 4. 配置部署参数

编辑 `config/policy/velocity/v0/params/deploy.yaml`：
- **TODO**: 需要从训练好的模型中获取以下参数：
  - `joint_ids_map`: 关节ID映射
  - `stiffness`: 关节刚度
  - `damping`: 关节阻尼
  - `default_joint_pos`: 默认关节位置
  - `actions.JointPositionAction.scale`: 动作缩放因子
  - `actions.JointPositionAction.offset`: 动作偏移量

这些参数通常在导出 ONNX 模型时会自动生成，可以通过以下方式获取：
- 查看训练日志中的元数据
- 使用 `mjlab/rl/exporter_utils.py` 中的 `get_base_metadata` 函数
- 从训练配置文件中提取

## 使用方法

### 1. 启动 pnd_deploy_ros2

确保 `pnd_deploy_ros2` 正在运行并发布以下话题：
- `robot_state_actual` (RobotState)
- `imu_data` (Imu)

### 2. 启动控制器

```bash
cd deploy/robots/adam_sp
./run_controller.sh
```

### 3. 控制机器人

- **LB按钮**: 启用控制
- **B按钮**: 禁用控制

## 配置说明

### ros2.yaml

主要配置项：
- `model_pb_run`: ONNX 模型路径
- `observation_method`: 观测方法（"locomotion_run", "beyond_mimic", "opentrack"）
- `locomotion_run.velocity_commands`: 速度命令配置

### deploy.yaml

包含以下关键配置：
- 关节映射和PD参数
- 动作缩放和偏移
- 观测配置
- 命令范围

## 待完成事项

以下功能需要根据实际情况实现：

1. **观测处理器** (`ObservationProcessor`)
   - 需要从 `pnd_deploy_python` 复制或适配
   - 需要定义 Adam SP 的关节配置类

2. **动作处理器** (`ActionProcessor`)
   - 需要从 `pnd_deploy_python` 复制或适配
   - 处理模型输出到关节命令的转换

3. **模型管理器** (`ModelManager`)
   - 需要从 `pnd_deploy_python` 复制或适配
   - 加载和运行 ONNX 模型

4. **关节配置** (`JointConfig`)
   - 需要定义 Adam SP 的关节顺序、名称、映射关系
   - 参考 `pnd_deploy_python/mimic/joint_config.py`

5. **部署参数**
   - 从训练好的模型中提取实际的 `deploy.yaml` 参数
   - 确保关节顺序和映射正确

## 通信协议

控制器通过以下 ROS2 话题与 `pnd_deploy_ros2` 通信：

**订阅**:
- `robot_state_actual` (RobotState): 机器人状态（关节位置、速度、力矩）
- `imu_data` (Imu): IMU数据（姿态、角速度、线加速度）
- `joy` (Joy): 手柄输入

**发布**:
- `joint_state_cmd` (JointStateCmd): 关节命令（位置、速度、力矩）

## 故障排除

1. **找不到 pndrobotstatepub**
   - 设置 `PNDROBOT_WS_PATH` 环境变量
   - 或确保 `pnd_deploy_ros2` 在常见路径下

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确保 ONNX 模型文件完整

3. **关节映射错误**
   - 检查 `deploy.yaml` 中的 `joint_ids_map`
   - 确保与 ROS2 消息中的关节顺序一致

4. **控制不响应**
   - 检查 `pnd_deploy_ros2` 是否正在运行
   - 检查话题是否正确发布/订阅
   - 使用 `ros2 topic list` 和 `ros2 topic echo` 检查

## 参考

- `pnd_deploy_python`: Python ROS2 控制器参考实现
- `pnd_deploy_ros2`: ROS2 机器人状态发布器
- `mjlab/tasks/velocity/config/adam_sp/`: Adam SP 训练配置
