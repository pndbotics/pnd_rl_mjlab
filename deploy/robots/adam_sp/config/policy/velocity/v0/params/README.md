# Adam SP Velocity Policy Deployment Configuration

## 文件说明

`deploy.yaml` 文件包含了 Adam SP 速度跟踪策略部署所需的所有配置参数。这个文件提供了当 ONNX 模型的 `metadata_props` 不可用时的备选配置源。

## 参数来源

所有参数都从训练配置中提取：

- **Joint PD Gains (stiffness/damping)**: 来自 `mjlab/asset_zoo/robots/adam_sp/adam_sp_constant.py` 中的 actuator 配置
- **Action Scales**: 来自 `ADAM_SP_ACTION_SCALE`，计算公式：`scale = 0.25 * effort_limit / stiffness`
- **Default Joint Positions**: 来自 `joint_config.py` 中的 `DEFAULT_JOINT_POSITIONS`（与训练环境完全匹配）

## 关节顺序

配置文件中的关节顺序（23个观察关节，排除手腕关节）：

1. **左腿 (6个关节)**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
2. **右腿 (6个关节)**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
3. **腰部 (3个关节)**: roll, pitch, yaw
4. **左臂 (4个关节)**: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
5. **右臂 (4个关节)**: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow

**注意**: 手腕关节（wrist_yaw, wrist_pitch, wrist_roll）不包含在观察中，因此不在配置文件中。

## 使用方式

### 当前实现（Python ROS2）

当前的 `Controller.py` 实现主要从 ONNX 模型的 `metadata_props` 读取配置。`deploy.yaml` 文件作为：

1. **参考文档**: 提供参数值的参考
2. **未来扩展**: 可以修改 `Controller.py` 以支持从 YAML 文件读取配置（类似其他 C++ 部署实现）

### 与 C++ 实现的对比

- **adam_sp (Python ROS2)**: 从 ONNX `metadata_props` 读取配置，`deploy.yaml` 作为备选

## 参数说明

### Joint PD Gains

- **stiffness (Kp)**: 位置刚度，单位 N⋅m/rad
- **damping (Kd)**: 速度阻尼，单位 N⋅m⋅s/rad

### Action Scale

动作缩放因子，将模型输出的归一化动作转换为实际关节位置偏移（相对于默认位置）。

### Default Joint Positions

默认关节位置（弧度），用于：
- 初始化机器人姿态
- 作为动作的参考位置（动作是相对于默认位置的偏移）

## 更新配置

如果需要更新配置参数：

1. 确保参数值与训练配置一致
2. 如果修改了训练配置，需要重新生成此文件
3. 可以使用 `scripts/generate_deploy_yaml.py` 脚本自动生成（需要安装训练环境依赖）

## 验证

可以通过以下方式验证配置：

```bash
# 检查 YAML 文件格式
python3 -c "import yaml; yaml.safe_load(open('deploy.yaml'))"

# 检查参数数量（应该是23个关节）
python3 -c "
import yaml
with open('deploy.yaml') as f:
    cfg = yaml.safe_load(f)
    print('Stiffness:', len(cfg['stiffness']))
    print('Damping:', len(cfg['damping']))
    print('Action scale:', len(cfg['actions']['JointPositionAction']['scale']))
    print('Default pos:', len(cfg['default_joint_pos']))
"
```
