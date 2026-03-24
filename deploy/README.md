# Deploy 说明

本目录在仿真或真机上运行 Adam 策略：有限状态机（FSM）在 **被动 / 回零姿态 /  locomotion / Beyond Mimic** 等模式间切换，各模式参数由各自子目录下的 YAML 提供。

## 目录结构

| 路径 | 作用 |
|------|------|
| `common/` | 控制数据结构、数学工具、手柄抽象、`yaml_utils.load_yaml` |
| `FSM/` | 状态机与状态基类 |
| `adam_description/` | 机器人关节名、默认姿态、PD 增益等（`adam_sp.py`） |
| `policy/*/` | 各策略实现 + `config/*.yaml` + `model/`（权重） |
| `deploy_mujoco/` | MuJoCo 仿真入口与 `config/mujoco.yaml` |
| `deploy_real/` | 真机 DDS 入口与 `config/real.yaml` |

从 `deploy_mujoco` 或 `deploy_real` 启动时，脚本会把 `deploy/` 加入 `sys.path`，因此模块以 `common.*`、`FSM.*`、`policy.*` 等形式导入。

## MuJoCo 仿真

依赖：`mujoco`、`numpy`、`pyyaml`、策略所需 `torch` / `onnxruntime` 等。

```bash
cd deploy
python deploy_mujoco/deploy_mujoco.py
```

- 仿真与控制周期：`deploy_mujoco/config/mujoco.yaml` 中的 `xml_path` 为相对 **`deploy/`** 目录的路径（与 `common/path_config.PROJECT_ROOT` 拼接后加载 MuJoCo 模型）。
- 键盘：`7`/`8` 调节悬挂高度，`9` 开关悬挂力；站立类策略（locomotion、Beyond Mimic）会自动关闭悬挂，避免角色漂浮。

### 手柄映射（仿真）

| 按键 | 作用 |
|------|------|
| **B** 松开 | 被动模式 |
| **START** 松开 | 回固定姿态（插值到默认角） |
| **A** + **RB** 同时按下 | 被动 |
| **A** 松开 | Beyond Mimic（需配置好 `policy/adam_beyond_mimic` 下数据与 ONNX） |
| **Y** 松开 | Locomotion |
| **SELECT** 按下 | 退出 |
| 左摇杆 / 右摇杆 | 速度指令（用于 locomotion） |

## 真机（DDS）

需已安装并配置厂商 SDK（`pndbotics_sdk_py`）与 DDS 网络。

```bash
cd deploy
python deploy_real/deploy_real.py
```

连接与话题名、关节数、控制周期见 `deploy_real/config/real.yaml`。

### DDS 服务（systemd 示例）

```bash
sudo systemctl start pnd_service_dds.service    # 启动
sudo systemctl restart pnd_service_dds.service  # 重启
sudo systemctl stop pnd_service_dds.service     # 停止
journalctl -f -u pnd_service_dds.service        # 日志
```

### 遥控器进入控制

同时按下方向轮 **L0 + R0** 进入控制流程（与具体遥控器映射以 `common/remote_controller.py` 为准）。

| 按键 | 作用 |
|------|------|
| **START** | 固定姿态 |
| **A** | Beyond Mimic |
| **Y** | Locomotion |
| **B** | 被动 |
| **A** + **RB** | 被动 |
| **SELECT** | 退出主循环 |

## 策略配置速查

- **被动**：`policy/passive/config/Passive.yaml`
- **固定姿态**：`policy/fixedpose/config/FixedPose.yaml`
- **行走**：`policy/loco_mode/config/LocoMode.yaml`（`model/` 下放 TorchScript，如 `policy_29dof.pt`）
- **Beyond Mimic**：`policy/adam_beyond_mimic/config/adam_beyond_mimic.yaml`（参考运动 `.npz`、`model/` 下 ONNX；关节 PD 与默认角仍来自 `adam_sp.py`）

YAML 统一通过 `common/yaml_utils.load_yaml` 读取，便于后续扩展。

## 替换模型或动作

1. 将新权重放入对应策略的 `model/`。
2. 修改该策略目录下 `config/*.yaml` 中的 `policy_path`（或 Beyond Mimic 的 `motion_path` 等字段）。
3. 若观测维度或关节数变化，需同步修改策略代码与 YAML 中的 `num_actions` / 观测构造逻辑。

## 代码整理说明（近期）

- 去掉未使用的 `PROJECT_ROOT` 导入、重复的 PD 包装函数、仅注释调用的 `absoluteWait` 等死代码。
- Beyond Mimic 中未使用的观测缓存列表、未调用的睡眠与线程占位已删除；推理入口重命名为 `_run_policy_step`。
- `LocoMode.yaml` 中的 `tau_limit` 字段保留在文件中供参考，当前策略未做力矩裁剪；若需限幅可在 `LocoMode.run` 中自行接入。
