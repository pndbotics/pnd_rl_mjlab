# 真机部署（deploy_real）

在 **PND Adam SP** 上通过 DDS 运行 **Beyond Mimic** ONNX 策略：订阅 `LowState`、发布 `LowCmd`，观测与 `deploy_mujoco` 中逻辑对齐（参考动作 + IMU + 关节误差等）。

当前主程序 **`deploy_real.py`** 与 **`config.py`** 仅支持 **`msg_type: "adam_sp"`**、**`imu_type: "pelvis"`**，**29 自由度** 部署。

---

## 依赖

- `numpy`、`pyyaml`、`onnxruntime`
- `pndbotics_sdk_py` 与 `cyclonedds`（见仓库根目录 [`install.sh`](../install.sh) 与 [pnd_sdk_python](https://github.com/pndbotics/pnd_sdk_python)）

不再使用 `torch` / `legged_gym` 加载策略；策略文件为 **ONNX**。

---

## 运行方式

```bash
cd <PROJECT_ROOT>/deploy/deploy_real
python deploy_real.py <网卡名> [配置文件名]
```

| 参数 | 说明 |
|------|------|
| `网卡名` | 与机器人通信的网卡，如 `enp3s0` |
| `配置文件名` | 位于 `configs/` 下，默认 `adam_sp.yaml` |

示例：

```bash
python deploy_real.py enp3s0 adam_sp.yaml
```

配置路径在代码中固定为：`<PROJECT_ROOT>/deploy/deploy_real/configs/<文件名>`，其中 `<PROJECT_ROOT>` 为 **`deploy` 的父目录**。

---

## 启动流程（与代码一致）

1. 扶稳或吊起机器人，网络与 DDS 按厂商说明配置好。
2. 启动程序，等待终端出现已连接 **LowState**。
3. 按遥控器 **Start**：程序在约 2 s 内将关节插值到配置中的 `default_angles`。
4. 按 **A**：进入策略控制循环（ONNX 推理 + PD 下发）。
5. 按 **B** 或 `Ctrl+C` 退出。

部署会直接下发关节目标与增益，请在安全环境下使用。

---

## 配置文件要点（`configs/*.yaml`）

- **`control_dt`**：控制周期（秒）。
- **`policy_path` / `motion_path`**：支持 `{PROJECT_ROOT}` 占位符（见仓库根目录 README）。
- **`joint2motor_idx`**：长度 29，与 `LowState` 电机顺序对应。
- **`kps` / `kds` / `default_angles`**：长度均为 29。
- **`num_actions`: 29，`num_obs`: 154**（与默认 Beyond Mimic 观测构造一致）。
- **`action_output_to_target_idx` / `motion_joint_indices`**：与训练及动作文件列顺序一致。
- **DDS**：`lowcmd_topic`、`lowstate_topic`（如 `rt/lowcmd`、`rt/lowstate`）。

更完整的字段说明与仿真侧对照见仓库根目录 [`README.md`](../README.md)。

---

## 观测（Beyond Mimic）

与 `deploy_real.py` 中 `build_obs` 一致，标量拼接为：

`motion_q(29) + motion_dq(29) + anchor_ori_b(6) + ang_vel(3) + q_offset(29) + dq(29) + last_action(29)` → **154** 维。

若 ONNX 另有时间/相位输入，由程序根据当前参考帧索引自动填充。

---

## 遥控键位（`common/remote_controller.py`）

- **Start**：开始回默认姿态插值。
- **A**：进入策略闭环。
- **B**：退出主循环。

Beyond Mimic 只依赖遥控器的按键状态；`remote_controller` 中的 **`get_walk_*_direction_speed`**（摇杆速度）等接口当前未接入观测，仍保留在模块内供其它用途。

---

## `common/` 说明

| 文件 | Beyond Mimic 当前使用 | 保留但未接入 BM 的接口 |
|------|------------------------|-------------------------|
| `command_helper.py` | `MotorMode`、`init_cmd_adam` | `create_damping_cmd`、`create_zero_cmd` |
| `rotation_helper.py` | `ypr_to_quaternion` | `get_gravity_orientation`、`transform_imu_data`（依赖 `scipy`） |
| `remote_controller.py` | `set`、`button`、`KeyMap` | `get_walk_x/y/yaw_direction_speed` |

---

## 文档资源

- **`img/`**：说明用配图（例如用 `ifconfig` / `ip` 查看网卡时的截图）。
