import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# install.sh 克隆位置：deploy/pnd_sdk_python/
# _pnd = Path(__file__).resolve().parent.parent / "pnd_sdk_python"
_pnd = Path("/home/pnd-humanoid/pnd_sdk_python")
_s = str(_pnd)
print(_pnd)
if (_pnd / "pndbotics_sdk_py").is_dir() and _s not in sys.path:
    sys.path.insert(0, _s)


from pndbotics_sdk_py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__LowState_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_, LowState_

from common.command_helper import MotorMode, init_cmd_adam
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import ypr_to_quaternion
from config import Config


ADAM_SP_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "waist_pitch_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
]


def infer_flat_dim(shape, name: str) -> int:
    dims = shape[1:] if len(shape) > 1 else shape
    total = 1
    for dim in dims:
        if not isinstance(dim, int):
            raise ValueError(f"{name} must have static ONNX dimensions, got {shape!r}.")
        total *= dim
    return total


def quat_inv(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    denom = float(np.dot(quat, quat))
    if denom <= 1e-9:
        raise ValueError("Quaternion norm is too small.")
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32) / denom


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dtype=np.float32,
    )


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    norm_sq = float(np.dot(quat, quat))
    if norm_sq <= 1e-9:
        raise ValueError("Quaternion norm is too small.")
    two_s = 2.0 / norm_sq
    return np.array(
        [
            [1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w)],
            [two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w)],
            [two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    rot_wb = matrix_from_quat(quat_wxyz)
    return (rot_wb.T @ gravity_w).astype(np.float32)


class BeyondMimicMotionLoader:
    def __init__(self, config: Config):
        motion_path = config.motion_path
        if motion_path is None:
            raise ValueError("motion_path is required.")
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        if config.anchor_body_name not in ADAM_SP_BODY_NAMES:
            raise ValueError(f"Unsupported anchor_body_name: {config.anchor_body_name}")

        data = np.load(motion_path)
        start = config.motion_start
        end = None if config.motion_end < 0 else config.motion_end
        self.joint_pos = np.asarray(data["joint_pos"][start:end], dtype=np.float32)
        self.joint_vel = np.asarray(data["joint_vel"][start:end], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"][start:end], dtype=np.float32)
        self.anchor_quat_w = body_quat_w[:, ADAM_SP_BODY_NAMES.index(config.anchor_body_name)]
        self.motion_joint_indices = config.motion_joint_indices
        self.length = int(self.joint_pos.shape[0])
        self.loop = config.motion_loop

        if self.joint_pos.shape != self.joint_vel.shape:
            raise ValueError("Motion joint_pos and joint_vel shapes do not match.")
        if self.length == 0:
            raise ValueError("Motion file does not contain any frames.")
        if self.joint_pos.shape[1] <= int(np.max(self.motion_joint_indices)):
            raise ValueError("motion_joint_indices exceed the motion joint dimension.")

    def get_reference(self, timestep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if self.loop:
            index = int(timestep % self.length)
        else:
            index = int(np.clip(timestep, 0, self.length - 1))
        return (
            self.joint_pos[index, self.motion_joint_indices].astype(np.float32),
            self.joint_vel[index, self.motion_joint_indices].astype(np.float32),
            self.anchor_quat_w[index].astype(np.float32),
            index,
        )


class Controller:
    def __init__(self, base_config: Config, task_config_by_name: dict[str, Config]) -> None:
        self.config = base_config
        self.task_config_by_name = task_config_by_name
        self.active_task = None
        self.remote_controller = RemoteController()
        self.motion_loader = None
        self.motion_timestep = 0
        self.prev_a_pressed = False
        self.prev_y_pressed = False

        self.low_cmd = pnd_adam_msg_dds__LowCmd_(29)
        self.low_state = pnd_adam_msg_dds__LowState_(29)
        self.qj = np.zeros(0, dtype=np.float32)
        self.dqj = np.zeros(0, dtype=np.float32)
        self.action = np.zeros(0, dtype=np.float32)
        self.target_dof_pos = np.zeros(0, dtype=np.float32)
        self.policy_session = None
        self.policy_obs_input_name = None
        self.policy_time_input_name = None

        self.mode_pr_ = MotorMode.PR
        self.lowcmd_publisher_ = ChannelPublisher(base_config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(base_config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.wait_for_low_state()
        init_cmd_adam(self.low_cmd, self.mode_pr_)
        print("Loaded real deploy in idle mode. Press A for beyondmimic, Y for velocity_flat.")

    def activate_task(self, task_name: str) -> None:
        if task_name not in self.task_config_by_name:
            raise ValueError(f"Unknown task: {task_name}")
        self.active_task = task_name
        self.config = self.task_config_by_name[task_name]
        self.motion_loader = BeyondMimicMotionLoader(self.config) if self.config.task_type == "beyondmimic" else None
        self.motion_timestep = 0

        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = self.config.default_angles.copy()

        policy_path = self.config.policy_path
        if policy_path is None:
            raise ValueError("policy_path is required.")
        self.policy_session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        policy_inputs = self.policy_session.get_inputs()
        self.policy_obs_input_name = None
        self.policy_time_input_name = None
        for policy_input in policy_inputs:
            input_dim = infer_flat_dim(policy_input.shape, f"input {policy_input.name}")
            if input_dim == self.config.num_obs:
                self.policy_obs_input_name = policy_input.name
            else:
                self.policy_time_input_name = policy_input.name
        if self.policy_obs_input_name is None:
            raise ValueError("Could not find policy obs input in the ONNX model.")

        output_meta = self.policy_session.get_outputs()[0]
        output_dim = infer_flat_dim(output_meta.shape, "policy output")
        if output_dim != self.config.num_actions:
            raise ValueError(
                f"Policy output dimension mismatch: config={self.config.num_actions}, policy={output_dim}."
            )
        print(
            f"Switched task -> {self.config.task_type} (obs={self.config.num_obs}, act={self.config.num_actions})"
        )

    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_) -> None:
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self) -> None:
        print("Successfully connected to the robot.")

    def zero_torque_state(self) -> None:
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self) -> None:
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            init_dof_pos[i] = self.low_state.motor_state[motor_idx].q

        for step in range(num_step):
            alpha = step / num_step
            for i, motor_idx in enumerate(self.config.joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[i] * (1 - alpha) + self.config.default_angles[i] * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self) -> None:
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i, motor_idx in enumerate(self.config.joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def get_robot_quat(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.asarray(
            ypr_to_quaternion(
                self.low_state.imu_state.ypr[0],
                self.low_state.imu_state.ypr[1],
                self.low_state.imu_state.ypr[2],
            ),
            dtype=np.float32,
        )
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32).reshape(-1)
        return quat, ang_vel

    def build_obs(self, quat: np.ndarray, ang_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.config.task_type == "velocity_flat":
            velocity_commands = np.asarray(
                [
                    self.remote_controller.get_walk_x_direction_speed(),
                    self.remote_controller.get_walk_y_direction_speed(),
                    self.remote_controller.get_walk_yaw_direction_speed(),
                ],
                dtype=np.float32,
            )
            velocity_commands[0] = float(np.clip(velocity_commands[0], -0.5, 0.5))
            q_offset = self.qj - self.config.default_angles
            obs = np.concatenate(
                [
                    ang_vel.astype(np.float32),
                    projected_gravity_from_quat(quat),
                    velocity_commands,
                    q_offset.astype(np.float32),
                    self.dqj.astype(np.float32),
                    self.action.astype(np.float32),
                ],
                dtype=np.float32,
            )
            time_step = None
        else:
            motion_q, motion_dq, motion_anchor_quat, motion_idx = self.motion_loader.get_reference(self.motion_timestep)
            roll, pitch, _ = euler_xyz_from_quat(motion_anchor_quat)
            _, _, yaw = euler_xyz_from_quat(quat)
            corrected_anchor_quat = quat_from_euler_xyz(float(roll), float(pitch), float(yaw))
            quat_diff = quat_mul(quat_inv(quat), corrected_anchor_quat)
            anchor_ori_b = matrix_from_quat(quat_diff)[:, :2].reshape(-1).astype(np.float32)

            q_offset = self.qj - self.config.default_angles
            obs = np.concatenate(
                [
                    motion_q,
                    motion_dq,
                    anchor_ori_b,
                    ang_vel.astype(np.float32),
                    q_offset.astype(np.float32),
                    self.dqj.astype(np.float32),
                    self.action.astype(np.float32),
                ],
                dtype=np.float32,
            )

            if self.policy_time_input_name is None:
                time_step = None
            elif self.motion_loader.length > 1:
                time_step = np.array([[motion_idx / float(self.motion_loader.length - 1)]], dtype=np.float32)
            else:
                time_step = np.zeros((1, 1), dtype=np.float32)
        if obs.shape[0] != self.config.num_obs:
            raise ValueError(
                f"Observation dimension mismatch: config={self.config.num_obs}, runtime={obs.shape[0]}."
            )
        return obs, time_step

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        if self.config.action_scale.ndim == 0:
            return self.config.default_angles + action * float(self.config.action_scale.item())
        if self.config.action_scale.shape == action.shape:
            return self.config.default_angles + action * self.config.action_scale
        raise ValueError(f"action_scale must be a scalar or a {self.config.num_actions}D vector.")

    def run_policy(self, obs: np.ndarray, time_step: np.ndarray | None) -> np.ndarray:
        inputs = {self.policy_obs_input_name: obs.reshape(1, -1).astype(np.float32)}
        if self.policy_time_input_name is not None:
            if time_step is None:
                raise ValueError("Policy requires time_step input.")
            inputs[self.policy_time_input_name] = time_step
        action = self.policy_session.run(None, inputs)[0]
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def run(self) -> None:
        a_pressed = self.remote_controller.button[KeyMap.A] == 1
        y_pressed = self.remote_controller.button[KeyMap.Y] == 1
        if a_pressed and not self.prev_a_pressed and self.active_task != "beyondmimic":
            self.activate_task("beyondmimic")
        elif y_pressed and not self.prev_y_pressed and self.active_task != "velocity_flat":
            self.activate_task("velocity_flat")
        self.prev_a_pressed = a_pressed
        self.prev_y_pressed = y_pressed

        if self.active_task is None:
            loop_start = time.time()
            for i in range(len(self.config.joint2motor_idx)):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            elapsed = time.time() - loop_start
            if elapsed < self.config.control_dt:
                time.sleep(self.config.control_dt - elapsed)
            return

        loop_start = time.time()

        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        quat, ang_vel = self.get_robot_quat()
        obs, time_step = self.build_obs(quat, ang_vel)
        self.action = self.run_policy(obs, time_step)
        self.target_dof_pos = self.scale_action(self.action)

        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = float(self.target_dof_pos[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd)
        self.motion_timestep += 1

        elapsed = time.time() - loop_start
        if elapsed < self.config.control_dt:
            time.sleep(self.config.control_dt - elapsed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_filename_by_task = {
        "beyondmimic": "adam_sp_beyondmimic.yaml",
        "velocity_flat": "adam_sp_velocity_flat.yaml",
    }
    config_by_task = {
        task_name: Config(os.path.join(project_root, "deploy", "deploy_real", "configs", cfg_name))
        for task_name, cfg_name in config_filename_by_task.items()
    }

    ChannelFactoryInitialize(1, args.net)
    controller = Controller(config_by_task["beyondmimic"], config_by_task)

    try:
        controller.zero_torque_state()
        controller.move_to_default_pos()
        while True:
            try:
                controller.run()
                if controller.remote_controller.button[KeyMap.B] == 1:
                    break
            except KeyboardInterrupt:
                break
    finally:
        print("Exit")
