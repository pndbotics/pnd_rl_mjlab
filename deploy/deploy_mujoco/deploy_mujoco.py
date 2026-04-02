"""
MuJoCo Beyond Mimic 部署。

motion_q + motion_dq + anchor_ori(6) + ang_vel(3) + q_offset + dq + last_action
"""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


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


class BeyondMimicMotionLoader:
    def __init__(self, motion_path: str, motion_start: int, motion_end: int, motion_loop: bool, motion_joint_indices, anchor_body_name: str):
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        if anchor_body_name not in ADAM_SP_BODY_NAMES:
            raise ValueError(f"Unsupported anchor_body_name: {anchor_body_name}")

        data = np.load(motion_path)
        start = motion_start
        end = None if motion_end < 0 else motion_end
        self.joint_pos = np.asarray(data["joint_pos"][start:end], dtype=np.float32)
        self.joint_vel = np.asarray(data["joint_vel"][start:end], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"][start:end], dtype=np.float32)
        self.anchor_quat_w = body_quat_w[:, ADAM_SP_BODY_NAMES.index(anchor_body_name)]
        self.motion_joint_indices = np.asarray(motion_joint_indices, dtype=np.int64)
        self.length = int(self.joint_pos.shape[0])
        self.loop = bool(motion_loop)

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


if __name__ == "__main__":
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    parser = argparse.ArgumentParser(description="MuJoCo Beyond Mimic 部署")
    parser.add_argument("config_file", type=str, help="configs/ 下的配置文件名")
    args = parser.parse_args()

    config_path = os.path.join(project_root, "deploy", "deploy_mujoco", "configs", args.config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def resolve_path(path_str: str) -> str:
        if path_str is None:
            return None
        path = path_str.replace("{PROJECT_ROOT}", project_root)
        path = os.path.normpath(path)
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))

    policy_path = resolve_path(config["policy_path"])
    motion_path = resolve_path(config["motion_path"])
    xml_path = resolve_path(config["xml_path"])

    simulation_duration = float(config["simulation_duration"])
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    action_scale = np.asarray(config.get("action_scale", 0.5), dtype=np.float32)
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    motion_start = int(config.get("motion_start", 0))
    motion_end = int(config.get("motion_end", -1))
    motion_loop = bool(config.get("motion_loop", False))
    anchor_body_name = config.get("anchor_body_name", "pelvis")
    motion_joint_indices = config.get("motion_joint_indices", list(range(num_actions)))

    if num_actions != 29:
        raise ValueError("MuJoCo Beyond Mimic deployment expects num_actions=29.")
    if default_angles.shape != (29,):
        raise ValueError("default_angles must contain 29 values.")

    motion_loader = BeyondMimicMotionLoader(
        motion_path=motion_path,
        motion_start=motion_start,
        motion_end=motion_end,
        motion_loop=motion_loop,
        motion_joint_indices=motion_joint_indices,
        anchor_body_name=anchor_body_name,
    )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    if model.nu != num_actions:
        raise ValueError(f"MuJoCo actuator count mismatch: xml has {model.nu}, config expects {num_actions}.")

    session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    obs_input_name = None
    time_input_name = None
    for policy_input in session.get_inputs():
        input_dim = infer_flat_dim(policy_input.shape, f"input {policy_input.name}")
        if input_dim == num_obs:
            obs_input_name = policy_input.name
        else:
            time_input_name = policy_input.name
    if obs_input_name is None:
        raise ValueError("Could not find Beyond Mimic obs input in the ONNX model.")

    output_meta = session.get_outputs()[0]
    output_dim = infer_flat_dim(output_meta.shape, "policy output")
    if output_dim != num_actions:
        raise ValueError(f"Policy output dimension mismatch: config={num_actions}, policy={output_dim}.")

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    motion_timestep = 0
    step_counter = 0

    print(
        f"Loaded Beyond Mimic mujoco deploy: obs={num_obs}, act={num_actions}, "
        f"motion_frames={motion_loader.length}, time_input={time_input_name is not None}"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(
                target_dof_pos,
                data.qpos[7 : 7 + num_actions],
                kps,
                np.zeros(num_actions, dtype=np.float32),
                data.qvel[6 : 6 + num_actions],
                kds,
            )
            data.ctrl[:num_actions] = tau
            mujoco.mj_step(model, data)
            step_counter += 1

            if step_counter % control_decimation == 0:
                qj = data.qpos[7 : 7 + num_actions].astype(np.float32)
                dqj = data.qvel[6 : 6 + num_actions].astype(np.float32)
                quat = data.qpos[3:7].astype(np.float32)
                ang_vel = data.qvel[3:6].astype(np.float32)

                motion_q, motion_dq, motion_anchor_quat, motion_idx = motion_loader.get_reference(motion_timestep)
                roll, pitch, _ = euler_xyz_from_quat(motion_anchor_quat)
                _, _, yaw = euler_xyz_from_quat(quat)
                corrected_anchor_quat = quat_from_euler_xyz(float(roll), float(pitch), float(yaw))
                quat_diff = quat_mul(quat_inv(quat), corrected_anchor_quat)
                anchor_ori_b = matrix_from_quat(quat_diff)[:, :2].reshape(-1).astype(np.float32)

                obs = np.concatenate(
                    [
                        motion_q,
                        motion_dq,
                        anchor_ori_b,
                        ang_vel,
                        qj - default_angles,
                        dqj,
                        action,
                    ],
                    dtype=np.float32,
                ).reshape(1, -1)

                policy_inputs = {obs_input_name: obs}
                if time_input_name is not None:
                    if motion_loader.length > 1:
                        time_step = np.array([[motion_idx / float(motion_loader.length - 1)]], dtype=np.float32)
                    else:
                        time_step = np.zeros((1, 1), dtype=np.float32)
                    policy_inputs[time_input_name] = time_step

                action = np.asarray(session.run(None, policy_inputs)[0], dtype=np.float32).reshape(-1)
                if action_scale.ndim == 0:
                    target_dof_pos = default_angles + action * float(action_scale.item())
                elif action_scale.shape == action.shape:
                    target_dof_pos = default_angles + action * action_scale
                else:
                    raise ValueError("action_scale must be a scalar or a 29D vector.")
                motion_timestep += 1

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
