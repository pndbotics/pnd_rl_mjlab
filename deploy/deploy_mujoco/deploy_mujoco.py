"""
MuJoCo Beyond Mimic 部署。

motion_q + motion_dq + anchor_ori(6) + ang_vel(3) + q_offset + dq + last_action
"""
import os
import struct
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
XBOX_BUTTON_A = 0x00
XBOX_BUTTON_Y = 0x04
XBOX_AXIS_LY = 0x01


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


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    rot_wb = matrix_from_quat(quat_wxyz)
    return (rot_wb.T @ gravity_w).astype(np.float32)


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


class JoystickReader:
    def __init__(self, device_path: str = "/dev/input/js0"):
        self.device_path = device_path
        self.fd = None
        self.buttons = {}
        self.prev_buttons = {}
        self.axes = {}
        # Match the C++ mapping style in pnd_adam_deploy_private/joystick.cpp
        self.dead_area = 5000
        self.max_value = 32767
        self.ly_dir = -1.0
        self.max_speed_x = 0.5
        self.min_speed_x = -0.5

    def _try_open(self) -> None:
        if self.fd is not None:
            return
        try:
            self.fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
            print(f"Joystick connected: {self.device_path}")
        except OSError:
            self.fd = None

    def poll(self) -> None:
        self._try_open()
        if self.fd is None:
            return
        while True:
            try:
                packet = os.read(self.fd, 8)
            except BlockingIOError:
                break
            except OSError:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
                self.buttons.clear()
                self.prev_buttons.clear()
                break
            if len(packet) != 8:
                break
            _, value, event_type, number = struct.unpack("IhBB", packet)
            event_type = event_type & ~JS_EVENT_INIT
            if event_type == JS_EVENT_BUTTON:
                self.buttons[int(number)] = int(value)
            elif event_type == JS_EVENT_AXIS:
                self.axes[int(number)] = int(value)

    def button_rising(self, button_id: int) -> bool:
        curr = int(self.buttons.get(button_id, 0)) == 1
        prev = int(self.prev_buttons.get(button_id, 0)) == 1
        self.prev_buttons[button_id] = int(self.buttons.get(button_id, 0))
        return curr and not prev

    def get_walk_x_direction_speed(self) -> float:
        x_value = int(self.axes.get(XBOX_AXIS_LY, 0))
        abs_x = abs(x_value)
        if abs_x > self.dead_area and abs_x <= self.max_value:
            ratio = (abs_x - self.dead_area) / float(self.max_value - self.dead_area)
            if x_value > 0:
                return self.ly_dir * self.max_speed_x * ratio
            return self.ly_dir * self.min_speed_x * ratio
        return 0.0


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    config_filename_by_task = {
        "beyondmimic": "adam_sp_beyondmimic.yaml",
        "velocity_flat": "adam_sp_velocity_flat.yaml",
    }

    def resolve_path(path_str: str) -> str:
        if path_str is None:
            return None
        path = path_str.replace("{PROJECT_ROOT}", project_root)
        path = os.path.normpath(path)
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))

    def load_task_context(task_name: str) -> dict:
        config_path = os.path.join(
            project_root,
            "deploy",
            "deploy_mujoco",
            "configs",
            config_filename_by_task[task_name],
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        policy_path = resolve_path(config["policy_path"])
        motion_path = resolve_path(config.get("motion_path"))
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
        velocity_commands = np.asarray(config.get("velocity_commands", [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity_joint_indices = np.asarray(
            config.get("velocity_joint_indices", list(range(num_actions))),
            dtype=np.int64,
        )

        if default_angles.shape != (num_actions,):
            raise ValueError(f"default_angles must contain {num_actions} values.")
        if kps.shape != (num_actions,) or kds.shape != (num_actions,):
            raise ValueError(f"kps and kds must both contain {num_actions} values.")
        if task_name == "velocity_flat" and velocity_commands.shape != (3,):
            raise ValueError("velocity_commands must contain exactly 3 values [vx, vy, wz].")
        if velocity_joint_indices.shape != (num_actions,):
            raise ValueError(f"velocity_joint_indices must contain {num_actions} values.")

        motion_loader = None
        if task_name == "beyondmimic":
            if motion_path is None:
                raise ValueError("beyondmimic task requires motion_path in config.")
            motion_loader = BeyondMimicMotionLoader(
                motion_path=motion_path,
                motion_start=motion_start,
                motion_end=motion_end,
                motion_loop=motion_loop,
                motion_joint_indices=motion_joint_indices,
                anchor_body_name=anchor_body_name,
            )

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
            raise ValueError(f"Could not find obs input for task={task_name} in the ONNX model.")

        output_meta = session.get_outputs()[0]
        output_dim = infer_flat_dim(output_meta.shape, "policy output")
        if output_dim != num_actions:
            raise ValueError(f"Policy output dimension mismatch: config={num_actions}, policy={output_dim}.")

        return {
            "task_type": task_name,
            "xml_path": xml_path,
            "simulation_duration": simulation_duration,
            "simulation_dt": simulation_dt,
            "control_decimation": control_decimation,
            "kps": kps,
            "kds": kds,
            "default_angles": default_angles,
            "action_scale": action_scale,
            "num_actions": num_actions,
            "num_obs": num_obs,
            "motion_loader": motion_loader,
            "velocity_commands": velocity_commands,
            "velocity_joint_indices": velocity_joint_indices,
            "session": session,
            "obs_input_name": obs_input_name,
            "time_input_name": time_input_name,
        }

    task_contexts = {
        "beyondmimic": load_task_context("beyondmimic"),
        "velocity_flat": load_task_context("velocity_flat"),
    }
    xml_path = os.path.join(project_root, "deploy", "adam_despription", "scene.xml")

    simulation_duration = max(
        task_contexts["beyondmimic"]["simulation_duration"],
        task_contexts["velocity_flat"]["simulation_duration"],
    )
    simulation_dt = task_contexts["beyondmimic"]["simulation_dt"]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if np.any(task_contexts["velocity_flat"]["velocity_joint_indices"] < 0) or np.any(
        task_contexts["velocity_flat"]["velocity_joint_indices"] >= model.nu
    ):
        raise ValueError(f"velocity_joint_indices must be in [0, {model.nu - 1}] for the unified xml model.")
    active_task = None
    active_ctx = None
    action = np.zeros(0, dtype=np.float32)
    target_dof_pos = np.zeros(0, dtype=np.float32)
    motion_timestep = 0
    step_counter = 0
    joystick = JoystickReader()

    print("Gamepad controls: A -> beyondmimic, Y -> velocity_flat")
    print("Loaded mujoco deploy in idle mode. Press gamepad A/Y to enter a task.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            pending_task = None
            joystick.poll()
            if joystick.button_rising(XBOX_BUTTON_A):
                pending_task = "beyondmimic"
            elif joystick.button_rising(XBOX_BUTTON_Y):
                pending_task = "velocity_flat"

            if pending_task in task_contexts and pending_task != active_task:
                active_task = pending_task
                active_ctx = task_contexts[active_task]
                model.opt.timestep = active_ctx["simulation_dt"]
                action = np.zeros(active_ctx["num_actions"], dtype=np.float32)
                target_dof_pos = active_ctx["default_angles"].copy()
                motion_timestep = 0
                step_counter = 0
                print(
                    f"Switched task -> {active_task} (obs={active_ctx['num_obs']}, act={active_ctx['num_actions']})"
                )

            if active_ctx is None:
                mujoco.mj_step(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue

            if active_task == "velocity_flat":
                qj_all = data.qpos[7 : 7 + model.nu].astype(np.float32)
                dqj_all = data.qvel[6 : 6 + model.nu].astype(np.float32)
                qj = qj_all[active_ctx["velocity_joint_indices"]]
                dqj = dqj_all[active_ctx["velocity_joint_indices"]]
            else:
                qj = data.qpos[7 : 7 + active_ctx["num_actions"]].astype(np.float32)
                dqj = data.qvel[6 : 6 + active_ctx["num_actions"]].astype(np.float32)

            tau = pd_control(
                target_dof_pos,
                qj,
                active_ctx["kps"],
                np.zeros(active_ctx["num_actions"], dtype=np.float32),
                dqj,
                active_ctx["kds"],
            )
            if active_task == "velocity_flat":
                data.ctrl[active_ctx["velocity_joint_indices"]] = tau
            else:
                data.ctrl[: active_ctx["num_actions"]] = tau
            mujoco.mj_step(model, data)
            step_counter += 1

            if step_counter % active_ctx["control_decimation"] == 0:
                quat = data.qpos[3:7].astype(np.float32)
                ang_vel = data.qvel[3:6].astype(np.float32)
                if active_task == "velocity_flat":
                    velocity_commands = active_ctx["velocity_commands"].copy()
                    velocity_commands[0] = joystick.get_walk_x_direction_speed()
                    projected_gravity = projected_gravity_from_quat(quat)
                    obs = np.concatenate(
                        [
                            ang_vel,
                            projected_gravity,
                            velocity_commands,
                            qj - active_ctx["default_angles"],
                            dqj,
                            action,
                        ],
                        dtype=np.float32,
                    ).reshape(1, -1)
                    motion_idx = motion_timestep
                else:
                    motion_q, motion_dq, motion_anchor_quat, motion_idx = active_ctx["motion_loader"].get_reference(
                        motion_timestep
                    )
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
                            qj - active_ctx["default_angles"],
                            dqj,
                            action,
                        ],
                        dtype=np.float32,
                    ).reshape(1, -1)

                if obs.shape[1] != active_ctx["num_obs"]:
                    raise ValueError(
                        f"Observation dimension mismatch for task={active_task}: "
                        f"config={active_ctx['num_obs']}, runtime={obs.shape[1]}"
                    )

                policy_inputs = {active_ctx["obs_input_name"]: obs}
                if active_ctx["time_input_name"] is not None and active_ctx["motion_loader"] is not None:
                    if active_ctx["motion_loader"].length > 1:
                        time_step = np.array(
                            [[motion_idx / float(active_ctx["motion_loader"].length - 1)]], dtype=np.float32
                        )
                    else:
                        time_step = np.zeros((1, 1), dtype=np.float32)
                    policy_inputs[active_ctx["time_input_name"]] = time_step

                action = np.asarray(active_ctx["session"].run(None, policy_inputs)[0], dtype=np.float32).reshape(-1)
                if active_ctx["action_scale"].ndim == 0:
                    target_dof_pos = active_ctx["default_angles"] + action * float(active_ctx["action_scale"].item())
                elif active_ctx["action_scale"].shape == action.shape:
                    target_dof_pos = active_ctx["default_angles"] + action * active_ctx["action_scale"]
                else:
                    raise ValueError(
                        f"action_scale must be a scalar or a {active_ctx['num_actions']}D vector."
                    )
                motion_timestep += 1

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
