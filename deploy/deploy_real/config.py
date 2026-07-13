import os

import numpy as np
import yaml


def _resolve_path(path, project_root):
    if path is None:
        return None
    return path.replace("{PROJECT_ROOT}", project_root)


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.project_root = project_root

        self.control_dt = float(config["control_dt"])
        self.msg_type = config["msg_type"]
        self.imu_type = config.get("imu_type", "pelvis")
        self.num_motors = int(config.get("num_motors", 29))
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]
        self.fixed_motor_joints = list(config.get("fixed_motor_joints", []))

        self.task_type = str(config.get("task_type", "beyondmimic")).strip().lower()
        self.policy_path = _resolve_path(config["policy_path"], project_root)
        self.motion_path = _resolve_path(config.get("motion_path"), project_root)
        self.motion_start = int(config.get("motion_start", 0))
        self.motion_end = int(config.get("motion_end", -1))
        self.motion_loop = bool(config.get("motion_loop", False))
        self.anchor_body_name = config.get("anchor_body_name", "pelvis")
        self.velocity_commands = np.asarray(config.get("velocity_commands", [0.0, 0.0, 0.0]), dtype=np.float32)

        self.joint2motor_idx = list(config["joint2motor_idx"])
        self.kps = np.asarray(config["kps"], dtype=np.float32)
        self.kds = np.asarray(config["kds"], dtype=np.float32)
        self.default_angles = np.asarray(config["default_angles"], dtype=np.float32)
        self.action_scale = np.asarray(config.get("action_scale", 0.5), dtype=np.float32)

        self.num_actions = int(config["num_actions"])
        self.num_obs = int(config["num_obs"])
        self.action_output_to_target_idx = np.asarray(
            config.get("action_output_to_target_idx", list(range(self.num_actions))), dtype=np.int64
        )
        self.motion_joint_indices = np.asarray(
            config.get("motion_joint_indices", list(range(self.num_actions))), dtype=np.int64
        )

        if self.task_type not in {"beyondmimic", "velocity_flat"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}.")
        if self.msg_type not in {"adam_sp", "adam_pro"}:
            raise ValueError("Supported msg_type values: 'adam_sp', 'adam_pro'.")
        if self.msg_type == "adam_sp" and self.num_motors != 29:
            raise ValueError("adam_sp deployment expects num_motors=29.")
        if self.msg_type == "adam_pro" and self.num_motors != 31:
            raise ValueError("adam_pro deployment expects num_motors=31.")
        if self.imu_type != "pelvis":
            raise ValueError("This Beyond Mimic deployment currently only supports imu_type='pelvis'.")
        if self.policy_path is None:
            raise ValueError("policy_path is required.")
        if len(self.joint2motor_idx) != self.num_actions:
            raise ValueError(f"joint2motor_idx must contain {self.num_actions} joints.")
        if self.default_angles.shape != (self.num_actions,):
            raise ValueError(f"default_angles must contain {self.num_actions} values.")
        if self.kps.shape != (self.num_actions,) or self.kds.shape != (self.num_actions,):
            raise ValueError(f"kps and kds must contain {self.num_actions} values.")
        if self.action_output_to_target_idx.shape != (self.num_actions,):
            raise ValueError(f"action_output_to_target_idx must contain {self.num_actions} values.")
        for entry in self.fixed_motor_joints:
            if "motor_idx" not in entry:
                raise ValueError("Each fixed_motor_joints entry must include motor_idx.")
            if int(entry["motor_idx"]) < 0 or int(entry["motor_idx"]) >= self.num_motors:
                raise ValueError(
                    f"fixed_motor_joints motor_idx out of range: {entry['motor_idx']} "
                    f"(num_motors={self.num_motors})."
                )

        if self.task_type == "beyondmimic":
            if self.num_actions != 29:
                raise ValueError("Beyond Mimic deployment expects num_actions=29.")
            if self.motion_path is None:
                raise ValueError("beyondmimic requires motion_path.")
            if self.motion_joint_indices.shape != (self.num_actions,):
                raise ValueError(f"motion_joint_indices must contain {self.num_actions} values.")
        else:
            if self.num_actions != 23:
                raise ValueError("velocity_flat deployment expects num_actions=23.")
            if self.num_obs != 78:
                raise ValueError("velocity_flat deployment expects num_obs=78.")
            if self.velocity_commands.shape != (3,):
                raise ValueError("velocity_commands must contain exactly 3 values.")
