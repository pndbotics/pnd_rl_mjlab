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
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        self.policy_path = _resolve_path(config["policy_path"], project_root)
        self.motion_path = _resolve_path(config["motion_path"], project_root)
        self.motion_start = int(config.get("motion_start", 0))
        self.motion_end = int(config.get("motion_end", -1))
        self.motion_loop = bool(config.get("motion_loop", False))
        self.anchor_body_name = config.get("anchor_body_name", "pelvis")

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

        if self.msg_type != "adam_sp":
            raise ValueError("This Beyond Mimic deployment currently only supports msg_type='adam_sp'.")
        if self.imu_type != "pelvis":
            raise ValueError("This Beyond Mimic deployment currently only supports imu_type='pelvis'.")
        if self.policy_path is None or self.motion_path is None:
            raise ValueError("policy_path and motion_path are required.")
        if len(self.joint2motor_idx) != 29:
            raise ValueError("joint2motor_idx must contain 29 joints.")
        if self.default_angles.shape != (29,):
            raise ValueError("default_angles must contain 29 values.")
        if self.kps.shape != (29,) or self.kds.shape != (29,):
            raise ValueError("kps and kds must contain 29 values.")
        if self.num_actions != 29:
            raise ValueError("Beyond Mimic deployment expects num_actions=29.")
        if self.action_output_to_target_idx.shape != (29,):
            raise ValueError("action_output_to_target_idx must contain 29 values.")
        if self.motion_joint_indices.shape != (29,):
            raise ValueError("motion_joint_indices must contain 29 values.")
