from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from common.yaml_utils import load_yaml
import numpy as np
import onnxruntime
import torch
import os
import common.math_utils as math_utils
from collections.abc import Sequence

from adam_description.adam_sp import (
    ACTION_SCALE,
    BODY_NAMES,
    DEFAULT_ANGLES,
    JOINT_DAMPING,
    JOINT_KP,
)


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], start: int = 0, end: int = -1):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"][start:end, ...], dtype=torch.float32, device="cpu")
        self.joint_vel = torch.tensor(data["joint_vel"][start:end, ...], dtype=torch.float32, device="cpu")
        self._body_pos_w = torch.tensor(data["body_pos_w"][start:end, ...], dtype=torch.float32, device="cpu")
        self._body_quat_w = torch.tensor(data["body_quat_w"][start:end, ...], dtype=torch.float32, device="cpu")
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"][start:end, ...], dtype=torch.float32, device="cpu")
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"][start:end, ...], dtype=torch.float32, device="cpu")
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class AdamBeyondMimic(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.ADAM_BEYOND_MIMIC
        self.name_str = "adam_beyond_mimic"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = load_yaml(os.path.join(current_dir, "config", "adam_beyond_mimic.yaml"))

        self.tracking_body_names = list(config["tracking_body_names"])
        anchor_name = config["anchor_body_name"]
        assert anchor_name in self.tracking_body_names, (
            f"anchor_body_name {anchor_name!r} must be in tracking_body_names"
        )
        self.anchor_index = self.tracking_body_names.index(anchor_name)
        body_indexes = [BODY_NAMES.index(body) for body in self.tracking_body_names]

        motion_file = os.path.join(current_dir, config["motion_path"])
        motion_start = int(config["motion_start"])
        motion_end = int(config["motion_end"])
        self.motion_loader = MotionLoader(
            motion_file=motion_file,
            body_indexes=body_indexes,
            start=motion_start,
            end=motion_end if motion_end >= 0 else -1,
        )
        self.motion_length = self.motion_loader.time_step_total
        self.stiffness = JOINT_KP
        self.damping = JOINT_DAMPING

        self.num_actions = int(config["num_actions"])
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.processed_action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_timestep = 0

        self.onnx_path = os.path.join(current_dir, "model", config["policy_path"])
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)

    def _update_obs_buffer(self, timestep):
        command = self._get_command(timestep).flatten()
        motion_anchor_ori_b = self._get_anchor_ori(timestep).flatten()
        robot_angular_vel = torch.tensor(self.state_cmd.ang_vel, dtype=torch.float32).flatten()
        joint_pos = torch.tensor(self.state_cmd.q - DEFAULT_ANGLES, dtype=torch.float32).flatten()
        joint_vel = torch.tensor(self.state_cmd.dq, dtype=torch.float32).flatten()
        last_action = torch.from_numpy(self.last_action).flatten()
        obs_full = torch.cat(
            [command, motion_anchor_ori_b, robot_angular_vel, joint_pos, joint_vel, last_action]
        )
        return obs_full.reshape(1, -1).cpu().numpy().astype(np.float32)

    def _get_command(self, timestep):
        return torch.cat(
            [self.motion_loader.joint_pos[timestep], self.motion_loader.joint_vel[timestep]], dim=-1
        )

    def _get_anchor_ori(self, timestep):
        robot_pelvis_ori_w = torch.Tensor(self.state_cmd.base_quat)
        _, _, yaw = math_utils.euler_xyz_from_quat(robot_pelvis_ori_w.view(1, 4))
        motion_anchor_ori_w = self.motion_loader.body_quat_w[timestep, self.anchor_index]
        roll, pitch, _ = math_utils.euler_xyz_from_quat(motion_anchor_ori_w.view(1, 4))
        quat_motion_corrected = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        quat_diff = math_utils.quat_mul(
            math_utils.quat_inv(robot_pelvis_ori_w), quat_motion_corrected.squeeze(0)
        )
        mat = math_utils.matrix_from_quat(quat_diff)
        return mat[..., :2].reshape(mat.shape[0], -1).flatten()

    def _run_policy_step(self):
        obs_full = self._update_obs_buffer(self.motion_timestep)
        if self.motion_length > 1:
            t_norm = float(self.motion_timestep) / float(self.motion_length - 1)
        else:
            t_norm = 0.0
        time_step = np.array([[t_norm]], dtype=np.float32)
        outputs_result = self.ort_session.run(None, {"obs": obs_full, "time_step": time_step})
        self.last_action = outputs_result[0]
        action_29 = np.asarray(self.last_action, dtype=np.float32).reshape(-1)
        scale = np.asarray(ACTION_SCALE, dtype=np.float32)
        self.processed_action = (action_29 * scale + DEFAULT_ANGLES).astype(np.float32)
        self.policy_output.actions = self.processed_action.copy()
        self.policy_output.kps = np.array(list(self.stiffness.values()), dtype=np.float32)
        self.policy_output.kds = np.array(list(self.damping.values()), dtype=np.float32)
        self.motion_timestep += 1

    def enter(self):
        self.motion_timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def run(self):
        self._run_policy_step()

    def exit(self):
        self.motion_timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def checkChange(self):
        if self.motion_timestep >= self.motion_length - 1:
            return FSMStateName.PASSIVE
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        if self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        return FSMStateName.ADAM_BEYOND_MIMIC
