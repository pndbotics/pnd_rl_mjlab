"""
Joint configuration and mapping for Adam Pro.

Compared to Adam SP, Adam Pro adds two neck joints between waist_yaw and
left_shoulder_pitch. Neck joints are not controlled by the RL policy and are
held at a constant 0 rad on the real robot.
"""
import numpy as np
from typing import Dict, List


class JointConfig:
    """Joint configuration and ID mapping for Adam Pro (31 motors)."""

    # Joint name to motor index mapping (LowState / LowCmd motor_state order).
    # Total 31 joints: left_leg(6) -> right_leg(6) -> waist(3) -> neck(2) ->
    #                   left_arm(7) -> right_arm(7)
    JOINT_NAME_TO_ID: Dict[str, int] = {
        # Lower body left (6 joints)
        "left_hip_pitch_joint": 0,
        "left_hip_roll_joint": 1,
        "left_hip_yaw_joint": 2,
        "left_knee_joint": 3,
        "left_ankle_pitch_joint": 4,
        "left_ankle_roll_joint": 5,
        # Lower body right (6 joints)
        "right_hip_pitch_joint": 6,
        "right_hip_roll_joint": 7,
        "right_hip_yaw_joint": 8,
        "right_knee_joint": 9,
        "right_ankle_pitch_joint": 10,
        "right_ankle_roll_joint": 11,
        # Waist (3 joints)
        "waist_roll_joint": 12,
        "waist_pitch_joint": 13,
        "waist_yaw_joint": 14,
        # Neck (2 joints, fixed at 0 rad, not in policy obs/actions)
        "neck_yaw_joint": 15,
        "neck_pitch_joint": 16,
        # Upper body left (7 joints)
        "left_shoulder_pitch_joint": 17,
        "left_shoulder_roll_joint": 18,
        "left_shoulder_yaw_joint": 19,
        "left_elbow_joint": 20,
        "left_wrist_yaw_joint": 21,
        "left_wrist_pitch_joint": 22,
        "left_wrist_roll_joint": 23,
        # Upper body right (7 joints)
        "right_shoulder_pitch_joint": 24,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 26,
        "right_elbow_joint": 27,
        "right_wrist_yaw_joint": 28,
        "right_wrist_pitch_joint": 29,
        "right_wrist_roll_joint": 30,
    }

    # Policy observation / action joint names (23 joints, same order as Adam SP training).
    OBS_JOINT_NAMES: List[str] = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    NECK_JOINT_NAMES: List[str] = ["neck_yaw_joint", "neck_pitch_joint"]
    WRIST_YAW_JOINT_NAMES: List[str] = ["left_wrist_yaw_joint", "right_wrist_yaw_joint"]

    # Fixed neck joint commands (constant 0 deg).
    NECK_FIXED_ANGLES: List[float] = [0.0, 0.0]
    NECK_KP: List[float] = [60.0, 60.0]
    NECK_KD: List[float] = [3.0, 3.0]

    # Policy-controlled joints mapped to LowState / LowCmd motor indices.
    JOINT2MOTOR_IDX: List[int] = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12, 13, 14,
        17, 18, 19, 20,
        24, 25, 26, 27,
    ]

    DEFAULT_JOINT_POSITIONS: np.ndarray = np.array([
        -0.32, 0.0, -0.18, 0.66, -0.32, -0.0,
        -0.32, -0.0, 0.18, 0.66, -0.32, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, -0.3,
        0.0, -0.1, 0.0, -0.3,
    ], dtype=np.float32)

    K_BASE_NUM: int = 6
    K_OBS_DOF: int = 23
    K_NUM_MOTORS: int = 31

    @classmethod
    def get_joint_ids_from_names(cls, joint_names: List[str]) -> List[int]:
        joint_ids = []
        for name in joint_names:
            if name in cls.JOINT_NAME_TO_ID:
                joint_ids.append(cls.JOINT_NAME_TO_ID[name])
            else:
                print(f"Warning: Joint {name} not found in joint mapping")
                joint_ids.append(-1)
        return joint_ids

    @classmethod
    def get_obs_joint_ids(cls) -> List[int]:
        return cls.get_joint_ids_from_names(cls.OBS_JOINT_NAMES)

    @classmethod
    def get_neck_ids(cls) -> List[int]:
        return cls.get_joint_ids_from_names(cls.NECK_JOINT_NAMES)

    @classmethod
    def get_wrist_yaw_ids(cls) -> List[int]:
        return cls.get_joint_ids_from_names(cls.WRIST_YAW_JOINT_NAMES)

    @classmethod
    def get_fixed_neck_commands(cls) -> List[Dict[str, float]]:
        """Return fixed PD targets for neck joints (constant 0 rad)."""
        commands = []
        for i, motor_idx in enumerate(cls.get_neck_ids()):
            commands.append(
                {
                    "motor_idx": motor_idx,
                    "q": cls.NECK_FIXED_ANGLES[i],
                    "kp": cls.NECK_KP[i],
                    "kd": cls.NECK_KD[i],
                }
            )
        return commands
