"""
Joint configuration and mapping
"""
import numpy as np
from typing import List, Dict


class JointConfig:
    """Joint configuration and ID mapping"""
    
    # Joint name to ID mapping
    # Note: Joint names now use ONNX format (snake_case with _joint suffix)
    # Order matches training environment robot.joint_names (from XML file, excluding freejoint)
    # Total 29 joints: left_leg(6) -> right_leg(6) -> waist(3) -> left_arm(7) -> right_arm(7)
    JOINT_NAME_TO_ID: Dict[str, int] = {
        # Lower body left (6 joints)
        "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
        "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
        # Lower body right (6 joints)
        "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
        "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
        # Waist (3 joints)
        "waist_roll_joint": 12, "waist_pitch_joint": 13, "waist_yaw_joint": 14,
        # Upper body left (7 joints)
        "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
        "left_elbow_joint": 18, "left_wrist_yaw_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_roll_joint": 21,
        # Upper body right (7 joints)
        "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
        "right_elbow_joint": 25, "right_wrist_yaw_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_roll_joint": 28
    }
    
    # Observation joint names (23 joints, matching ONNX metadata format)
    # Note: These names must match the ONNX model metadata joint names exactly
    # Order must match training environment robot.joint_names (excluding wrist joints)
    # Training order from XML: left_leg(6) -> right_leg(6) -> waist(3) -> left_arm(4) -> right_arm(4)
    # Wrist joints (wrist_yaw, wrist_pitch, wrist_roll) are excluded from observations
    OBS_JOINT_NAMES: List[str] = [
        # lower body left (6 joints) - matching XML/training order
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        # lower body right (6 joints) - matching XML/training order
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # waist (3 joints) - matching XML/training order
        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
        # upper body left (4 joints) - matching XML/training order (excluding wrist)
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        # upper body right (4 joints) - matching XML/training order (excluding wrist)
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint"
    ]
    
    # Wrist yaw joint names (not used in observation, but kept for compatibility)
    WRIST_YAW_JOINT_NAMES: List[str] = ["left_wrist_yaw_joint", "right_wrist_yaw_joint"]
    
    # Default joint positions (matching training environment from PND_ADAM_LITE_CFG)
    # Reference: /home/chenmt/workplace/unitree_rl_lab/source/pnd_rl_lab/pnd_rl_lab/assets/robots/pnd.py
    # Training environment init_state.joint_pos values from PND_ADAM_LITE_CFG.init_state.joint_pos
    # IMPORTANT: These values MUST match training environment exactly for consistency
    # Note: Joint order matches OBS_JOINT_NAMES (Controller order, not ONNX order)
    DEFAULT_JOINT_POSITIONS: np.ndarray = np.array([
        # Left leg (6 joints) - matching training environment exactly
        -0.32,   # left_hip_pitch_joint
        0.0,     # left_hip_roll_joint
        -0.18,   # left_hip_yaw_joint
        0.66,    # left_knee_joint
        -0.32,   # left_ankle_pitch_joint
        -0.0,    # left_ankle_roll_joint (note: -0.0 == 0.0, but keeping exact match)
        # Right leg (6 joints) - matching training environment exactly
        -0.32,   # right_hip_pitch_joint
        -0.0,    # right_hip_roll_joint (note: -0.0 == 0.0, but keeping exact match)
        0.18,    # right_hip_yaw_joint
        0.66,    # right_knee_joint
        -0.32,   # right_ankle_pitch_joint
        0.0,     # right_ankle_roll_joint
        # Waist (3 joints) - matching training environment exactly
        0.0,     # waist_roll_joint
        0.0,     # waist_pitch_joint
        0.0,     # waist_yaw_joint
        # Left arm (4 joints) - matching training environment exactly
        0.0,     # left_shoulder_pitch_joint
        0.1,     # left_shoulder_roll_joint
        0.0,     # left_shoulder_yaw_joint
        -0.3,    # left_elbow_joint
        # Right arm (4 joints) - matching training environment exactly
        0.0,     # right_shoulder_pitch_joint
        -0.1,    # right_shoulder_roll_joint
        0.0,     # right_shoulder_yaw_joint
        -0.3     # right_elbow_joint
    ], dtype=np.float32)
    
    # Constants
    K_BASE_NUM: int = 6  # Floating base DOF
    K_OBS_DOF: int = 23  # Number of observed joints
    
    @classmethod
    def get_joint_ids_from_names(cls, joint_names: List[str]) -> List[int]:
        """
        Get joint IDs from joint names
        
        Args:
            joint_names: List of joint names
            
        Returns:
            List of joint IDs
        """
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
        """Get observation joint IDs"""
        return cls.get_joint_ids_from_names(cls.OBS_JOINT_NAMES)
    
    @classmethod
    def get_wrist_yaw_ids(cls) -> List[int]:
        """Get wrist yaw joint IDs"""
        return cls.get_joint_ids_from_names(cls.WRIST_YAW_JOINT_NAMES)
