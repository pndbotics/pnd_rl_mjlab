"""
Observation computation module (DDS version) - velocity_flat format
"""
import numpy as np
from typing import Optional, Any
from scipy.spatial.transform import Rotation as R
from .joint_config import JointConfig

class RobotState:
    """Simple RobotState interface for type hints"""
    q_a: list
    q_dot_a: list
    tau_a: list


class ObservationProcessor:
    """Processes robot state to generate observations for RL policy (velocity_flat format)"""
    
    def __init__(self, config, default_joint_positions: Optional[np.ndarray] = None):
        """
        Initialize observation processor
        
        Args:
            config: Configuration object
            default_joint_positions: Default joint positions in Controller order
                                     If None, falls back to joint_config.DEFAULT_JOINT_POSITIONS
        """
        self.config = config
        self.joint_config = JointConfig
        
        if default_joint_positions is not None:
            if len(default_joint_positions) != self.joint_config.K_OBS_DOF:
                raise ValueError(
                    f"default_joint_positions length mismatch: "
                    f"got={len(default_joint_positions)}, expected={self.joint_config.K_OBS_DOF}"
                )
            self.default_joint_positions = np.array(default_joint_positions, dtype=np.float32)
        else:
            self.default_joint_positions = self.joint_config.DEFAULT_JOINT_POSITIONS.copy()
        
        # velocity_flat observation: base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) 
        # + joint_pos_rel(23) + joint_vel_rel(23) + last_action(23) = 78
        self.obs_num = config.obs_num if hasattr(config, 'obs_num') and config.obs_num > 0 else 78
        
        self.action_last = np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
        self.robot_anchor_rpy = np.zeros(3, dtype=np.float32)
        self.robot_anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.robot_anchor_ang_vel = np.zeros(3, dtype=np.float32)
        
        print(f"✓ ObservationProcessor: velocity_flat format ({self.obs_num}D)")
    
    def quat_inv(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inverse of quaternion
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Inverse quaternion [w, x, y, z]
        """
        # For unit quaternion, inverse is conjugate: [w, -x, -y, -z]
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    
    def compute_gvec_pelvis(self, robot_anchor_quat: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector in pelvis frame (projected_gravity)
        
        Args:
            robot_anchor_quat: Robot anchor quaternion [w, x, y, z]
            
        Returns:
            Gravity vector in pelvis frame [3]
        """
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        robot_quat_inv = self.quat_inv(robot_anchor_quat)
        w, x, y, z = robot_quat_inv[0], robot_quat_inv[1], robot_quat_inv[2], robot_quat_inv[3]
        vx, vy, vz = gravity_w[0], gravity_w[1], gravity_w[2]
        
        gvec_b = np.array([
            (1 - 2*(y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
            2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
            2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz
        ], dtype=np.float32)
        
        return gvec_b
    
    def compute_obs(self, low_state: Any, 
                    velocity_commands: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute observation data for velocity_flat format
        
        Observation format (78D):
        - base_ang_vel: 3D (angular velocity from IMU)
        - projected_gravity: 3D (gravity vector in pelvis frame)
        - velocity_commands: 3D (lin_vel_x, lin_vel_y, ang_vel_z)
        - joint_pos_rel: 23D (joint positions relative to default)
        - joint_vel_rel: 23D (joint velocities)
        - last_action: 23D (last action taken)
        
        Args:
            low_state: LowState_ message from DDS
            velocity_commands: Velocity commands [lin_vel_x, lin_vel_y, ang_vel_z] (3D, optional)
                                If None, uses zeros
            
        Returns:
            Observation array [obs_num]
        """
        quat_wxyz = np.array(low_state.imu_state.quaternion)
        self.robot_anchor_quat[:] = quat_wxyz
        
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rpy = rot.as_euler('xyz')
        self.robot_anchor_rpy[:] = rpy
        
        ang_vel = np.array(low_state.imu_state.gyroscope)
        self.robot_anchor_ang_vel[:] = ang_vel
        
        wrist_yaw_ids = self.joint_config.get_wrist_yaw_ids()
        k_dof = self.joint_config.K_OBS_DOF
        
        cur_joint_pos = np.zeros(k_dof, dtype=np.float32)
        cur_joint_vel = np.zeros(k_dof, dtype=np.float32)
        
        joint_idx = 0
        for motor_idx in range(25):
            if motor_idx not in wrist_yaw_ids and joint_idx < k_dof:
                cur_joint_pos[joint_idx] = low_state.motor_state[motor_idx].q
                cur_joint_vel[joint_idx] = low_state.motor_state[motor_idx].dq
                joint_idx += 1
        
        obs = np.zeros(self.obs_num, dtype=np.float32)
        idx = 0
        
        # base_ang_vel (3D)
        obs[idx:idx+3] = self.robot_anchor_ang_vel[:3]
        idx += 3
        
        # projected_gravity (3D)
        obs[idx:idx+3] = self.compute_gvec_pelvis(self.robot_anchor_quat)
        idx += 3
        
        # velocity_commands (3D)
        if velocity_commands is not None and len(velocity_commands) >= 3:
            obs[idx:idx+3] = velocity_commands[:3]
        else:
            obs[idx:idx+3] = 0.0
        idx += 3
        
        # joint_pos_rel (23D)
        joint_pos_rel = cur_joint_pos - self.default_joint_positions
        obs[idx:idx+k_dof] = joint_pos_rel[:k_dof]
        idx += k_dof
        
        # joint_vel_rel (23D)
        obs[idx:idx+k_dof] = cur_joint_vel[:k_dof]
        idx += k_dof
        
        # last_action (23D)
        normalized_last_action = self.action_last.copy()
        max_obs_safe_range = 30.0
        critical_obs_range = 50.0
        if np.any(np.abs(normalized_last_action) > critical_obs_range):
            print(f"⚠️  CRITICAL: last_action in observation exceeds critical range (max={critical_obs_range:.1f}): "
                  f"min={normalized_last_action.min():.4f}, max={normalized_last_action.max():.4f}")
            print(f"   Using zeros in observation to prevent instability")
            normalized_last_action = np.zeros_like(normalized_last_action)
        elif np.any(np.abs(normalized_last_action) > max_obs_safe_range):
            print(f"⚠️  Warning: last_action in observation exceeds safe range (max={max_obs_safe_range:.1f}): "
                  f"min={normalized_last_action.min():.4f}, max={normalized_last_action.max():.4f}")
            print(f"   Clipping last_action in observation to safe range")
            normalized_last_action = np.clip(normalized_last_action, -max_obs_safe_range, max_obs_safe_range)
        elif np.any(np.abs(normalized_last_action) > 10.0):
            print(f"⚠️  Warning: last_action exceeds expected range (training: [-7.82, 8.03]): "
                  f"min={normalized_last_action.min():.4f}, max={normalized_last_action.max():.4f}")
        
        obs[idx:idx+k_dof] = normalized_last_action[:k_dof]
        idx += k_dof
        
        if idx != self.obs_num:
            raise ValueError(f"Observation dimension mismatch: expected {self.obs_num}, got {idx}")
        
        return obs
    
    def update_action_last(self, action: np.ndarray) -> None:
        """
        Update last action
        
        Args:
            action: Last action taken (raw model output, NOT clipped)
        """
        max_safe_range = 30.0
        critical_range = 50.0
        
        if np.any(np.abs(action) > critical_range):
            print(f"⚠️  CRITICAL: last_action values exceed critical range (max={critical_range:.1f}): "
                  f"min={action.min():.4f}, max={action.max():.4f}")
            print(f"   Resetting last_action to zeros to break feedback loop")
            self.action_last[:] = 0.0
        elif np.any(np.abs(action) > max_safe_range):
            clipped_action = np.clip(action, -max_safe_range, max_safe_range)
            print(f"⚠️  Warning: last_action values exceed safe range (max={max_safe_range:.1f}): "
                  f"min={action.min():.4f}, max={action.max():.4f}")
            print(f"   Clipping last_action to safe range (instead of resetting to zeros)")
            self.action_last[:] = clipped_action
        else:
            self.action_last[:] = action.copy()
