"""
Configuration loader and manager
"""
import yaml
import os
import numpy as np
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for RL controller"""
    
    def __init__(self, project_path: str):
        """
        Initialize configuration manager
        
        Args:
            project_path: Root path of the project
        """
        self.project_path = project_path
        
        # Observation and model parameters
        self.obs_num: int = 0
        self.observation_method: str = ""  # Observation method: "velocity_flat"
        self.model_pb_velocity_flat: str = ""
        
        # Locomotion Run specific parameters
        self.velocity_flat_velocity_commands: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Joint config path
        self.joint_config_path: str = ""
        
        # Joint PD gains (Kp, Kd) - will be loaded from ONNX metadata only
        self.joint_kp: Dict[str, float] = {}
        self.joint_kd: Dict[str, float] = {}
        
        # Load configuration
        self.load_config()
        # Note: Joint PD config is no longer loaded from file - only from ONNX metadata
    
    def load_config(self) -> None:
        """Load configuration from ros2.yaml"""
        # project_path is already the mimic directory, so use config/ros2.yaml directly
        config_path = os.path.join(
            self.project_path, 
            "config/ros2.yaml"
        )
        
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Load parameters
            if 'model_pb_velocity_flat' not in config_data:
                raise KeyError("'model_pb_velocity_flat' must be specified in config")
            self.model_pb_velocity_flat = config_data['model_pb_velocity_flat']
            self.observation_method = config_data.get('observation_method', 'velocity_flat').lower().strip()
            
            # Force observation_method to be "velocity_flat"
            if self.observation_method != "velocity_flat":
                self.observation_method = "velocity_flat"
            
            if 'obs_num' in config_data:
                self.obs_num = config_data['obs_num']
            else:
                # velocity_flat observation for adam_sp (23 observed joints):
                # base_ang_vel: 3 + projected_gravity: 3 + velocity_commands: 3 
                # + joint_pos_rel: 23 + joint_vel_rel: 23 + last_action: 23 = 78
                self.obs_num = 78
            
            # Velocity Flat specific configuration
            velocity_flat_config = config_data.get('velocity_flat', {})
            if isinstance(velocity_flat_config, dict):
                velocity_cmd = velocity_flat_config.get('velocity_commands', {})
                if isinstance(velocity_cmd, dict):
                    self.velocity_flat_velocity_commands = np.array([
                        float(velocity_cmd.get('lin_vel_x', 0.0)),
                        float(velocity_cmd.get('lin_vel_y', 0.0)),
                        float(velocity_cmd.get('ang_vel_z', 0.0))
                    ], dtype=np.float32)
                elif isinstance(velocity_cmd, list) and len(velocity_cmd) >= 3:
                    self.velocity_flat_velocity_commands = np.array([
                        float(velocity_cmd[0]),
                        float(velocity_cmd[1]),
                        float(velocity_cmd[2])
                    ], dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")
    
    def get_full_model_path(self, relative_path: str) -> str:
        """
        Get full path for a model file
        
        Args:
            relative_path: Relative path from project root
            
        Returns:
            Full absolute path
        """
        return os.path.join(self.project_path, relative_path)
    
    @property
    def policy_path(self) -> str:
        """Get full path to policy model"""
        return self.get_full_model_path(self.model_pb_velocity_flat)
    
    # Removed load_joint_pd_config() - joint PD gains are now loaded ONLY from ONNX metadata
    # This ensures consistency with the trained model and eliminates config file dependencies
    
    def get_joint_kp(self, joint_name: str) -> float:
        """
        Get Kp gain for a joint
        
        Joint PD gains must be loaded from ONNX metadata.
        If not found, raises an error (no fallback to config file).
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Kp gain value
            
        Raises:
            KeyError: If joint Kp is not found (should not happen if ONNX metadata was loaded)
        """
        kp = self.joint_kp.get(joint_name)
        if kp is None:
            raise KeyError(
                f"Kp not found for joint {joint_name}. "
                f"Joint PD gains must be loaded from ONNX metadata. "
                f"Please ensure the ONNX model includes joint_stiffness in metadata."
            )
        return kp
    
    def get_joint_kd(self, joint_name: str) -> float:
        """
        Get Kd gain for a joint
        
        Joint PD gains must be loaded from ONNX metadata.
        If not found, raises an error (no fallback to config file).
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Kd gain value
            
        Raises:
            KeyError: If joint Kd is not found (should not happen if ONNX metadata was loaded)
        """
        kd = self.joint_kd.get(joint_name)
        if kd is None:
            raise KeyError(
                f"Kd not found for joint {joint_name}. "
                f"Joint PD gains must be loaded from ONNX metadata. "
                f"Please ensure the ONNX model includes joint_damping in metadata."
            )
        return kd
    
    def load_joint_pd_from_onnx_metadata(self, joint_stiffness: np.ndarray, joint_damping: np.ndarray, joint_names: list) -> None:
        """
        Load joint PD gains from ONNX metadata
        
        This is the ONLY source for joint PD gains - config file reading has been removed.
        If metadata is invalid or missing, an error will be raised.
        
        Args:
            joint_stiffness: Array of Kp values (23 elements)
            joint_damping: Array of Kd values (23 elements)
            joint_names: List of joint names (23 elements)
        """
        if len(joint_stiffness) != len(joint_names) or len(joint_damping) != len(joint_names):
            raise ValueError(
                f"Joint PD arrays length mismatch: stiffness={len(joint_stiffness)}, "
                f"damping={len(joint_damping)}, names={len(joint_names)}"
            )
        
        # Check if all values are zero (common issue with ONNX export)
        all_stiffness_zero = np.allclose(joint_stiffness, 0.0)
        all_damping_zero = np.allclose(joint_damping, 0.0)
        
        if all_stiffness_zero or all_damping_zero:
            raise ValueError(
                f"ONNX metadata has zero Kp/Kd values (all zeros detected). "
                f"This indicates the ONNX model was exported without proper PD gains. "
                f"Please ensure the ONNX model includes valid joint PD gains in metadata."
            )
        
        # Update joint_kp and joint_kd dictionaries
        for i, joint_name in enumerate(joint_names):
            self.joint_kp[joint_name] = float(joint_stiffness[i])
            self.joint_kd[joint_name] = float(joint_damping[i])
