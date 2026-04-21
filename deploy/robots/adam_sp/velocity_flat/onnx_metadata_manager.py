"""
ONNX Metadata Manager
Centralized manager for loading and accessing ONNX model metadata
"""
import numpy as np
from typing import Optional, List
from .onnx_metadata_loader import ONNXMetadataLoader


class ONNXMetadataManager:
    """Centralized manager for ONNX metadata operations"""
    
    def __init__(self, onnx_path: str, joint_config, is_onnx_model: bool):
        """
        Initialize ONNX metadata manager
        
        Args:
            onnx_path: Path to ONNX model file
            joint_config: JointConfig instance
            is_onnx_model: Whether this is an ONNX model
        """
        self.onnx_path = onnx_path
        self.joint_config = joint_config
        self.is_onnx_model = is_onnx_model
        self._metadata_loader = None
        self._onnx_joint_names = None
        self._cached_data = {}
        
        if self.is_onnx_model:
            try:
                self._metadata_loader = ONNXMetadataLoader(onnx_path)
                self._onnx_joint_names = self._metadata_loader.get_joint_names()
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize ONNX metadata loader: {e}")
    
    def get_joint_names(self) -> Optional[List[str]]:
        """Get ONNX joint names"""
        return self._onnx_joint_names
    
    def load_joint_pd(self) -> tuple:
        """
        Load joint PD gains (Kp, Kd) from ONNX metadata and reorder to Controller order
        
        Returns:
            Tuple of (reordered_stiffness, reordered_damping) in Controller order
            
        Raises:
            RuntimeError: If metadata is missing or invalid
        """
        if not self.is_onnx_model:
            raise RuntimeError("Not an ONNX model, cannot load joint PD from metadata")
        
        if self._metadata_loader is None:
            raise RuntimeError("ONNX metadata loader not initialized")
        
        onnx_joint_stiffness = self._metadata_loader.get_joint_stiffness()
        onnx_joint_damping = self._metadata_loader.get_joint_damping()
        
        if onnx_joint_stiffness is None or onnx_joint_damping is None or self._onnx_joint_names is None:
            raise RuntimeError(
                f"Joint PD metadata not found in ONNX model: "
                f"stiffness={'None' if onnx_joint_stiffness is None else 'found'}, "
                f"damping={'None' if onnx_joint_damping is None else 'found'}, "
                f"names={'None' if self._onnx_joint_names is None else 'found'}"
            )
        
        if len(onnx_joint_stiffness) != len(onnx_joint_damping) or len(onnx_joint_stiffness) != len(self._onnx_joint_names):
            raise RuntimeError(
                f"Joint PD metadata length mismatch: "
                f"stiffness={len(onnx_joint_stiffness)}, damping={len(onnx_joint_damping)}, "
                f"names={len(self._onnx_joint_names)}"
            )
        
        if len(onnx_joint_stiffness) != len(self.joint_config.OBS_JOINT_NAMES):
            raise RuntimeError(
                f"Joint PD metadata length mismatch with Controller joint order: "
                f"metadata={len(onnx_joint_stiffness)}, expected={len(self.joint_config.OBS_JOINT_NAMES)}"
            )
        
        reordered_stiffness = np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
        reordered_damping = np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
        
        onnx_name_to_index = {name: idx for idx, name in enumerate(self._onnx_joint_names)}
        
        missing_joints = []
        for i, joint_name in enumerate(self.joint_config.OBS_JOINT_NAMES):
            if joint_name in onnx_name_to_index:
                onnx_idx = onnx_name_to_index[joint_name]
                reordered_stiffness[i] = onnx_joint_stiffness[onnx_idx]
                reordered_damping[i] = onnx_joint_damping[onnx_idx]
            else:
                missing_joints.append(joint_name)
        
        if missing_joints:
            raise RuntimeError(
                f"Some joints not found in ONNX metadata: {missing_joints}. "
                f"ONNX joint names: {self._onnx_joint_names}"
            )
        
        if np.allclose(reordered_stiffness, 0.0) or np.allclose(reordered_damping, 0.0):
            raise RuntimeError(
                f"ONNX metadata has zero Kp/Kd values (all zeros detected). "
                f"This indicates the ONNX model was exported without proper PD gains."
            )
        
        return reordered_stiffness, reordered_damping
    
    def load_action_scales(self) -> np.ndarray:
        """
        Load action_scales from ONNX metadata (ONNX order, no reordering)
        
        支持两种格式：
        1. 标量值（单个数值）：所有关节使用相同的缩放因子
        2. 数组（23维）：每个关节使用不同的缩放因子
        
        Returns:
            Action scales array in ONNX metadata order (23维)
            
        Raises:
            RuntimeError: If metadata is missing or invalid
        """
        if not self.is_onnx_model:
            raise RuntimeError("Cannot load action_scales: Not an ONNX model")
        
        if self._metadata_loader is None:
            raise RuntimeError("ONNX metadata loader not initialized")
        
        onnx_action_scale = self._metadata_loader.get_action_scale()
        
        if onnx_action_scale is None:
            raise RuntimeError(
                "action_scale not found in ONNX metadata. "
                "Please ensure the ONNX model includes action_scale in metadata."
            )
        
        # 转换为 numpy 数组
        onnx_action_scale = np.array(onnx_action_scale, dtype=np.float32)
        
        # 处理标量情况：如果只有一个值，扩展为23维数组（所有关节使用相同的缩放因子）
        if onnx_action_scale.ndim == 0 or len(onnx_action_scale) == 1:
            scale_value = float(onnx_action_scale.flatten()[0])
            print(f"⚠️  Info: action_scale is scalar ({scale_value}), expanding to {self.joint_config.K_OBS_DOF}D array for all joints")
            return np.full(self.joint_config.K_OBS_DOF, scale_value, dtype=np.float32)
        
        # 处理数组情况：验证维度
        if len(onnx_action_scale) != self.joint_config.K_OBS_DOF:
            raise RuntimeError(
                f"ONNX metadata action_scale length mismatch: "
                f"scale={len(onnx_action_scale)}, expected={self.joint_config.K_OBS_DOF}. "
                f"If you want to use a single scale for all joints, provide a scalar value."
            )
        
        return onnx_action_scale
    
    def load_default_joint_positions(self, reorder_to_controller: bool = False) -> np.ndarray:
        """
        Load default_joint_positions from ONNX metadata
        
        Args:
            reorder_to_controller: If True, reorder to Controller order; otherwise keep ONNX order
            
        Returns:
            Default joint positions array
        """
        if not self.is_onnx_model:
            raise RuntimeError("Cannot load default_joint_positions: Not an ONNX model")
        
        if self._metadata_loader is None:
            raise RuntimeError("ONNX metadata loader not initialized")
        
        onnx_default_joint_pos = self._metadata_loader.get_default_joint_pos()
        
        if onnx_default_joint_pos is None:
            raise RuntimeError(
                "default_joint_pos not found in ONNX metadata. "
                "Please ensure the ONNX model includes default_joint_pos in metadata."
            )
        
        if len(onnx_default_joint_pos) != self.joint_config.K_OBS_DOF:
            raise RuntimeError(
                f"ONNX metadata default_joint_pos length mismatch: "
                f"pos={len(onnx_default_joint_pos)}, expected={self.joint_config.K_OBS_DOF}"
            )
        
        default_pos = np.array(onnx_default_joint_pos, dtype=np.float32)
        
        if reorder_to_controller:
            if self._onnx_joint_names is None:
                raise RuntimeError(
                    "joint_names not found in ONNX metadata. "
                    "Cannot reorder default_joint_pos without joint_names."
                )
            
            onnx_name_to_index = {name: idx for idx, name in enumerate(self._onnx_joint_names)}
            reordered_default_pos = np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
            missing_joints = []
            
            for i, joint_name in enumerate(self.joint_config.OBS_JOINT_NAMES):
                if joint_name in onnx_name_to_index:
                    onnx_idx = onnx_name_to_index[joint_name]
                    reordered_default_pos[i] = default_pos[onnx_idx]
                else:
                    missing_joints.append(joint_name)
            
            if missing_joints:
                raise RuntimeError(
                    f"Some joints not found in ONNX metadata: {missing_joints}. "
                    f"ONNX joint names: {self._onnx_joint_names}"
                )
            
            return reordered_default_pos
        
        return default_pos
    
    def get_run_path(self) -> Optional[str]:
        """
        Get run path from ONNX metadata
        
        Returns:
            Run path string (e.g., W&B run path), or None if not found
        """
        if not self.is_onnx_model or self._metadata_loader is None:
            return None
        
        return self._metadata_loader.get_run_path()
    
    def get_observation_names(self) -> Optional[List[str]]:
        """
        Get observation names from ONNX metadata
        
        Returns:
            List of observation names, or None if not found
        """
        if not self.is_onnx_model or self._metadata_loader is None:
            return None
        
        return self._metadata_loader.get_observation_names()
    
    def get_command_names(self) -> Optional[List[str]]:
        """
        Get command names from ONNX metadata
        
        Returns:
            List of command names, or None if not found
        """
        if not self.is_onnx_model or self._metadata_loader is None:
            return None
        
        return self._metadata_loader.get_command_names()
    
