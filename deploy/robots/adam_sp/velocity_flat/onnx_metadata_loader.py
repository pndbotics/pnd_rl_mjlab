"""
ONNX Metadata Loader
Load configuration from ONNX model metadata
"""
import onnx
import os
import numpy as np
from typing import Dict, Any, Optional, List, Union


def parse_csv_string(csv_str: str) -> List[Union[float, str]]:
    """
    Parse CSV string to list of values
    
    Args:
        csv_str: CSV string like "1.0,2.0,3.0" or "item1,item2,item3"
        
    Returns:
        List of parsed values (float if numeric, str otherwise)
    """
    if not csv_str or csv_str.strip() == "":
        return []
    
    parts = csv_str.split(',')
    result = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Try to parse as float
        try:
            value = float(part)
            # Check if it's an integer
            if value.is_integer():
                result.append(int(value))
            else:
                result.append(value)
        except ValueError:
            # Not a number, keep as string
            result.append(part)
    
    return result


def parse_string_list(csv_str: str) -> List[str]:
    """
    Parse CSV string to list of strings
    
    Args:
        csv_str: CSV string like "item1,item2,item3"
        
    Returns:
        List of strings
    """
    if not csv_str or csv_str.strip() == "":
        return []
    
    return [s.strip() for s in csv_str.split(',') if s.strip()]


class ONNXMetadataLoader:
    """Load configuration from ONNX model metadata"""
    
    def __init__(self, onnx_path: str):
        """
        Initialize metadata loader
        
        Args:
            onnx_path: Path to ONNX model file
        """
        self.onnx_path = onnx_path
        self.metadata: Dict[str, Any] = {}
        self.model: Optional[onnx.ModelProto] = None
        self.load_metadata()
    
    def load_metadata(self) -> None:
        """Load metadata from ONNX model"""
        if not os.path.exists(self.onnx_path):
            print(f"⚠️  ONNX file not found: {self.onnx_path}")
            return
        
        try:
            self.model = onnx.load(self.onnx_path)
            
            # Extract metadata
            for prop in self.model.metadata_props:
                key = prop.key
                value = prop.value
                self.metadata[key] = value
            
            # If normalizer parameters are not in metadata, try to extract from model structure
            if "normalizer_mean" not in self.metadata or "normalizer_std" not in self.metadata:
                self._extract_normalizer_from_model()
            
        except Exception as e:
            print(f"⚠️  Failed to load ONNX metadata: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_normalizer_from_model(self) -> None:
        """Extract normalizer parameters from ONNX model structure or external data"""
        if self.model is None:
            return
        
        try:
            normalizer_mean = None
            normalizer_std = None
            
            for init in self.model.graph.initializer:
                name = init.name
                
                # Check for normalizer._mean
                if "normalizer._mean" in name or name == "normalizer._mean":
                    if init.data_location == onnx.TensorProto.DEFAULT:
                        raw_data = init.raw_data
                        if raw_data:
                            data = np.frombuffer(raw_data, dtype=np.float32)
                            normalizer_mean = data.copy()
                    elif init.data_location == onnx.TensorProto.EXTERNAL:
                        normalizer_mean = self._load_external_data(init, "normalizer._mean")
                
                # Check for normalizer._std or "add" (which might be std + eps)
                elif "normalizer._std" in name or name == "normalizer._std":
                    if init.data_location == onnx.TensorProto.DEFAULT:
                        raw_data = init.raw_data
                        if raw_data:
                            data = np.frombuffer(raw_data, dtype=np.float32)
                            normalizer_std = data.copy()
                    elif init.data_location == onnx.TensorProto.EXTERNAL:
                        normalizer_std = self._load_external_data(init, "normalizer._std")
                
                # Some models use "add" for normalizer._std + eps
                elif name == "add" and normalizer_std is None:
                    if normalizer_mean is not None:
                        shape = list(init.dims)
                        if len(shape) == 1 and shape[0] == len(normalizer_mean):
                            if init.data_location == onnx.TensorProto.DEFAULT:
                                raw_data = init.raw_data
                                if raw_data:
                                    data = np.frombuffer(raw_data, dtype=np.float32)
                                    normalizer_std = data.copy()
                            elif init.data_location == onnx.TensorProto.EXTERNAL:
                                normalizer_std = self._load_external_data(init, "add")
            
            if normalizer_mean is not None:
                self.metadata["normalizer_mean"] = normalizer_mean
            if normalizer_std is not None:
                self.metadata["normalizer_std"] = normalizer_std
                
        except Exception as e:
            print(f"⚠️  Failed to extract normalizer from model structure: {e}")
    
    def _load_external_data(self, init: onnx.TensorProto, expected_name: str) -> Optional[np.ndarray]:
        """Load data from external file"""
        try:
            external_data = {}
            for entry in init.external_data:
                external_data[entry.key] = entry.value
            
            location = external_data.get("location", external_data.get("Location", ""))
            offset_str = external_data.get("offset", external_data.get("Offset", "0"))
            length_str = external_data.get("length", external_data.get("Length", "0"))
            
            try:
                offset = int(offset_str)
                length = int(length_str)
            except (ValueError, TypeError):
                return None
            
            if not location or length == 0:
                return None
            
            onnx_dir = os.path.dirname(self.onnx_path)
            data_path = os.path.join(onnx_dir, location)
            
            if not os.path.exists(data_path):
                return None
            
            with open(data_path, 'rb') as f:
                f.seek(offset)
                raw_data = f.read(length)
            
            if len(raw_data) != length:
                return None
            
            dtype_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT64: np.int64,
            }
            
            dtype = dtype_map.get(init.data_type, np.float32)
            data = np.frombuffer(raw_data, dtype=dtype)
            
            if len(init.dims) > 0:
                shape = tuple(init.dims)
                data = data.reshape(shape)
            
            return data.flatten() if len(data.shape) > 1 else data
            
        except Exception as e:
            print(f"⚠️  Failed to load external data for {expected_name}: {e}")
            return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if metadata key exists"""
        return key in self.metadata
    
    def get_joint_names(self) -> Optional[List[str]]:
        """Get joint names from metadata"""
        value = self.get("joint_names")
        if value is None:
            return None
        
        if isinstance(value, str):
            return parse_string_list(value)
        elif isinstance(value, list):
            return value
        else:
            return None
    
    def get_joint_stiffness(self) -> Optional[np.ndarray]:
        """Get joint stiffness (Kp) from metadata"""
        value = self.get("joint_stiffness")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_joint_damping(self) -> Optional[np.ndarray]:
        """Get joint damping (Kd) from metadata"""
        value = self.get("joint_damping")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_default_joint_pos(self) -> Optional[np.ndarray]:
        """Get default joint positions from metadata"""
        value = self.get("default_joint_pos")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list,    np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_action_scale(self) -> Optional[np.ndarray]:
        """
        Get action scale from metadata
        
        支持多种格式：
        1. 字符串：CSV 格式，如 "1.0,2.0,3.0" 或单个值 "0.1"
        2. 列表/数组：多个值或单个值的列表
        3. 标量（数字）：单个数值，所有关节使用相同的缩放因子
        
        Returns:
            Action scale array (可能是标量或数组)
        """
        value = self.get("action_scale")
        if value is None:
            return None
        
        if isinstance(value, str):
            # 字符串格式：可能是 CSV 或单个值
            values = parse_csv_string(value)
            if len(values) == 0:
                return None
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            # 列表/数组格式
            return np.array(value, dtype=np.float32)
        elif isinstance(value, (int, float)):
            # 标量格式：单个数值
            return np.array([float(value)], dtype=np.float32)
        else:
            return None
    
    def get_normalizer_mean(self) -> Optional[np.ndarray]:
        """Get normalizer mean from metadata or model structure"""
        value = self.get("normalizer_mean")
        if value is None:
            if self.model is not None:
                self._extract_normalizer_from_model()
                value = self.get("normalizer_mean")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_normalizer_std(self) -> Optional[np.ndarray]:
        """Get normalizer std from metadata or model structure"""
        value = self.get("normalizer_std")
        if value is None:
            if self.model is not None:
                self._extract_normalizer_from_model()
                value = self.get("normalizer_std")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_observation_names(self) -> Optional[List[str]]:
        """Get observation names from metadata"""
        value = self.get("observation_names")
        if value is None:
            return None
        
        if isinstance(value, str):
            return parse_string_list(value)
        elif isinstance(value, list):
            return value
        else:
            return None
    
    def get_run_path(self) -> Optional[str]:
        """Get run path from metadata"""
        return self.get("run_path")
    
    def get_command_names(self) -> Optional[List[str]]:
        """Get command names from metadata"""
        value = self.get("command_names")
        if value is None:
            return None
        
        if isinstance(value, str):
            return parse_string_list(value)
        elif isinstance(value, list):
            return value
        else:
            return None
    
    def get_obs_dim(self) -> Optional[int]:
        """Get observation dimension from metadata"""
        value = self.get("obs_dim")
        if value is None:
            # Try to infer from normalizer_mean shape
            normalizer_mean = self.get_normalizer_mean()
            if normalizer_mean is not None:
                return len(normalizer_mean)
            return None
        
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return None
    
    def get_motion_length(self) -> Optional[int]:
        """Get motion length from metadata"""
        value = self.get("motion_length")
        if value is None:
            return None
        
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return None
    
    def get_action_offset(self) -> Optional[np.ndarray]:
        """Get action offset from metadata
        
        Action offset is used when use_reference=True mode (supports both beyond_mimic and OpenTrack).
        It represents the reference joint positions used as offset for actions.
        Note: When use_reference=True, ref_joint_pos from model output is preferred over action_offset.
        
        Returns:
            Action offset array in ONNX order, or None if not found
        """
        value = self.get("action_offset")
        if value is None:
            return None
        
        if isinstance(value, str):
            values = parse_csv_string(value)
            return np.array(values, dtype=np.float32)
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32)
        else:
            return None
    
    def get_use_reference(self) -> bool:
        """Get use_reference flag from metadata
        
        Supports both beyond_mimic and OpenTrack formats.
        When use_reference=True, actions are computed as: model_output * action_scale + ref_joint_pos
        where ref_joint_pos comes from model output (joint_pos, typically the 2nd output).
        
        Returns:
            True if use_reference is set to "true", False otherwise (defaults to False)
        """
        value = self.get("use_reference")
        if value is None:
            return False
        
        if isinstance(value, str):
            return value.lower() == "true"
        elif isinstance(value, bool):
            return value
        else:
            return False
    
    def get_motion_file(self) -> Optional[str]:
        """Get motion file path from metadata
        
        Returns:
            Motion file path string, or None if not found
        """
        return self.get("motion_file")
    
    def get_anchor_body_name(self) -> Optional[str]:
        """Get anchor body name from metadata
        
        Returns:
            Anchor body name string, or None if not found
        """
        return self.get("anchor_body_name")
    
    def get_body_names(self) -> Optional[List[str]]:
        """Get body names list from metadata
        
        Returns:
            List of body names, or None if not found
        """
        value = self.get("body_names")
        if value is None:
            return None
        
        if isinstance(value, str):
            return parse_string_list(value)
        elif isinstance(value, list):
            return value
        else:
            return None
    
    def get_policy_frame_stack(self) -> Optional[int]:
        """Get policy frame stack number from metadata
        
        Returns:
            Policy frame stack number, or None if not found
        """
        value = self.get("policy_frame_stack")
        if value is None:
            return None
        
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def get_critic_frame_stack(self) -> Optional[int]:
        """Get critic frame stack number from metadata
        
        Returns:
            Critic frame stack number, or None if not found
        """
        value = self.get("critic_frame_stack")
        if value is None:
            return None
        
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def print_summary(self) -> None:
        """Print metadata summary"""
        print("\n" + "=" * 80)
        print("ONNX Metadata Summary")
        print("=" * 80)
        
        if not self.metadata:
            print("  No metadata found")
            return
        
        # Group metadata by category
        categories = {
            "Model Info": ["run_path", "model_type", "obs_dim", "has_normalizer"],
            "Joint Config": ["joint_names", "default_joint_pos", "action_scale"],
            "PD Gains": ["joint_stiffness", "joint_damping"],
            "Normalizer": ["normalizer_mean", "normalizer_std"],
            "Observation": ["observation_names"],
            "Command": ["command_names"],
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if key in self.metadata:
                    value = self.metadata[key]
                    if isinstance(value, str):
                        if len(value) > 80:
                            print(f"  ✓ {key}: {value[:80]}...")
                        else:
                            print(f"  ✓ {key}: {value}")
                    else:
                        print(f"  ✓ {key}: {type(value).__name__}")
                else:
                    print(f"  ✗ {key}: Not found")
        
        # Print other metadata
        other_keys = [k for k in self.metadata.keys() if not any(k in keys for keys in categories.values())]
        if other_keys:
            print(f"\nOther Metadata:")
            for key in other_keys:
                value = self.metadata[key]
                if isinstance(value, str) and len(value) > 80:
                    print(f"  ✓ {key}: {value[:80]}...")
                else:
                    print(f"  ✓ {key}: {value}")
        
        print("=" * 80)
