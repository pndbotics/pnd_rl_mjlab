"""
RL Controller Package - Modular design

This package contains modular components for the RL controller:
- config: Configuration loading and management
- joint_config: Joint mapping and configuration
- observation: Observation data computation
- models: Model loading and inference
- onnx_metadata_manager: ONNX metadata management
- action_processor: Action processing
- joint_command_publisher: Joint command publishing
- utils: Utility functions (quaternion operations, matrix conversions, joint reordering)
- Controller: Main controller with modular design
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .config import Config
from .joint_config import JointConfig
from .observation import ObservationProcessor
from .models import ModelManager
from .onnx_metadata_manager import ONNXMetadataManager
from .action_processor import ActionProcessor
from .joint_command_publisher import JointCommandPublisher

# Import utility functions
from .utils import (
    quat_mul, quat_inv, quat_to_matrix, matrix_from_quat, yaw_quat,
    reorder_array_to_controller_order, reorder_array_to_onnx_order,
    euler_xyz_to_matrix, matrix_to_euler_xyz
)

__all__ = [
    'Config',
    'JointConfig',
    'ObservationProcessor',
    'ModelManager',
    'ONNXMetadataManager',
    'ActionProcessor',
    'JointCommandPublisher',
    # Utility functions
    'quat_mul',
    'quat_inv',
    'quat_to_matrix',
    'matrix_from_quat',
    'yaw_quat',
    'reorder_array_to_controller_order',
    'reorder_array_to_onnx_order',
    'euler_xyz_to_matrix',
    'matrix_to_euler_xyz',
]

