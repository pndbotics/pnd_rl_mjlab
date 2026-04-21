"""
Action Processor
Handles action scaling, reordering, and processing
"""
import numpy as np
from typing import Optional
from .onnx_metadata_manager import ONNXMetadataManager


class ActionProcessor:
    """Processes actions: scaling, reordering, and safety checks"""
    
    def __init__(self, joint_config, config, metadata_manager: Optional[ONNXMetadataManager] = None):
        """
        Initialize action processor
        
        Args:
            joint_config: JointConfig instance
            config: Config instance
            metadata_manager: ONNXMetadataManager instance (optional)
        """
        self.joint_config = joint_config
        self.config = config
        self.metadata_manager = metadata_manager
        
        self.training_action_scales = None
        self.default_joint_positions = None
        self.use_reference = False
        self.action_offset_from_metadata = None
        
        if metadata_manager is not None and metadata_manager.is_onnx_model:
            self._load_from_metadata()
    
    def _load_from_metadata(self):
        """Load action scales, default joint positions, and use_reference flag from metadata"""
        if self.metadata_manager is None:
            return
        
        try:
            self.training_action_scales = self.metadata_manager.load_action_scales()
            self.default_joint_positions = self.metadata_manager.load_default_joint_positions(reorder_to_controller=False)
            self.use_reference = self.metadata_manager.get_use_reference()
            self.action_offset_from_metadata = self.metadata_manager.load_action_offset(reorder_to_controller=False)
        except Exception as e:
            print(f"❌ ERROR: Failed to load action parameters from metadata: {e}")
            raise
    
    def process_action(self, model_output: np.ndarray, ref_joint_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process model output to final joint command
        
        Supports two action modes:
        - use_reference=True: final_joint_cmd = model_output * action_scale + ref_joint_pos
          (ref_joint_pos comes from model output, supports both beyond_mimic and OpenTrack)
        - use_reference=False: final_joint_cmd = model_output * action_scale + default_joint_positions
          (uses default joint positions from metadata)
        
        All inputs are in ONNX order, then reordered to Controller order.
        
        Args:
            model_output: Raw model output in ONNX order
            ref_joint_pos: Reference joint positions from model output (ONNX order, optional)
                          Required if use_reference=True
            
        Returns:
            Final joint command in Controller order
        """
        if self.training_action_scales is None or self.default_joint_positions is None:
            raise RuntimeError("Action processor not initialized: missing action scales or default positions")
        
        # Determine which offset to use
        if self.use_reference:
            # use_reference mode: use reference joint positions as offset (from model output)
            # Supports both beyond_mimic and OpenTrack formats
            if ref_joint_pos is not None:
                offset_onnx_order = ref_joint_pos
            elif self.action_offset_from_metadata is not None:
                # Fallback to metadata action_offset if available (backward compatibility)
                offset_onnx_order = self.action_offset_from_metadata
            else:
                raise RuntimeError(
                    "use_reference=True but ref_joint_pos not provided and action_offset not in metadata. "
                    "Please provide ref_joint_pos from model output."
                )
        else:
            # use_reference=False mode: use default joint positions from metadata
            offset_onnx_order = self.default_joint_positions
        
        # Apply scaling and offset
        mlp_out_scaled_onnx_order = model_output * self.training_action_scales + offset_onnx_order
        
        # Reorder to Controller order
        if self.metadata_manager is not None:
            mlp_out_scaled = self.metadata_manager.reorder_to_controller(mlp_out_scaled_onnx_order)
        else:
            mlp_out_scaled = mlp_out_scaled_onnx_order
        
        return mlp_out_scaled
    
    def check_safety(self, model_output: np.ndarray) -> np.ndarray:
        """
        Check and clip model output for safety
        
        Args:
            model_output: Raw model output
            
        Returns:
            Clipped model output if exceeds safe range
        """
        max_model_safe_range = 50.0
        if np.any(np.abs(model_output) > max_model_safe_range):
            print(f"⚠️  CRITICAL: Model output exceeds safe range (max={max_model_safe_range:.1f}): "
                  f"min={model_output.min():.4f}, max={model_output.max():.4f}")
            print(f"   This may indicate model instability or feedback loop. Clipping model output for safety.")
            return np.clip(model_output, -max_model_safe_range, max_model_safe_range)
        return model_output

