"""
Model loading and inference module
"""
import torch
import torch.jit
import numpy as np
import os
from typing import Tuple, Optional

# Try to import onnxruntime for ONNX model support
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not available, ONNX models will not be supported")


# 优化Python运行时设置 - 单线程推理
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class ModelManager:
    """Manages PyTorch and ONNX models for RL policy"""
    
    def __init__(self, config):
        """
        Initialize model manager
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.mlp_model = None  # PyTorch JIT model or ONNX session
        self.onnx_session = None  # ONNX Runtime session
        self.is_onnx_model = False
        # Note: normalizer_mean/std are not used for ONNX models (embedded normalization)
        # Kept for backward compatibility with JIT models if needed
        self.normalizer_mean = None
        self.normalizer_std = None
        self.has_embedded_normalizer = False
        self.model_input_dim = None  # Actual model input dimension (for ONNX models)
        
        # Load model
        self.load_model()
        # Check if model has embedded normalizer
        self._check_embedded_normalizer()
        # Note: Normalizer is NOT loaded separately - ONNX models have embedded normalization
    
    def load_model(self) -> None:
        """Load PyTorch JIT or ONNX policy model"""
        try:
            policy_path = self.config.policy_path
            
            # Check if file exists
            if not os.path.exists(policy_path):
                raise FileNotFoundError(f"Policy model file not found: {policy_path}")
            
            # Determine model type by file extension
            file_ext = os.path.splitext(policy_path)[1].lower()
            
            if file_ext == '.onnx':
                # Load ONNX model
                if not ONNX_AVAILABLE:
                    raise RuntimeError("onnxruntime is not available. Please install it: pip install onnxruntime")
                
                self.onnx_session = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
                self.is_onnx_model = True
                self.mlp_model = self.onnx_session  # For compatibility
                
                # Get actual model input dimension from ONNX model
                # This is important for frame stack handling
                # ONNX input shape is typically [batch_size, feature_dim] or [batch_size, ...]
                if len(self.onnx_session.get_inputs()) > 0:
                    input_shape = self.onnx_session.get_inputs()[0].shape
                    # Calculate total input dimension (ignore batch dimension)
                    if len(input_shape) >= 2:
                        # Shape is [batch_size, feature_dim, ...] or [batch_size, feature_dim]
                        # For most RL models: [batch_size, feature_dim]
                        # Calculate total feature dimension (all dimensions except batch)
                        total_dim = 1
                        for dim in input_shape[1:]:  # Skip batch dimension (index 0)
                            if isinstance(dim, int) and dim > 0:
                                total_dim *= dim
                            elif dim is None or (isinstance(dim, str) and (dim.startswith('dim') or dim.startswith('batch'))):
                                # Dynamic dimension, cannot determine
                                total_dim = None
                                break
                        self.model_input_dim = total_dim
                    elif len(input_shape) == 1:
                        # Single dimension: could be [batch_size] or [feature_dim]
                        # Usually [batch_size], so feature_dim is 1 (scalar)
                        dim = input_shape[0]
                        if isinstance(dim, int) and dim > 0:
                            # If it's a large number, it's likely feature_dim, not batch_size
                            # But typically batch_size is 1, so this is ambiguous
                            # For safety, assume it's feature_dim if > 10, else it's batch_size
                            if dim > 10:
                                self.model_input_dim = dim
                            else:
                                # Likely batch_size, feature_dim is 1 (scalar input)
                                self.model_input_dim = 1
                        else:
                            self.model_input_dim = None
                    else:
                        # Empty shape or unknown
                        self.model_input_dim = None
                else:
                    self.model_input_dim = None
                
                output_names = [out.name for out in self.onnx_session.get_outputs()]
                self.onnx_output_names = output_names
                
                # Print model information
                self._print_model_info(policy_path)
                
            elif file_ext == '.pt':
                # Load PyTorch JIT model
                self.mlp_model = torch.jit.load(policy_path, map_location='cpu')
                self.mlp_model.eval()
                self.is_onnx_model = False
                
            else:
                raise ValueError(f"Unsupported model file format: {file_ext}. Supported formats: .pt (PyTorch JIT), .onnx (ONNX)")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load policy model: {str(e)}")
    
    # Removed load_normalizer() - ONNX models have embedded normalization
    # Raw observations are passed directly to ONNX models (no manual normalization needed)
    
    def _check_embedded_normalizer(self) -> None:
        """
        Check if the loaded model has embedded normalizer in its forward pass.
        
        For ONNX models: Always assume they have embedded normalization (matching RoboMimic_Deploy behavior).
        ONNX models handle normalization internally, so raw observations are passed directly.
        """
        if self.mlp_model is None:
            return
        
        try:
            # For ONNX models: always assume embedded normalization
            # Reference: /home/chenmt/workplace/RoboMimic_Deploy/policy/beyond_mimic/BeyondMimic.py
            # ONNX models in that implementation use raw observations without manual normalization
            if self.is_onnx_model:
                self.has_embedded_normalizer = True
                return
            
            # For PyTorch JIT models: check if they have embedded normalizer
            # Check if model has normalizer attributes (PolicyWithNormalizer wrapper)
            if hasattr(self.mlp_model, 'normalizer_mean') and hasattr(self.mlp_model, 'normalizer_std'):
                self.has_embedded_normalizer = True
                return
            
            # Try to inspect the model's code to see if it has normalization
            # For JIT models, we can check the code string
            if hasattr(self.mlp_model, 'code'):
                code_str = str(self.mlp_model.code)
                # Check for normalization patterns in the code
                if 'normalizer_mean' in code_str or 'normalizer_std' in code_str:
                    self.has_embedded_normalizer = True
                    return
            
            # JIT model without embedded normalizer
            self.has_embedded_normalizer = False
            pass
        except Exception as e:
            pass
            # Default to False if check fails
            self.has_embedded_normalizer = False
    
    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation using empirical normalizer (matching training environment)
        
        Args:
            obs: Raw observation array
            
        Returns:
            Normalized observation array
        """
        if self.normalizer_mean is not None and self.normalizer_std is not None:
            # Apply normalization: (obs - mean) / std
            # Add small epsilon to std to prevent division by zero
            eps = 1e-8
            normalized_obs = (obs - self.normalizer_mean) / (self.normalizer_std + eps)
            return normalized_obs.astype(np.float32)
        else:
            # No normalizer available, return as-is
            return obs
    
    def inference_policy(self, policy_input: np.ndarray, counter_step: Optional[int] = None, time_step: Optional[float] = None) -> np.ndarray:
        """
        Run policy model inference
        
        Supports both velocity policies and motion policies:
        - Velocity policies: single input "obs", single output "actions"
        - Motion policies: two inputs "obs" and "time_step", multiple outputs
        
        Args:
            policy_input: Policy input array (should already be normalized if normalizer is available)
            counter_step: Counter step (integer, for motion policies only)
            time_step: Time step (deprecated, kept for backward compatibility, for motion policies only)
            
        Returns:
            Action output (or list of outputs for multi-output models)
        """
        if self.mlp_model is None:
            raise RuntimeError("Policy model not loaded")
        
        if self.is_onnx_model:
            # ONNX model inference
            if self.onnx_session is None:
                raise RuntimeError("ONNX session not initialized")
            
            # Get all input names from ONNX model
            input_names = [inp.name for inp in self.onnx_session.get_inputs()]
            
            # Log input names for debugging
            if not hasattr(self, '_input_names_logged'):
                self._input_names_logged = True
            
            # Prepare observation dictionary
            observation = {}
            
            # Prepare observation input (always the first input)
            # Velocity policies: input_names=["obs"]
            # Motion policies: input_names=["obs", "time_step"]
            obs_input_name = input_names[0]  # Usually 'obs' or first input name
            if policy_input.ndim == 1:
                # Add batch dimension: reshape(1, -1) matches unsqueeze(0)
                observation[obs_input_name] = policy_input.reshape(1, -1).astype(np.float32)
            else:
                observation[obs_input_name] = policy_input.astype(np.float32)
            
            # Prepare time_step input (only for motion policies with 2+ inputs)
            # Velocity policies don't have time_step input
            if len(input_names) > 1:
                counter_input_name = input_names[1]  # Should be 'time_step' for motion policies
                
                # Use counter_step if provided (preferred)
                if counter_step is not None:
                    # Clamp time_step to valid range (matching exporter.py behavior)
                    # Note: motion_length should be set to time_step_total from exporter
                    # The clamping is done in the model, but we ensure it's a valid integer
                    time_step_value = int(counter_step)
                    observation[counter_input_name] = np.array([[time_step_value]], dtype=np.float32)
                elif time_step is not None:
                    # Fallback to time_step for backward compatibility
                    time_step_value = int(time_step)
                    observation[counter_input_name] = np.array([[time_step_value]], dtype=np.float32)
                else:
                    # Default to 0 if neither provided
                    observation[counter_input_name] = np.array([[0]], dtype=np.float32)
            
            # Run inference
            outputs = self.onnx_session.run(None, observation)
            
            # Handle output format
            # Velocity policies: single output "actions"
            # Motion policies: multiple outputs (actions, joint_pos, joint_vel, etc.)
            if len(outputs) == 1:
                # Single output (velocity policy) - return actions directly
                return outputs[0][0] if outputs[0].ndim > 1 else outputs[0]
            else:
                # Multiple outputs (motion policy) - return as tuple
                # Remove batch dimension if present
                processed_outputs = []
                for output in outputs:
                    if output.ndim > 1:
                        processed_outputs.append(output[0])
                    else:
                        processed_outputs.append(output)
                return tuple(processed_outputs)
        else:
            # PyTorch JIT model inference
            input_tensor = torch.from_numpy(policy_input).unsqueeze(0)
            with torch.no_grad():
                output_data = self.mlp_model(input_tensor)
            
            # Handle multi-output models
            if isinstance(output_data, (list, tuple)):
                return tuple(out[0].numpy() if torch.is_tensor(out) else out[0] for out in output_data)
            else:
                return output_data[0].numpy() if output_data.ndim > 1 else output_data.numpy()
    
    def _print_model_info(self, model_path: str) -> None:
        """
        Print ONNX model information including dimensions and key metadata
        
        Args:
            model_path: Path to the ONNX model file
        """
        if not self.is_onnx_model or self.onnx_session is None:
            return
        
        print("\n" + "=" * 80)
        print("📦 ONNX模型加载信息")
        print("=" * 80)
        print(f"模型路径: {model_path}")
        
        # Input information
        inputs = self.onnx_session.get_inputs()
        print(f"\n输入信息:")
        for i, inp in enumerate(inputs):
            input_shape = inp.shape
            shape_str = " × ".join(str(dim) if dim is not None and isinstance(dim, int) else str(dim) for dim in input_shape)
            print(f"  输入[{i}]: {inp.name}")
            print(f"    形状: [{shape_str}]")
            print(f"    类型: {inp.type}")
            
            # Calculate feature dimension (excluding batch dimension)
            if len(input_shape) >= 2:
                feature_dims = []
                for dim in input_shape[1:]:
                    if isinstance(dim, int) and dim > 0:
                        feature_dims.append(dim)
                if feature_dims:
                    total_feature_dim = 1
                    for d in feature_dims:
                        total_feature_dim *= d
                    print(f"    特征维度: {total_feature_dim}D")
                    if self.model_input_dim is not None:
                        print(f"    模型输入维度: {self.model_input_dim}D")
        
        # Output information
        outputs = self.onnx_session.get_outputs()
        print(f"\n输出信息:")
        for i, out in enumerate(outputs):
            output_shape = out.shape
            shape_str = " × ".join(str(dim) if dim is not None and isinstance(dim, int) else str(dim) for dim in output_shape)
            print(f"  输出[{i}]: {out.name}")
            print(f"    形状: [{shape_str}]")
            print(f"    类型: {out.type}")
            
            # Calculate output dimension
            if len(output_shape) >= 2:
                feature_dims = []
                for dim in output_shape[1:]:
                    if isinstance(dim, int) and dim > 0:
                        feature_dims.append(dim)
                if feature_dims:
                    total_output_dim = 1
                    for d in feature_dims:
                        total_output_dim *= d
                    print(f"    特征维度: {total_output_dim}D")
            elif len(output_shape) == 1:
                dim = output_shape[0]
                if isinstance(dim, int) and dim > 0:
                    print(f"    特征维度: {dim}D")
        
        print(f"\n输出数量: {len(outputs)}")
        print("=" * 80 + "\n")
    
    def warmup(self, dummy_input_policy: np.ndarray, num_iterations: int = 20) -> None:
        """
        Warmup policy model with dummy inputs
        
        Args:
            dummy_input_policy: Dummy input for policy
            num_iterations: Number of warmup iterations
        """
        for i in range(num_iterations):
            counter_step = i if self.is_onnx_model else None
            self.inference_policy(dummy_input_policy, counter_step=counter_step)

