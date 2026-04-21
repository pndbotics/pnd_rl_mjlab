"""
Adam SP RL Controller with ROS2 communication
基于ROS2与pnd_deploy_ros2通信的Adam SP机器人控制器
Velocity Flat任务专用
"""
import rclpy
import numpy as np
import threading
import os
import sys
from typing import Optional
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node

from pndrobotstatepub.msg import RobotState, JointStateCmd, Imu
from sensor_msgs.msg import Joy

# Import velocity_flat modules
from velocity_flat.config import Config
from velocity_flat.joint_config import JointConfig
from velocity_flat.observation import ObservationProcessor
from velocity_flat.models import ModelManager
from velocity_flat.onnx_metadata_manager import ONNXMetadataManager
from velocity_flat.joint_command_publisher import JointCommandPublisher


class RlController(Node):
    """Modular RL Controller for Adam SP robot (ROS2 version) - Velocity Flat task
    
    该控制器通过ROS2与pnd_deploy_ros2通信：
    - 订阅: robot_state_actual (RobotState), imu_data (Imu), joy (Joy)
    - 发布: joint_state_cmd (JointStateCmd)
    """
    
    def __init__(self):
        super().__init__('adam_sp_rl_controller')
        
        self.project_path = self._get_project_path()
        
        # Initialize configuration
        self.config = Config(self.project_path)
        self.joint_config = JointConfig
        
        # Initialize model manager
        self.model_manager = ModelManager(self.config)
        
        # Initialize metadata manager
        self.metadata_manager = ONNXMetadataManager(
            self.config.policy_path,
            self.joint_config,
            self.model_manager.is_onnx_model
        )
        
        # Load joint PD gains from ONNX metadata
        self._load_joint_pd_from_metadata()
        
        # Load action scales from ONNX metadata
        self.action_scales = self.metadata_manager.load_action_scales()
        
        # Load default joint positions (use from joint_config, which matches training)
        self.default_joint_positions = self.joint_config.DEFAULT_JOINT_POSITIONS.copy()
        
        # Initialize observation processor
        self.obs_processor = ObservationProcessor(
            self.config,
            default_joint_positions=self.default_joint_positions
        )
        self.obs_num = self.obs_processor.obs_num
        
        # Initialize input array for model inference
        self.input_array = np.zeros(self.obs_num, dtype=np.float32)
        
        # Control state
        self.control_enabled = False
        self.rl_counter = 0
        self.counter_step = 0
        
        # Thread locks
        self.robot_state_mutex = threading.Lock()
        self.control_mutex = threading.Lock()
        self.imu_mutex = threading.Lock()
        self.current_state = None
        self.current_imu = None
        
        # Timing parameters
        self.enable_timing_log = os.environ.get('ENABLE_TIMING_LOG', 'false').lower() == 'true'
        self.enable_interval_log = os.environ.get('ENABLE_INTERVAL_LOG', 'false').lower() == 'true'
        self.need_warmup = True
        
        # ROS2 publisher
        self.jointcmd_pub_ = self.create_publisher(JointStateCmd, 'joint_state_cmd', 10)
        
        # Initialize command publisher
        self.command_publisher = JointCommandPublisher(
            self.joint_config,
            self.config,
            self.jointcmd_pub_,
            self.control_mutex,
            clock=self.get_clock() if hasattr(self, 'get_clock') else None
        )
        self.command_publisher.set_enable_interval_log(self.enable_interval_log)
        
        # Setup subscribers
        self._setup_subscribers()
        
        # Main control loop timer (400Hz = 0.0025s)
        timer_period = 0.0025
        self.timer = self.create_timer(timer_period, self.Control)
        
        self.get_logger().info("Adam SP RL Controller initialized (Velocity Flat)")
        self.get_logger().info(f"  Observation dimension: {self.obs_num}")
        self.get_logger().info(f"  Joint DOF: {self.joint_config.K_OBS_DOF}")
        self.get_logger().info(f"  Policy path: {self.config.policy_path}")
    
    def _get_project_path(self) -> str:
        """Get project path from environment or default location"""
        if 'PND_PROJECT_PATH' in os.environ:
            return os.environ['PND_PROJECT_PATH']
        return os.path.dirname(os.path.abspath(__file__))
    
    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers"""
        # Subscribe to robot state (from pnd_deploy_ros2)
        self.robotstate_sub_ = self.create_subscription(
            RobotState, "robot_state_actual", self.getRobotState, 10)
        
        # Subscribe to joystick input
        self.joy_sub_ = self.create_subscription(Joy, "joy", self.joy_callback, 10)
        
        # Subscribe to IMU data (from pnd_deploy_ros2)
        self.imu_sub_ = self.create_subscription(Imu, "imu_data", self.getImu, 10)
        
        # Read control enabled state from environment variable
        self.control_enabled = os.environ.get('CONTROL_ENABLED', 'false').lower() == 'true'
    
    def _load_joint_pd_from_metadata(self) -> None:
        """Load joint PD gains from ONNX metadata"""
        if not self.model_manager.is_onnx_model:
            self.get_logger().warn("Not an ONNX model, skipping joint PD loading from metadata")
            return
        
        try:
            reordered_stiffness, reordered_damping = self.metadata_manager.load_joint_pd()
            joint_names = self.joint_config.OBS_JOINT_NAMES
            
            # Update config with joint PD gains
            for i, joint_name in enumerate(joint_names):
                self.config.joint_kp[joint_name] = float(reordered_stiffness[i])
                self.config.joint_kd[joint_name] = float(reordered_damping[i])
            
            self.get_logger().info(f"Loaded joint PD gains from ONNX metadata for {len(joint_names)} joints")
        except Exception as e:
            self.get_logger().error(f"Failed to load joint PD from metadata: {e}")
            raise
    
    def joy_callback(self, msg: Joy):
        """Handle joystick input"""
        B_BUTTON = 1
        LB_BUTTON = 6  # Xbox controller LB button
        
        if msg.buttons[LB_BUTTON] == 1:
            self.control_enabled = True
            self.counter_step = 0
            # Reset observation processor state
            self.obs_processor.action_last.fill(0.0)
            self.get_logger().info("Control enabled via joystick")
        
        if msg.buttons[B_BUTTON] == 1:
            self.control_enabled = False
            self.get_logger().info("Control disabled via joystick")
    
    def getRobotState(self, msg: RobotState):
        """Handle robot state updates from RobotState (ROS2)
        
        从pnd_deploy_ros2接收机器人状态信息
        """
        with self.robot_state_mutex:
            # Copy message to avoid thread safety issues
            modified_msg = RobotState()
            modified_msg.q_a = list(msg.q_a)
            modified_msg.q_dot_a = list(msg.q_dot_a)
            modified_msg.tau_a = list(msg.tau_a)
            
            # Handle wrist yaw joints (set to zero)
            wrist_yaw_ids = self.joint_config.get_wrist_yaw_ids()
            for joint_id in wrist_yaw_ids:
                if joint_id != -1:
                    ros2_position = joint_id + self.joint_config.K_BASE_NUM
                    if ros2_position < len(modified_msg.q_a):
                        modified_msg.q_a[ros2_position] = 0.0
                        modified_msg.q_dot_a[ros2_position] = 0.0
            
            self.current_state = modified_msg
    
    def getImu(self, msg: Imu):
        """Handle IMU updates from Imu (ROS2)
        
        从pnd_deploy_ros2接收IMU数据
        """
        with self.imu_mutex:
            self.current_imu = msg
    
    def _create_robot_state_adapter(self, robot_state: RobotState):
        """Create adapter object compatible with observation processor"""
        class RobotStateAdapter:
            def __init__(self, robot_state, controller_ref):
                self.robot_state = robot_state
                self.controller_ref = controller_ref
                
                class MotorState:
                    def __init__(self, q, dq):
                        self.q = q
                        self.dq = dq
                
                class ImuState:
                    def __init__(self, imu_msg):
                        if imu_msg is not None:
                            rpy = [imu_msg.roll, imu_msg.pitch, imu_msg.yaw]
                            rot = R.from_euler('xyz', rpy)
                            quat_xyzw = rot.as_quat()
                            # Convert to [w, x, y, z] format
                            self.quaternion = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                            self.gyroscope = list(imu_msg.angular_velocity) if hasattr(imu_msg, 'angular_velocity') else [0.0, 0.0, 0.0]
                        else:
                            self.quaternion = [1.0, 0.0, 0.0, 0.0]
                            self.gyroscope = [0.0, 0.0, 0.0]
                
                with controller_ref.imu_mutex:
                    current_imu = controller_ref.current_imu
                self.imu_state = ImuState(current_imu)
                
                # Create motor_state from robot_state
                # ROS2 message has q_a, q_dot_a starting from base (6 DOF) + robot joints
                # We need to map to motor indices (0-24, excluding wrist yaw)
                self.motor_state = []
                wrist_yaw_ids = controller_ref.joint_config.get_wrist_yaw_ids()
                k_base_num = controller_ref.joint_config.K_BASE_NUM
                
                motor_idx = 0
                for motor_idx in range(25):  # 25 motors total (excluding base)
                    if motor_idx not in wrist_yaw_ids:
                        ros2_idx = motor_idx + k_base_num
                        if ros2_idx < len(robot_state.q_a):
                            q = robot_state.q_a[ros2_idx]
                            dq = robot_state.q_dot_a[ros2_idx] if ros2_idx < len(robot_state.q_dot_a) else 0.0
                        else:
                            q = 0.0
                            dq = 0.0
                        self.motor_state.append(MotorState(q, dq))
                    else:
                        # For wrist yaw joints, add dummy state (will be skipped in observation)
                        self.motor_state.append(MotorState(0.0, 0.0))
        
        return RobotStateAdapter(robot_state, self)
    
    def warmup_models(self):
        """Warmup models with dummy inputs"""
        try:
            # Create dummy observation
            if self.model_manager.is_onnx_model and self.model_manager.model_input_dim is not None:
                dummy_obs = np.zeros(self.model_manager.model_input_dim, dtype=np.float32)
            else:
                dummy_obs = np.zeros(self.obs_num, dtype=np.float32)
            
            # Warmup model
            self.model_manager.warmup(dummy_obs, num_iterations=20)
            
            self.need_warmup = False
            self.rl_counter = 0
            self.counter_step = 0
            self.get_logger().info("Model warmup completed")
        except Exception as e:
            self.get_logger().error(f"Model warmup failed: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute actions using policy model
        
        Args:
            obs: Observation array (obs_num dimension)
            
        Returns:
            Raw action output from model (23D)
        """
        try:
            if obs.shape[0] != self.obs_num:
                raise ValueError(f"Observation dimension mismatch: expected {self.obs_num}, got {obs.shape[0]}")
            
            self.input_array[:] = obs
            
            # Run inference (velocity policies don't need counter_step)
            model_output_raw = self.model_manager.inference_policy(self.input_array, counter_step=None)
            
            # For velocity policies, output is single array of actions
            if isinstance(model_output_raw, (list, tuple)):
                # If multiple outputs, take first one (actions)
                actions = model_output_raw[0] if len(model_output_raw) > 0 else np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
            else:
                actions = model_output_raw
            
            # Ensure correct shape
            if actions.ndim > 1:
                actions = actions.flatten()
            
            if len(actions) != self.joint_config.K_OBS_DOF:
                raise ValueError(f"Action dimension mismatch: expected {self.joint_config.K_OBS_DOF}, got {len(actions)}")
            
            self.rl_counter += 1
            self.counter_step += 1
            
            return actions.astype(np.float32)
            
        except Exception as e:
            self.get_logger().error(f"Error in compute_actions: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
    
    def process_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """
        Process raw actions from model output to joint position commands
        
        Args:
            raw_actions: Raw actions from model (23D, relative to default positions)
            
        Returns:
            Processed joint position commands (23D, absolute positions)
        """
        # Apply action scales
        scaled_actions = raw_actions * self.action_scales
        
        # Add default joint positions to get absolute positions
        joint_positions = scaled_actions + self.default_joint_positions
        
        return joint_positions.astype(np.float32)
    
    def Control(self):
        """Main control loop (400Hz)"""
        with self.robot_state_mutex:
            robot_state = self.current_state
        if robot_state is None:
            return
        
        # Model warmup
        if self.need_warmup:
            self.warmup_models()
            return
        
        if self.control_enabled:
            # Run inference at 50Hz (every 8 cycles at 400Hz)
            if not hasattr(self, 'control_frequency_counter'):
                self.control_frequency_counter = 0
            
            self.control_frequency_counter += 1
            
            if self.control_frequency_counter % 8 == 0:
                # Create robot state adapter
                robot_state_adapter = self._create_robot_state_adapter(robot_state)
                
                # Get velocity commands from config
                velocity_commands = self.config.velocity_flat_velocity_commands
                
                # Compute observation
                obs = self.obs_processor.compute_obs(
                    robot_state_adapter,
                    velocity_commands=velocity_commands
                )
                
                # Compute actions
                raw_actions = self.compute_actions(obs)
                
                # Process actions to joint positions
                joint_positions = self.process_actions(raw_actions)
                
                # Update last action for next observation
                self.obs_processor.update_action_last(raw_actions)
                
                # Store for publishing (will be published every cycle)
                self._last_joint_positions = joint_positions
                self._last_joint_velocities = np.zeros(self.joint_config.K_OBS_DOF, dtype=np.float32)
            
            # Publish joint commands every cycle (400Hz)
            if hasattr(self, '_last_joint_positions'):
                self.command_publisher.publish(
                    self._last_joint_positions,
                    self._last_joint_velocities
                )


if __name__ == '__main__':
    rclpy.init()
    rl_controller = RlController()
    try:
        rclpy.spin(rl_controller)
    except KeyboardInterrupt:
        rl_controller.get_logger().info("Shutting down controller...")
    finally:
        rl_controller.destroy_node()
        rclpy.shutdown()
