"""
Joint Command Publisher
Handles building and publishing joint commands via ROS2
"""
import numpy as np
import time
from typing import Optional
from pndrobotstatepub.msg import JointStateCmd


class JointCommandPublisher:
    """Publishes joint commands to robot via ROS2"""
    
    def __init__(self, joint_config, config, publisher, control_mutex, clock=None):
        """
        Initialize joint command publisher
        
        Args:
            joint_config: JointConfig instance
            config: Config instance
            publisher: ROS2 Publisher instance
            control_mutex: Threading lock for control operations
            clock: ROS2 Clock instance (optional, for timing logs)
        """
        self.joint_config = joint_config
        self.config = config
        self.publisher = publisher
        self.control_mutex = control_mutex
        self.clock = clock
        
        self.wrist_yaw_ids = self.joint_config.get_wrist_yaw_ids()
        self.joint_names = self.joint_config.OBS_JOINT_NAMES
        self.num_motors = 25
        self.num_joints = self.joint_config.K_OBS_DOF
        
        self.last_jointcmd_publish_time = None
        self.enable_interval_log = False
    
    def set_enable_interval_log(self, enable: bool):
        """Enable/disable interval logging"""
        self.enable_interval_log = enable
    
    def publish(self, joint_positions: np.ndarray, joint_velocities: Optional[np.ndarray] = None) -> bool:
        """
        Publish joint commands
        
        Args:
            joint_positions: Joint positions in Controller order
            joint_velocities: Joint velocities in Controller order (optional)
            
        Returns:
            True if publish succeeded, False otherwise
        """
        if joint_velocities is None:
            joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        
        try:
            with self.control_mutex:
                # Create ROS2 JointStateCmd message
                # Note: ROS2 JointStateCmd.q_d should contain only robot joints (K_OBS_DOF), not base joints
                # The C++ code reads from segment(6, size), meaning it expects only robot joints
                # joint_positions is already in Controller order (K_OBS_DOF, excluding wrist yaw)
                msg = JointStateCmd()
                
                # Check joint PD gains for logging
                for obs_idx in range(min(self.num_joints, len(joint_positions))):
                    joint_name = self.joint_names[obs_idx]
                    kp = self.config.get_joint_kp(joint_name)
                    if kp == 0.0 or kp is None:
                        print(f"⚠️  CRITICAL: Joint {obs_idx} ({joint_name}) has Kp=0.0 - robot will NOT move!")
                        print(f"   This is likely why the robot is not moving. Check ONNX metadata for joint_stiffness.")
                
                # Set message fields (joint_positions is already in Controller order, excluding wrist yaw)
                msg.q_d = joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else list(joint_positions)
                msg.q_dot_d = joint_velocities.tolist() if isinstance(joint_velocities, np.ndarray) else list(joint_velocities)
                msg.tau_d = [0.0] * self.num_joints
                
                # Publish message
                self.publisher.publish(msg)
                success = True
                
                if self.enable_interval_log:
                    if self.clock is not None:
                        now_time = self.clock.now()
                        if self.last_jointcmd_publish_time is not None:
                            delta_ns = (now_time - self.last_jointcmd_publish_time).nanoseconds
                            delta_ms = delta_ns / 1e6
                            print(f"joint_state_cmd publish interval: {delta_ms:.3f} ms")
                        self.last_jointcmd_publish_time = now_time
                    else:
                        now_time = time.perf_counter()
                        if self.last_jointcmd_publish_time is not None:
                            delta_ms = (now_time - self.last_jointcmd_publish_time) * 1000
                            print(f"joint_state_cmd publish interval: {delta_ms:.3f} ms")
                        self.last_jointcmd_publish_time = now_time
                
                return success
        
        except Exception as e:
            print(f"Control error: {str(e)}")
            return False

