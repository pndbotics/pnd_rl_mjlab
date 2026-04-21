#!/usr/bin/env python3
"""
Generate deploy.yaml for Adam SP velocity policy deployment
从训练配置中提取参数并生成 deploy.yaml 文件
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import training configurations
from mjlab.asset_zoo.robots.adam_sp.adam_sp_constant import (
    ADAM_SP_HIPPITCH_KNEEPITCH_ACTUATOR,
    ADAM_SP_HIPROLL_ACTUATOR,
    ADAM_SP_HIPYAW_ACTUATOR,
    ADAM_SP_ANKLEPITCH_ACTUATOR,
    ADAM_SP_ANKLEROLL_ACTUATOR,
    ADAM_SP_WAIST_ACTUATOR,
    ADAM_SP_ARM_SHOULDER_ACTUATOR,
    ADAM_SP_ARM_ELBOW_ACTUATOR,
    ADAM_SP_ACTION_SCALE,
    HOME_KEYFRAME,
)
from deploy.robots.adam_sp.velocity_flat.joint_config import JointConfig
import numpy as np


def get_actuator_for_joint(joint_name: str):
    """Get actuator configuration for a joint"""
    actuators = [
        ADAM_SP_HIPPITCH_KNEEPITCH_ACTUATOR,
        ADAM_SP_HIPROLL_ACTUATOR,
        ADAM_SP_HIPYAW_ACTUATOR,
        ADAM_SP_ANKLEPITCH_ACTUATOR,
        ADAM_SP_ANKLEROLL_ACTUATOR,
        ADAM_SP_WAIST_ACTUATOR,
        ADAM_SP_ARM_SHOULDER_ACTUATOR,
        ADAM_SP_ARM_ELBOW_ACTUATOR,
    ]
    
    for actuator in actuators:
        import re
        for pattern in actuator.target_names_expr:
            if re.match(pattern, joint_name):
                return actuator
    return None


def generate_deploy_yaml():
    """Generate deploy.yaml content"""
    
    # Get joint names in observation order (23 joints)
    obs_joint_names = JointConfig.OBS_JOINT_NAMES
    default_joint_pos = JointConfig.DEFAULT_JOINT_POSITIONS
    
    # Extract stiffness and damping for each joint
    stiffness_list = []
    damping_list = []
    action_scale_list = []
    
    for joint_name in obs_joint_names:
        actuator = get_actuator_for_joint(joint_name)
        if actuator:
            stiffness_list.append(actuator.stiffness)
            damping_list.append(actuator.damping)
        else:
            print(f"Warning: No actuator found for joint {joint_name}")
            stiffness_list.append(0.0)
            damping_list.append(0.0)
        
        # Get action scale
        if joint_name in ADAM_SP_ACTION_SCALE:
            action_scale_list.append(ADAM_SP_ACTION_SCALE[joint_name])
        else:
            print(f"Warning: No action scale found for joint {joint_name}")
            action_scale_list.append(0.0)
    
    # Format lists for YAML
    def format_list(lst, decimals=3, per_line=15):
        """Format list for YAML output"""
        lines = []
        for i in range(0, len(lst), per_line):
            chunk = lst[i:i+per_line]
            line = ", ".join(f"{x:.{decimals}f}" if isinstance(x, float) else str(x) for x in chunk)
            if i + per_line < len(lst):
                line += ","
            lines.append(line)
        return "\n            ".join(lines)
    
    # Generate YAML content
    yaml_content = f"""# Adam SP Velocity Policy Deployment Configuration
# Generated from training configuration
# This file provides fallback parameters when ONNX metadata_props are not available

# Joint ID mapping (23 observed joints, excluding wrist joints)
# Note: This is for reference only - actual mapping handled by Controller
joint_ids_map: {list(range(23))}

# Control step size (seconds) - 50Hz control frequency
step_dt: 0.02

# Joint stiffness (Kp) - PD controller gains
# Order: left_leg(6) -> right_leg(6) -> waist(3) -> left_arm(4) -> right_arm(4)
stiffness: [
            {format_list(stiffness_list, decimals=1)}
]

# Joint damping (Kd) - PD controller gains
# Order: same as stiffness
damping: [
            {format_list(damping_list, decimals=1)}
]

# Default joint positions (radians)
# Order: same as stiffness/damping
# These match the training environment HOME_KEYFRAME
default_joint_pos: [
            {format_list(default_joint_pos.tolist(), decimals=3)}
]

# Command configuration
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]  # Linear velocity x range (m/s)
      lin_vel_y: [-0.5, 0.5]   # Linear velocity y range (m/s)
      ang_vel_z: [-1.0, 1.0]   # Angular velocity z range (rad/s)
      heading: null

# Action configuration
actions:
  JointPositionAction:
    clip: null
    joint_names: [.*]  # Match all joints
    # Action scale factors (from model output to joint position)
    # Formula: scale = 0.25 * effort_limit / stiffness
    # Order: same as stiffness/damping
    scale: [
            {format_list(action_scale_list, decimals=3)}
]
    # Action offset (default joint positions)
    # Order: same as stiffness/damping
    offset: [
            {format_list(default_joint_pos.tolist(), decimals=3)}
]
    joint_ids: null

# Observation configuration
observations:
  # Base angular velocity (3D)
  base_ang_vel:
    params: {{}}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  
  # Projected gravity (3D)
  projected_gravity:
    params: {{}}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  
  # Velocity commands (3D)
  velocity_commands:
    params: {{ command_name: base_velocity }}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  
  # Relative joint positions (23D)
  joint_pos_rel:
    params: {{}}
    clip: null
    scale: [
            {format_list([1.0] * 23, decimals=1)}
]
    history_length: 1
  
  # Relative joint velocities (23D)
  joint_vel_rel:
    params: {{}}
    clip: null
    scale: [
            {format_list([1.0] * 23, decimals=1)}
]
    history_length: 1
  
  # Last action (23D)
  last_action:
    params: {{}}
    clip: null
    scale: [
            {format_list([1.0] * 23, decimals=1)}
]
    history_length: 1
"""
    
    return yaml_content


def main():
    """Main function"""
    try:
        yaml_content = generate_deploy_yaml()
        
        # Output file path
        output_dir = project_root / "deploy" / "robots" / "adam_sp" / "config" / "policy" / "velocity" / "v0" / "params"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "deploy.yaml"
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Successfully generated deploy.yaml at: {output_file}")
        print(f"  - Stiffness values: {len(stiffness_list)} joints")
        print(f"  - Damping values: {len(damping_list)} joints")
        print(f"  - Action scales: {len(action_scale_list)} joints")
        print(f"  - Default positions: {len(default_joint_pos)} joints")
        
    except Exception as e:
        print(f"Error generating deploy.yaml: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
