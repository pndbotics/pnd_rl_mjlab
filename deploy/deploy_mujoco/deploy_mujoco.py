import sys
from pathlib import Path
import threading
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import os
from common.yaml_utils import load_yaml
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation, FSMStateName
from common.joystick import JoyStick, JoystickButton


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """关节 PD：位置/速度跟踪力矩。"""
    return (target_q - q) * kp + (target_dq - dq) * kd


height = 1.9
hanged = True

def keyboard_callback(keycode):
    global height
    global hanged
    if chr(keycode) == '9':
        hanged = not hanged
        print("hanged: ", hanged)
    elif chr(keycode) == '8':
        height += 0.05
        print("height: ", height)
    elif chr(keycode) == '7':
        height -= 0.05
        print("height: ", height)

def apply_force(m: mujoco.MjModel, d: mujoco.MjData, height: float):
    body_xpos = d.xpos[15]
    body_xmat = d.xmat[15].reshape(3, 3)
    world_point = body_xpos + body_xmat.dot(np.array([0,0,0.47]))
    x0 = np.array([0, 0, 2])
    x0[2] = height
    delta_x = x0 - world_point
    distance = np.linalg.norm(delta_x)
    direction = delta_x / distance
    v = np.sum(d.qvel[:3] * direction)
    f = np.zeros(3)
    if distance > 1:
        f = (100*(distance)-30*v)*direction
    f[2] += mujoco.mj_getTotalmass(m) * 9.81
    d.qfrc_applied[:] = 0
    
    mujoco.mj_applyFT(m,d,f,np.zeros(3),world_point,15, d.qfrc_applied)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    config = load_yaml(mujoco_yaml_path)
    xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    joystick = JoyStick()
    Running = True
    with mujoco.viewer.launch_passive(m, d, key_callback=keyboard_callback) as viewer:
        while viewer.is_running() and Running:
            try:
                # 跳舞/行走等策略需要站在地面上，不施加悬挂力，否则会漂浮
                standing_policies = (FSMStateName.ADAM_BEYOND_MIMIC, FSMStateName.LOCOMODE)
                if hanged and FSM_controller.cur_policy.name not in standing_policies:
                    apply_force(m, d, height)
                else:
                    d.qfrc_applied[:] = 0
                if(joystick.is_button_pressed(JoystickButton.SELECT)):
                    Running = False
                joystick.update()
                if joystick.is_button_released(JoystickButton.B):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                if joystick.is_button_released(JoystickButton.START):
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                if joystick.is_button_pressed(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.RB):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                if joystick.is_button_released(JoystickButton.A) :
                    print("BEYOND_MIMIC")
                    state_cmd.skill_cmd = FSMCommand.BEYOND_MIMIC
                if joystick.is_button_released(JoystickButton.Y):
                    state_cmd.skill_cmd = FSMCommand.LOCO
                
                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    inference_thread = threading.Thread(target=FSM_controller.run)
                    inference_thread.start()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
            except ValueError as e:
                print(str(e))
            except Exception as e:
                print(f"Error: {str(e)}")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        