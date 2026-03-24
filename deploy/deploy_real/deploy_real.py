import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.ctrlcomp import *
from FSM.FSM import *
import numpy as np
import time
import pndbotics_sdk_py
import threading

from pndbotics_sdk_py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__LowState_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_ 
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowState_ 

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd, MotorMode
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()
        self.num_joints = config.num_joints
        self.control_dt = config.control_dt
        
        
        self.low_cmd = pnd_adam_msg_dds__LowCmd_(self.config.num_joints+2)
        self.low_state = pnd_adam_msg_dds__LowState_(self.config.num_joints+2)
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()
        
        # inital connection
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.LowStateAdamHandler, 10)
        
        self.wait_for_low_state()
        
        init_cmd(self.low_cmd)
        
        self.policy_output_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kps = np.zeros(self.num_joints, dtype=np.float32)
        self.kds = np.zeros(self.num_joints, dtype=np.float32)
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0,0,-1], dtype=np.float32)
        
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.running = True
        self.counter_over_time = 0
        self.print_flag = True
        
        
    def LowStateAdamHandler(self, msg: LowState_):
        self.low_state = msg
        # self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            print("waiting for low state...")
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    def print_state(self):
        print("qj: ", self.qj)
        print("dqj: ", self.dqj)
        print("quat: ", self.quat)
        print("ang_vel: ", self.ang_vel)
        print("gravity_orientation: ", self.gravity_orientation)

    def run(self):
        try:
            # if(self.counter_over_time >= config.error_over_time):
            #     raise ValueError("counter_over_time >= error_over_time")
            
            loop_start_time = time.time()
            self.counter_over_time += 1
            if self.remote_controller.is_button_pressed(KeyMap.B):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.A):
                self.state_cmd.skill_cmd = FSMCommand.BEYOND_MIMIC
            if self.remote_controller.is_button_pressed(KeyMap.Y):
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.RB):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            
            self.state_cmd.vel_cmd[0] =  self.remote_controller.ly
            self.state_cmd.vel_cmd[1] =  self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] =  self.remote_controller.rx * -1

            for i in range(15):
                self.qj[i] = self.low_state.motor_state[i].q
                self.dqj[i] = self.low_state.motor_state[i].dq
            for i in range(17, self.num_joints+2):
                self.qj[i-2] = self.low_state.motor_state[i].q
                self.dqj[i-2] = self.low_state.motor_state[i].dq

            # imu_state quaternion: w, x, y, z
            self.quat = self.low_state.imu_state.quaternion
            self.ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
            
            self.gravity_orientation = get_gravity_orientation_real(self.quat)
            
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = self.gravity_orientation.copy()
            self.state_cmd.ang_vel = self.ang_vel.copy()
            self.state_cmd.base_quat = self.quat
            
            # self.FSM_controller.run()
            if self.counter_over_time == int(self.config.control_dt / 0.002):
                self.inference_thread = threading.Thread(target=self.FSM_controller.run)
                self.inference_thread.start()
                self.counter_over_time = 0
                # self.FSM_controller.run()
                # self.print_state()
            policy_output_action = self.policy_output.actions.copy().reshape(-1)
            kps = self.policy_output.kps.copy().reshape(-1)
            kds = self.policy_output.kds.copy().reshape(-1)
            # Build low cmd
            
            
            for i in range(15):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
            for i in range(15, 17):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].kd = 8
                self.low_cmd.motor_cmd[i].tau = 0
            for i in range(17, self.num_joints+2):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i-2]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i-2]
                self.low_cmd.motor_cmd[i].kd = kds[i-2]
                self.low_cmd.motor_cmd[i].tau = 0
                
            # send the command
            # create_damping_cmd(controller.low_cmd) # only for debug
            self.send_cmd(self.low_cmd)
            
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            if(delta_time < 0.002):
                time.sleep(0.002 - delta_time)
        except ValueError as e:
            print(str(e))
        
        
if __name__ == "__main__":
    config = Config()
    # Initialize DDS communication
    ChannelFactoryInitialize(1, config.net)
    
    controller = Controller(config)
    controller.wait_for_low_state()
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
        except KeyboardInterrupt:
            break
    
    # create_damping_cmd(controller.low_cmd)
    # controller.send_cmd(controller.low_cmd)
    print("Exit")
    