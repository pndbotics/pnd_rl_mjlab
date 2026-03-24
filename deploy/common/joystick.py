from common.path_config import PROJECT_ROOT

import pygame
from pygame.locals import *
from enum import IntEnum

class JoystickButton(IntEnum):
    # 注意：这里的“名字”会被仿真控制逻辑引用（例如 deploy_mujoco.py 里用到 R1/L3）。
    # 为了兼容不同手柄命名，这里允许同一个键值有多个别名（例如 R1/LT）。
    # Face buttons
    A = 0
    B = 1
    X = 2
    Y = 3

    # Shoulder / trigger (按你的当前映射)
    L1 = 4
    RT = 4      # alias (有些手柄把这个当 RT)
    R1 = 5
    LT = 5      # alias (有些手柄把这个当 LT)

    # Stick press
    L3 = 6
    LB = 6      # alias
    R3 = 7
    RB = 7      # alias

    HOME = 8
    SELECT = 10
    START = 11

    # D-pad 可能被映射成 hat，不一定是 button；如你的手柄是 button 再取消注释并改值
    # UP = 11
    # DOWN = 12
    # LEFT = 13
    # RIGHT = 14

class JoyStick:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No joystick connected!")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.button_count = self.joystick.get_numbuttons()
        self.button_states = [False] * self.button_count  
        self.button_pressed = [False] * self.button_count  
        self.button_released = [False] * self.button_count 

        self.axis_count = self.joystick.get_numaxes()
        self.axis_states = [0.0] * self.axis_count
        
        self.hat_count = self.joystick.get_numhats()
        self.hat_states = [(0, 0)] * self.hat_count
        
        
    def update(self):
        """update joystick state"""
        pygame.event.pump()  
        self.button_released = [False] * self.button_count
        
        for i in range(self.button_count):
            current_state = self.joystick.get_button(i) == 1
            if self.button_states[i] and not current_state:
                self.button_released[i] = True
            self.button_states[i] = current_state

        for i in range(self.axis_count):
            self.axis_states[i] = self.joystick.get_axis(i)
        
        for i in range(self.hat_count):
            self.hat_states[i] = self.joystick.get_hat(i)

    def is_button_pressed(self, button_id):
        """detect button pressed"""
        if 0 <= button_id < self.button_count:
            return self.button_states[button_id]
        return False

    def is_button_released(self, button_id):
        """detect button released"""
        if 0 <= button_id < self.button_count:
            return self.button_released[button_id]
        return False

    def get_axis_value(self, axis_id):
        """get joystick axis value"""
        if 0 <= axis_id < self.axis_count:
            return self.axis_states[axis_id]
        return 0.0

    def get_hat_direction(self, hat_id=0):
        """get joystick hat direction"""
        if 0 <= hat_id < self.hat_count:
            return self.hat_states[hat_id]
        return (0, 0)