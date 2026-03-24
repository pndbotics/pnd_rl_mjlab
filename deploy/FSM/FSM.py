from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.adam_beyond_mimic.AdamBeyondMimic import AdamBeyondMimic
from FSM.FSMState import *
from common.ctrlcomp import *
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.beyond_mimic_policy = AdamBeyondMimic(state_cmd, policy_output)
        
        print("initalized all policies!!!")
        
        self.cur_policy = self.passive_mode
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        if self.FSMmode == FSMMode.NORMAL:
            self.cur_policy.run()
            nextPolicyName = self.cur_policy.checkChange()

            if nextPolicyName != self.cur_policy.name:
                self.FSMmode = FSMMode.CHANGE
                self.cur_policy.exit()
                self.get_next_policy(nextPolicyName)
                print("Switched to ", self.cur_policy.name_str)

        elif self.FSMmode == FSMMode.CHANGE:
            self.cur_policy.enter()
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()

    def get_next_policy(self, policy_name:FSMStateName):
        if(policy_name == FSMStateName.PASSIVE):
            self.cur_policy = self.passive_mode
        elif((policy_name == FSMStateName.FIXEDPOSE)):
            self.cur_policy = self.fixed_pose_1
        elif((policy_name == FSMStateName.LOCOMODE)):
            self.cur_policy = self.loco_policy
        elif((policy_name == FSMStateName.ADAM_BEYOND_MIMIC)):
            self.cur_policy = self.beyond_mimic_policy
        else:
            pass
            
        
        