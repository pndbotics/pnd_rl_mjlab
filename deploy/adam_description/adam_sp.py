import numpy as np


JOINT_KP = {
    "hipPitch_Left": 300.0,
    "hipRoll_Left": 600.0,
    "hipYaw_Left": 300.0,
    "kneePitch_Left": 300.0,
    "anklePitch_Left": 130.0,
    "ankleRoll_Left": 70.0,
    "hipPitch_Right": 300.0,
    "hipRoll_Right": 600.0,
    "hipYaw_Right": 200.0,
    "kneePitch_Right": 300.0,
    "anklePitch_Right": 130.0,
    "ankleRoll_Right": 70.0,
    "waistRoll": 400.0,
    "waistPitch": 400.0,
    "waistYaw": 400.0,
    "shoulderPitch_Left": 60.0,
    "shoulderRoll_Left": 60.0,
    "shoulderYaw_Left": 60.0,
    "elbow_Left": 60.0,
    "wristYaw_Left": 20.0,
    "wristPitch_Left": 20.0,
    "wristRoll_Left": 20.0,
    "shoulderPitch_Right": 60.0,
    "shoulderRoll_Right": 60.0,
    "shoulderYaw_Right": 60.0,
    "elbow_Right": 60.0,
    "wristYaw_Right": 20.0,
    "wristPitch_Right": 20.0,
    "wristRoll_Right": 20.0,
}
JOINT_DAMPING = {
    "hipPitch_Left": 7.0,
    "hipRoll_Left": 10.0,
    "hipYaw_Left": 2.0,
    "kneePitch_Left": 7.0,
    "anklePitch_Left": 3.5,
    "ankleRoll_Left": 2.0,
    "hipPitch_Right": 7.0,
    "hipRoll_Right": 10.0,
    "hipYaw_Right": 2.0,
    "kneePitch_Right": 7.0,
    "anklePitch_Right": 3.5,
    "ankleRoll_Right": 2.0,
    "waistRoll": 11.0,
    "waistPitch": 11.0,
    "waistYaw": 11.0,
    "shoulderPitch_Left": 3.0,
    "shoulderRoll_Left": 3.0,
    "shoulderYaw_Left": 3.0,
    "elbow_Left": 3.0,
    "wristYaw_Left": 1.0,
    "wristPitch_Left": 1.0,
    "wristRoll_Left": 1.0,
    "shoulderPitch_Right": 3.0,
    "shoulderRoll_Right": 3.0,
    "shoulderYaw_Right": 3.0,
    "elbow_Right": 3.0,
    "wristYaw_Right": 1.0,
    "wristPitch_Right": 1.0,
    "wristRoll_Right": 1.0,
}

DEFAULT_ANGLES = np.array(
    [-0.32, 0.0,  -0.18,  0.66,  -0.39, 0.0,
     -0.32, 0.0,   0.18,  0.66,  -0.39, 0.0,
     0.0,  0.0, 0.0,
     0.0,  0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
     0.0, -0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
    ]
)



ACTION_SCALE = 0.5

JOINT_NAMES = list(JOINT_KP.keys())
# Link names from adam_sp_motor.xml (body name attributes, depth-first order)
BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_roll_link", "waist_pitch_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link",
]