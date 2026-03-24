"""PND Adam SP 23 DOF constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ADAM_SP_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "adam_sp" / "xmls" / "adam_sp_23_dof.xml"
)
assert ADAM_SP_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ADAM_SP_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_SP_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Adam SP).

ADAM_SP_HIPPITCH_KNEEPITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  stiffness=300.0,
  damping=7.0,
  effort_limit=230.0,
  armature=0.13426,
)
ADAM_SP_HIPROLL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
  ),
  stiffness=600.0,
  damping=10.0,
  effort_limit=180.0,
  armature=0.281573,
)
ADAM_SP_HIPYAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
  ),
  stiffness=300.0,
  damping=2.0,
  effort_limit=105.0,
  armature=0.23409,
)
ADAM_SP_ANKLEPITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
  ),
  stiffness=130.0,
  damping=3.5,
  effort_limit=80.0,
  armature=0.0549,
)
ADAM_SP_ANKLEROLL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_roll_joint",
  ),
  stiffness=70.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.0549,
)
ADAM_SP_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
  ),
  stiffness=400.0,
  damping=11.0,
  effort_limit=150.0,
  armature=0.23409,
)
ADAM_SP_ARM_SHOULDER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=60.0,
  damping=3.0,
  effort_limit=65.0,
  armature=0.01,
)
ADAM_SP_ARM_ELBOW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
  ),
  stiffness=60.0,
  damping=3.0,
  effort_limit=30.0,
  armature=0.01,
)




##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.89),
  joint_pos={
    ".*_hip_pitch_joint": -0.32,
    ".*_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.18,
    "right_hip_yaw_joint": 0.18,
    ".*_knee_joint": 0.66,
    ".*_ankle_pitch_joint": -0.39,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": -0.3,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-9]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-9]_collision$": 1},
  friction={r"^(left|right)_foot[1-9]_collision$": (0.6,)},
) 

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-9]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-9]_collision$": 1},
  friction={r"^(left|right)_foot[1-9]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-9]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

ADAM_SP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_SP_HIPPITCH_KNEEPITCH_ACTUATOR,
    ADAM_SP_HIPROLL_ACTUATOR,
    ADAM_SP_HIPYAW_ACTUATOR,
    ADAM_SP_ANKLEPITCH_ACTUATOR,
    ADAM_SP_ANKLEROLL_ACTUATOR,
    ADAM_SP_WAIST_ACTUATOR,
    ADAM_SP_ARM_SHOULDER_ACTUATOR,
    ADAM_SP_ARM_ELBOW_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_adam_sp_robot_cfg() -> EntityCfg:
  """Get a fresh Adam SP robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ADAM_SP_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_adam_sp_robot_cfg())

  viewer.launch(robot.spec.compile())
