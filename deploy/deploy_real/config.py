import os

from common.yaml_utils import load_yaml


class Config:
    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        real_yaml_path = os.path.join(current_dir, "config", "real.yaml")
        config = load_yaml(real_yaml_path)
        self.net = config["net"]
        self.num_joints = config["num_joints"]
        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]
        self.control_dt = config["control_dt"]
        self.error_over_time = config["error_over_time"]
