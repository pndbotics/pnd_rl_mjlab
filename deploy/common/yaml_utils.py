"""Small helpers shared by deploy entrypoints and policies."""

import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
