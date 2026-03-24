"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
import tyro

from mjlab.rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    motion_file: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    wandb_run_path: str | None = None
    log_dir_name: str | None = None
    """Override run directory name (e.g. new_dof_pos). If None, use timestamp."""
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    is_tracking_task = (
        "motion" in cfg.env.commands
        and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
    )

    if is_tracking_task:
        if not cfg.motion_file:
            raise ValueError(
                "For tracking tasks, --motion-file must be set to a local motion file path."
            )
        motion_path = Path(cfg.motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.motion_file = str(motion_path)
        print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    log_root_path = log_dir.parent

    resume_path: Path | None = None
    if cfg.agent.resume:
        if cfg.wandb_run_path is not None:
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            if rank == 0:
                run_id = resume_path.parent.name
                checkpoint_name = resume_path.name
                cached_str = "cached" if was_cached else "downloaded"
                print(
                    f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
                    f"(run: {run_id}, {cached_str})"
                )
        else:
            resume_path = get_checkpoint_path(
                log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
            )

    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = OnPolicyRunner

    runner_kwargs = {}
    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,
        init_at_random_ep_len=True
    )

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    if args.log_dir_name:
        log_dir_name = args.log_dir_name
    else:
        log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.agent.run_name:
            log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task_id, args, log_dir)
    else:
        import torchrunx

        logging.basicConfig(level=logging.INFO)

        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
