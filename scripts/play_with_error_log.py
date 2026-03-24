"""Play 脚本：实时记录各关节与 ref 的位置差与角度差，播放结束后汇总平均误差。"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rsl_rl.runners import OnPolicyRunner
from mjlab.rsl_rl.utils.exporter import export_policy_as_jit
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MultiMotionCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayWithErrorLogConfig:
    """Play 并记录关节误差的配置。"""

    agent: Literal["zero", "random", "trained"] = "trained"
    motion_file: str | None = None
    wandb_run_path: str | None = None
    checkpoint_file: str | None = None
    num_envs: int | None = 1
    """环境数量，建议用 1 以便得到清晰的单轨迹误差统计。"""
    device: str | None = None
    num_steps: int = 5000
    """总步数；播放将在达到该步数后停止并输出汇总。"""
    log_interval: int = 50
    """每隔多少步打印一次实时误差。"""
    output_csv: str | None = "play_joint_error_log.csv"
    """误差 CSV 保存路径；默认为 play_joint_error_log.csv。"""
    viewer: Literal["auto", "native", "viser"] = "auto"
    """可视化方式：auto 根据 DISPLAY 自动选择。"""
    trace_pt: bool = False


def _create_error_log_viewer(
    base_viewer_cls: type,
    env: Any,
    policy: Any,
    command: MotionCommand | MultiMotionCommand,
    joint_names: list[str],
    cfg: PlayWithErrorLogConfig,
    csv_path: Path,
    log_dir: Path | None,
) -> Any:
    """创建带误差记录功能的 viewer。"""

    num_joints = len(joint_names)
    device = env.unwrapped.device
    sum_abs_error = torch.zeros(num_joints, device=device)
    sum_squared_error = torch.zeros(num_joints, device=device)

    # Reward term 累加器（用于汇总平均 reward）
    reward_term_names: list[str] = []
    sum_reward_terms: list[float] = []
    reward_initialized = False
    values: list[float] = []
    total_reward = 0.0

    csv_file = open(csv_path, "w", encoding="utf-8")
    csv_header_written = False

    class ErrorLogViewer(base_viewer_cls):
        def step_simulation(self) -> None:
            super().step_simulation()
            nonlocal csv_header_written
            # 在 step 之后记录误差（command 已在 env.step 中更新）
            ref_pos = command.joint_pos
            act_pos = command.robot_joint_pos
            joint_dim = min(ref_pos.shape[1], act_pos.shape[1], num_joints)
            diff = ref_pos[:, :joint_dim] - act_pos[:, :joint_dim]
            abs_err = torch.abs(diff)
            mean_abs_per_joint = abs_err.mean(dim=0)
            sum_abs_error[:joint_dim] += mean_abs_per_joint
            sum_squared_error[:joint_dim] += (diff**2).mean(dim=0)

            # 记录 reward terms（tracking 任务使用的 reward）
            nonlocal reward_term_names, sum_reward_terms, reward_initialized, values, total_reward
            reward_manager = self.env.unwrapped.reward_manager
            values = []
            total_reward = 0.0
            if hasattr(reward_manager, "get_active_iterable_terms"):
                terms = reward_manager.get_active_iterable_terms(0)
                if terms:
                    if not reward_initialized:
                        reward_term_names = [name for name, _ in terms]
                        sum_reward_terms = [0.0] * len(reward_term_names)
                        reward_initialized = True
                    values = [v[0] for _, v in terms]
                    for i, v in enumerate(values):
                        sum_reward_terms[i] += v
                    total_reward = sum(values)

            if not csv_header_written:
                header = (
                    "step,"
                    + ",".join(f"{n}_rad,{n}_deg" for n in joint_names)
                )
                if reward_term_names:
                    header += "," + ",".join(
                        f"reward_{n}" for n in reward_term_names
                    ) + ",reward_total"
                csv_file.write(header + "\n")
                csv_header_written = True

            step = self._step_count
            if step % cfg.log_interval == 0:
                curr_mean = mean_abs_per_joint
                overall_rad = curr_mean.mean().item()
                overall_deg = math.degrees(overall_rad)
                reward_str = (
                    f" | total_reward: {total_reward:.4f}" if values else ""
                )
                print(
                    f"[Step {step}] 平均角度差(rad): {overall_rad:.4f} | "
                    f"平均角度差(deg): {overall_deg:.2f}{reward_str}"
                )

            row = [str(step)]
            for i in range(joint_dim):
                row.append(f"{mean_abs_per_joint[i].item():.6f}")
                row.append(f"{math.degrees(mean_abs_per_joint[i].item()):.4f}")
            if values:
                row.extend(f"{v:.6f}" for v in values)
                row.append(f"{total_reward:.6f}")
            csv_file.write(",".join(row) + "\n")
            csv_file.flush()

        def close(self) -> None:
            step_count = self._step_count
            super().close()
            csv_file.close()
            print(f"[INFO] 误差数据已写入 {csv_path}")

            if step_count == 0:
                return
            avg_abs_error = sum_abs_error / step_count
            avg_squared = sum_squared_error / step_count
            rmse = torch.sqrt(avg_squared)
            joint_dim = min(
                command.joint_pos.shape[1],
                command.robot_joint_pos.shape[1],
                num_joints,
            )
            print("\n" + "=" * 70)
            print("各关节与 Ref 的平均误差汇总")
            print("=" * 70)
            print(
                f"{'关节名':<30} {'平均|误差|(rad)':<18} {'平均|误差|(deg)':<18} {'RMSE(rad)':<14}"
            )
            print("-" * 70)
            for i in range(joint_dim):
                rad_val = avg_abs_error[i].item()
                deg_val = math.degrees(rad_val)
                rmse_val = rmse[i].item()
                print(
                    f"{joint_names[i]:<30} {rad_val:<18.6f} {deg_val:<18.4f} {rmse_val:<14.6f}"
                )
            overall_avg_rad = avg_abs_error[:joint_dim].mean().item()
            overall_avg_deg = math.degrees(overall_avg_rad)
            overall_rmse_rad = rmse[:joint_dim].mean().item()
            overall_rmse_deg = math.degrees(overall_rmse_rad)
            print("-" * 70)
            print(
                f"{'全关节平均':<30} {overall_avg_rad:<18.6f} {overall_avg_deg:<18.4f} {overall_rmse_rad:<14.6f}"
            )
            print("=" * 70)
            print(f"\n总步数: {step_count}")
            print(
                f"全关节平均 |角度差|: {overall_avg_rad:.6f} rad = {overall_avg_deg:.4f} deg"
            )
            print(
                f"全关节平均 RMSE: {overall_rmse_rad:.6f} rad = {overall_rmse_deg:.4f} deg"
            )

            # Reward 汇总
            if reward_term_names and sum_reward_terms and step_count > 0:
                print("\n" + "=" * 70)
                print("Reward 项平均汇总（tracking 任务）")
                print("=" * 70)
                print(f"{'Reward 项':<35} {'平均 step reward':<18}")
                print("-" * 70)
                for i, name in enumerate(reward_term_names):
                    avg = sum_reward_terms[i] / step_count
                    print(f"{name:<35} {avg:<18.6f}")
                total_avg = sum(sum_reward_terms) / step_count
                print("-" * 70)
                print(f"{'总 Reward 平均':<35} {total_avg:<18.6f}")
                print("=" * 70)

    return ErrorLogViewer(env, policy)


def run_play_with_error_log(task_id: str, cfg: PlayWithErrorLogConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # 检测 tracking task
    is_tracking_task = "motion" in env_cfg.commands and isinstance(
        env_cfg.commands["motion"], MotionCommandCfg
    )

    if not is_tracking_task:
        raise ValueError(
            f"play_with_error_log 仅支持 tracking 任务，当前任务 {task_id} 不是 tracking。"
        )

    motion_cmd_cfg = env_cfg.commands["motion"]
    assert isinstance(motion_cmd_cfg, MotionCommandCfg)

    if cfg.motion_file is None:
        raise ValueError("Tracking 任务需要 --motion-file 指定本地 motion npz 文件。")

    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion 文件不存在: {motion_path}")

    motion_cmd_cfg.motion_file = str(motion_path)
    motion_cmd_cfg.sampling_mode = "start"
    print(f"[INFO] 使用 motion 文件: {motion_cmd_cfg.motion_file}")

    log_dir: Path | None = None
    resume_path: Path | None = None

    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {resume_path}")
        print(f"[INFO] 加载 checkpoint: {resume_path.name}")
    elif TRAINED_MODE:
        if cfg.wandb_run_path is None:
            raise ValueError(
                "未提供 checkpoint_file 时需指定 wandb_run_path。"
            )
        log_root_path = (
            Path("logs") / "rsl_rl" / agent_cfg.experiment_name
        ).resolve()
        resume_path, _ = get_wandb_checkpoint_path(
            log_root_path, Path(cfg.wandb_run_path)
        )
        print(f"[INFO] 加载 checkpoint: {resume_path}")
    log_dir = resume_path.parent if resume_path else None

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = int(1e9)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        if cfg.trace_pt and log_dir:
            export_policy_as_jit(
                policy=runner.alg.policy,
                normalizer=runner.alg.policy.actor_obs_normalizer,
                path=str(log_dir),
                filename="policy.pt",
            )
        policy = runner.get_inference_policy(device=device)

    command = env.unwrapped.command_manager.get_term("motion")
    if not isinstance(command, (MotionCommand, MultiMotionCommand)):
        raise RuntimeError("命令不是 MotionCommand 或 MultiMotionCommand。")

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)

    # 默认 CSV 路径：log_dir 下或当前目录
    if cfg.output_csv:
        csv_path = Path(cfg.output_csv)
    else:
        csv_path = Path("play_joint_error_log.csv")
    if log_dir and not csv_path.is_absolute():
        csv_path = log_dir / csv_path.name
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    print("\n" + "=" * 60)
    print("开始播放（带可视化），实时记录各关节与 ref 的角度差")
    print(f"[INFO] CSV 将保存到: {csv_path}")
    print("=" * 60)

    obs = env.get_observations()
    env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

    base_cls = (
        NativeMujocoViewer if resolved_viewer == "native" else ViserPlayViewer
    )
    viewer = _create_error_log_viewer(
        base_cls, env, policy, command, joint_names, cfg, csv_path, log_dir
    )

    try:
        viewer.run(num_steps=cfg.num_steps)
    except RuntimeError as e:
        if resolved_viewer == "native" and (
            "viewer" in str(e).lower() or "window" in str(e).lower()
        ):
            print(
                "[WARN] Native viewer 失败，回退到 Viser web viewer。"
            )
            viewer = _create_error_log_viewer(
                ViserPlayViewer,
                env,
                policy,
                command,
                joint_names,
                cfg,
                csv_path,
                log_dir,
            )
            viewer.run(num_steps=cfg.num_steps)
        else:
            raise
    finally:
        env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        PlayWithErrorLogConfig,
        args=remaining_args,
        default=PlayWithErrorLogConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args, agent_cfg

    run_play_with_error_log(chosen_task, args)


if __name__ == "__main__":
    main()
