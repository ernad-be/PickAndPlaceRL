"""Training script for the Panda Pick-and-Place environment.

Usage:
    python train.py --algo SAC --timesteps 500000
    python train.py --algo PPO --timesteps 1000000
    python train.py --help
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)

import environments  # noqa: F401  (triggers Gymnasium registration)


# ---------------------------------------------------------------------------
# Custom TensorBoard callback – logs per-episode metrics from the info dict.
# ---------------------------------------------------------------------------
class DetailedTensorBoardCallback(BaseCallback):
    """Log rich per-episode metrics to TensorBoard.

    Metrics logged under ``custom/`` or ``reward_components/``:
        - episode success rate (rolling window)
        - episode touch & grasp rates
        - mean distances (grip→object, object→target)
        - mean grip forces (left / right)
        - mean object height
        - individual reward components (reach, touch, grasp, lift, place, success)
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._rolling_window = rolling_window

        # Per-step accumulators for the *current* episode
        self._ep_info_buffer: dict[str, list[float]] = defaultdict(list)

        # Rolling episode-level history for windowed stats
        self._ep_history: dict[str, list[float]] = defaultdict(list)

    # ---- helpers ----
    def _aggregate_episode(self) -> dict[str, float]:
        """Aggregate step-level buffer into a single episode summary."""
        ep_summary: dict[str, float] = {}

        # Boolean flags → fraction of episode steps where True
        for key in ("object_touched", "object_grasped", "is_success"):
            vals = self._ep_info_buffer.get(key)
            if vals:
                ep_summary[key] = float(np.mean(vals))

        # Scalar means
        for key in (
            "dist_grip_to_obj",
            "dist_obj_to_target",
            "left_grip_force",
            "right_grip_force",
            "object_height",
        ):
            vals = self._ep_info_buffer.get(key)
            if vals:
                ep_summary[key] = float(np.mean(vals))

        # Reward components -> sum over the episode
        for key, vals in self._ep_info_buffer.items():
            if key.startswith("reward/") and vals:
                ep_summary[key] = float(np.sum(vals))

        # Terminal success (was the *last* step a success?)
        success_vals = self._ep_info_buffer.get("is_success")
        if success_vals:
            ep_summary["episode_success"] = float(success_vals[-1])

        return ep_summary

    def _log_to_tensorboard(self) -> None:
        """Write rolling-window statistics to TensorBoard."""
        # Success rate (rolling)
        if "episode_success" in self._ep_history:
            self.logger.record("custom/success_rate", float(np.mean(self._ep_history["episode_success"])))

        # Touch & grasp fraction within episode (rolling mean)
        for flag, tb_name in [
            ("object_touched", "custom/touch_fraction"),
            ("object_grasped", "custom/grasp_fraction"),
        ]:
            if flag in self._ep_history:
                self.logger.record(tb_name, float(np.mean(self._ep_history[flag])))

        # Distance metrics (rolling mean of episode means)
        for key, tb_name in [
            ("dist_grip_to_obj", "custom/dist_grip_to_obj"),
            ("dist_obj_to_target", "custom/dist_obj_to_target"),
            ("object_height", "custom/object_height"),
        ]:
            if key in self._ep_history:
                self.logger.record(tb_name, float(np.mean(self._ep_history[key])))

        # Grip forces
        for key, tb_name in [
            ("left_grip_force", "custom/left_grip_force"),
            ("right_grip_force", "custom/right_grip_force"),
        ]:
            if key in self._ep_history:
                self.logger.record(tb_name, float(np.mean(self._ep_history[key])))

        # Reward component sums (rolling mean of per-episode sums)
        for key, vals in self._ep_history.items():
            if key.startswith("reward/") and vals:
                self.logger.record(f"reward_components/{key.split('/')[-1]}", float(np.mean(vals)))

    def _flush_episode(self) -> None:
        """Aggregate current-episode step data, push to history, and log."""
        if not self._ep_info_buffer:
            return

        ep_summary = self._aggregate_episode()

        # Push to rolling history
        for k, v in ep_summary.items():
            self._ep_history[k].append(v)
            if len(self._ep_history[k]) > self._rolling_window:
                self._ep_history[k] = self._ep_history[k][-self._rolling_window :]

        self._log_to_tensorboard()

        # Reset per-episode buffer
        self._ep_info_buffer = defaultdict(list)

    # ---- callback interface ----
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Accumulate step-level info
            for key in (
                "is_success",
                "dist_grip_to_obj",
                "dist_obj_to_target",
                "object_touched",
                "object_grasped",
                "left_grip_force",
                "right_grip_force",
                "object_height",
            ):
                if key in info:
                    self._ep_info_buffer[key].append(float(info[key]))

            # Dynamically collect any reward/ component keys
            for key, value in info.items():
                if key.startswith("reward/"):
                    self._ep_info_buffer[key].append(float(value))

            # Flush on episode boundary
            if dones is not None and dones[i]:
                self._flush_episode()

        return True


ALGO_MAP = {
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

# Default hyperparameters per algorithm
DEFAULT_HYPERPARAMS = {
    "PPO": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "SAC": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 512,
        "tau": 0.005,
        "gamma": 0.98,
        "learning_starts": 2000,
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
    },
    "TD3": {
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "batch_size": 512,
        "tau": 0.005,
        "gamma": 0.98,
        "learning_starts": 10_000,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
    },
    "DDPG": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 512,
        "tau": 0.001,
        "gamma": 0.98,
        "learning_starts": 10_000,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Panda Pick-and-Place agent.")
    parser.add_argument("--algo", type=str, default="SAC", choices=ALGO_MAP.keys(), help="RL algorithm (default: SAC)")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps (default: 500000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments (default: 1)")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"], help="Reward type")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=150,
        help="Maximum steps per episode (default: 150)",
    )
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Stop after this many evaluations with no improvement (default: 50, 0=disabled)",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Stop when mean eval reward reaches this value (default: 0=disabled)",
    )
    return parser.parse_args()


def make_env(
    reward_type: str = "dense",
    episode_steps: int = 150,
) -> gym.Env:
    """Create and wrap a single environment instance."""
    env_id = "PandaPickAndPlace-v0"
    env = gym.make(
        env_id,
        reward_type=reward_type,
        max_episode_steps=episode_steps,
    )
    return Monitor(env)


def main() -> None:
    args = parse_args()

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{args.reward_type}_{timestamp}"

    print(f"{'=' * 60}")
    print(f"Training: {run_name}")
    print(f"Algorithm: {args.algo}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Reward type: {args.reward_type}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Seed: {args.seed}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Early stop patience: {args.patience if args.patience > 0 else 'disabled'}")
    print(f"Reward threshold: {args.reward_threshold if args.reward_threshold > 0 else 'disabled'}")
    print(f"{'=' * 60}")


    # Build algorithm kwargs
    algo_cls = ALGO_MAP[args.algo]
    algo_kwargs = DEFAULT_HYPERPARAMS[args.algo].copy()
    algo_kwargs["seed"] = args.seed
    algo_kwargs["verbose"] = 0
    algo_kwargs["tensorboard_log"] = args.log_dir

    # Select policy type
    policy_type = "MlpPolicy"

    # Create training environment
    if args.n_envs == 1:
        env = make_env(
            reward_type=args.reward_type,
            episode_steps=args.episode_steps,
        )
    else:
        env = make_vec_env(
            lambda: make_env(
                reward_type=args.reward_type,
                episode_steps=args.episode_steps,
            ),
            n_envs=args.n_envs,
            seed=args.seed,
        )

    # Create evaluation environment
    eval_env = make_env(
        reward_type=args.reward_type,
        episode_steps=args.episode_steps,
    )

    # Create model
    model = algo_cls(policy_type, env, **algo_kwargs)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Policy: {model.policy.__class__.__name__}")

    # Callbacks
    # -- Build early-stopping callback chain (evaluated after each eval round) --
    stop_callbacks: list[BaseCallback] = []
    if args.reward_threshold > 0:
        stop_callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1))
    if args.patience > 0:
        stop_callbacks.append(
            StopTrainingOnNoModelImprovement(max_no_improvement_evals=args.patience, min_evals=10, verbose=1)
        )
    # Chain them: if *any* fires, training stops.
    callback_after_eval = CallbackList(stop_callbacks) if stop_callbacks else None

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, run_name),
        log_path=os.path.join(args.log_dir, run_name),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=0,
        callback_after_eval=callback_after_eval,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(args.save_dir, run_name, "checkpoints"),
        name_prefix="model",
        verbose=0,
    )
    tb_detail_callback = DetailedTensorBoardCallback(rolling_window=100)
    callbacks = CallbackList([eval_callback, checkpoint_callback, tb_detail_callback])

    # Train
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=run_name, progress_bar=True)

    # Save final model
    final_path = os.path.join(args.save_dir, run_name, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
