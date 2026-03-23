"""Evaluation script for trained Panda Pick-and-Place models.

Usage:
    python evaluate.py --model models/SAC_dense_20260215/best_model.zip --episodes 10
    python evaluate.py --model models/SAC_dense_20260215/best_model.zip --render
    python evaluate.py --help
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DDPG, PPO, SAC, TD3

import environments  # noqa: F401  (triggers Gymnasium registration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Panda Pick-and-Place agent.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--algo", type=str, default="SAC", choices=["DDPG", "PPO", "SAC", "TD3"], help="Algorithm used")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true", help="Render the environment (human mode)")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"], help="Reward type")
    parser.add_argument("--her", action="store_true", help="Use GoalEnv (for models trained with HER)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=150,
        help="Maximum steps per episode (default: 150)",
    )
    parser.add_argument("--record-video", action="store_true", help="Record evaluation episodes as MP4 videos")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save recorded videos (default: videos)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algo_map = {"DDPG": DDPG, "PPO": PPO, "SAC": SAC, "TD3": TD3}
    algo_cls = algo_map[args.algo]

    # Create environment
    if args.record_video:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None
    env_id = "PandaPickAndPlaceGoal-v0" if args.her else "PandaPickAndPlace-v0"
    env = gym.make(
        env_id,
        render_mode=render_mode,
        reward_type=args.reward_type,
        max_episode_steps=args.episode_steps,
    )
    if args.record_video:
        env = RecordVideo(
            env,
            video_folder=args.video_dir,
            name_prefix=f"{args.algo}_{args.reward_type}",
            video_length=1000,
            fps=25

        )
        print(f"Recording videos to: {args.video_dir}/")

    # Load model
    print(f"Loading model: {args.model}")
    model = algo_cls.load(args.model, env=env)

    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    successes = []

    print(f"\nEvaluating for {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        is_success = info.get("is_success", False)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(is_success)

        status = "SUCCESS" if is_success else "FAIL"
        print(f"  Episode {ep + 1:3d}: reward={total_reward:8.2f}  steps={steps:4d}  [{status}]")

    # Aggregate statistics
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    success_rate = np.mean(successes)

    print(f"\n{'=' * 50}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean reward:  {rewards.mean():8.2f} +/- {rewards.std():6.2f}")
    print(f"  Mean length:  {lengths.mean():8.1f} +/- {lengths.std():6.1f}")
    print(f"  Success rate: {success_rate * 100:6.1f}%")
    print(f"{'=' * 50}")

    env.close()


if __name__ == "__main__":
    main()
