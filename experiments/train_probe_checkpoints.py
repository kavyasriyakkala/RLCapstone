"""
Train models with intermediate checkpoints for linear probing over time.

This is needed for probe figures like:
results/figures/linear_probes_k1_seed42.png

Final models alone cannot produce R^2-vs-training-step curves, so this script
saves checkpoints every --checkpoint-freq steps.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gridworld import ShortcutGridWorld


TARGETS = ("a2c", "dqn", "vanilla", "l2", "augmentation", "par")


class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig = env.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(int(np.prod(orig.shape)),),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten().astype(np.float32)


class ColorJitterWrapper(gym.ObservationWrapper):
    def __init__(self, env, jitter_strength=0.2):
        super().__init__(env)
        self.jitter_strength = jitter_strength
        self.observation_space = env.observation_space

    def observation(self, obs):
        obs = obs.copy().astype(np.float32)
        for ch in (1, 2):
            scale = np.random.uniform(1.0 - self.jitter_strength, 1.0 + self.jitter_strength)
            obs[ch] *= scale
        return np.clip(obs, 0.0, 1.0).astype(np.float32)


class PARRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, alpha=4.0, reference_reward=0.0):
        super().__init__(env)
        self.alpha = alpha
        self.reference_reward = reference_reward

    def reward(self, reward):
        centered = reward - self.reference_reward
        shaped = 2.0 / (1.0 + np.exp(-self.alpha * centered)) - 1.0
        return float(shaped)


def build_env(grid_size, k, target):
    env = ShortcutGridWorld(grid_size=grid_size, num_spurious=k, train_mode=True)
    if target == "augmentation":
        env = ColorJitterWrapper(env, jitter_strength=0.2)
    if target == "par":
        env = PARRewardWrapper(env, alpha=4.0, reference_reward=0.0)
    return FlattenWrapper(env)


def make_env(grid_size, k, target):
    def _init():
        return build_env(grid_size, k, target)

    return _init


def get_model(target, vec_env, seed):
    if target == "a2c":
        return A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=seed,
        )

    if target == "dqn":
        return DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=seed,
        )

    common_kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    policy_kwargs = dict(net_arch=[256, 256])
    if target == "l2":
        policy_kwargs["optimizer_kwargs"] = dict(weight_decay=1e-4)
    if target in ("vanilla", "l2", "augmentation", "par"):
        return PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **common_kwargs)

    raise ValueError(f"Unknown target: {target}")


def train_with_checkpoints(target, grid_size, k, seed, steps, checkpoint_freq):
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = f"results/probe_checkpoints/{target}/k{k}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(grid_size, k, target)])
    model = get_model(target, vec_env, seed)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=out_dir,
        name_prefix="model",
    )

    print(
        f"Training {target} with checkpoints | k={k} | seed={seed} | "
        f"steps={steps} | freq={checkpoint_freq}"
    )
    model.learn(total_timesteps=steps, callback=checkpoint_callback)
    model.save(os.path.join(out_dir, "final_model"))
    print(f"Saved checkpoints and final model to {out_dir}")


def parse_targets(raw):
    targets = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [target for target in targets if target not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Options: {TARGETS}")
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="a2c,dqn,l2,augmentation,par")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    args = parser.parse_args()

    for target in parse_targets(args.targets):
        train_with_checkpoints(
            target=target,
            grid_size=args.grid_size,
            k=args.k,
            seed=args.seed,
            steps=args.steps,
            checkpoint_freq=args.checkpoint_freq,
        )


if __name__ == "__main__":
    main()
