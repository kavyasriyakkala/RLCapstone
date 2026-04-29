"""
Generate linear-probe-over-training figures from saved checkpoint series.

Input checkpoint layout:
results/probe_checkpoints/{target}/k{k}_seed{seed}/model_*_steps.zip

Output figure layout:
results/figures/linear_probes_{target}_k{k}_seed{seed}.png
"""
import argparse
import glob
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gridworld import ShortcutGridWorld


MODEL_SPECS = {
    "a2c": (A2C, "actor_critic"),
    "dqn": (DQN, "dqn"),
    "vanilla": (PPO, "actor_critic"),
    "l2": (PPO, "actor_critic"),
    "augmentation": (PPO, "actor_critic"),
    "par": (PPO, "actor_critic"),
}


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


def make_env(grid_size, k, train_mode):
    def _init():
        return FlattenWrapper(
            ShortcutGridWorld(
                grid_size=grid_size,
                num_spurious=k,
                train_mode=train_mode,
            )
        )

    return _init


def checkpoint_step(path):
    match = re.search(r"model_(\d+)_steps\.zip$", path)
    if not match:
        raise ValueError(f"Could not parse checkpoint step from {path}")
    return int(match.group(1))


def hidden_activation(model, obs, kind):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if kind == "actor_critic":
            features = model.policy.extract_features(
                obs_tensor,
                model.policy.features_extractor,
            )
            net = model.policy.mlp_extractor.policy_net
            act = net[0](features)
            if len(net) > 1:
                act = net[1](act)
            return act.squeeze(0).cpu().numpy()

        if kind == "dqn":
            features = model.policy.q_net.extract_features(
                obs_tensor,
                model.policy.q_net.features_extractor,
            )
            net = model.policy.q_net.q_net
            act = net[0](features)
            if len(net) > 1:
                act = net[1](act)
            return act.squeeze(0).cpu().numpy()

    raise ValueError(f"Unknown model kind: {kind}")


def extract_activations(model, env, kind, n_episodes):
    activations = []
    goal_positions = []
    spurious_positions = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            activations.append(hidden_activation(model, obs, kind))
            goal_positions.append(list(env.env.goal_pos))
            spurious_positions.append(list(env.env.spurious_positions[0]))

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return (
        np.asarray(activations),
        np.asarray(goal_positions),
        np.asarray(spurious_positions),
    )


def ridge_r2(activations, labels):
    alpha = 1.0
    x_mean = activations.mean(axis=0, keepdims=True)
    x_std = activations.std(axis=0, keepdims=True) + 1e-8
    x = (activations - x_mean) / x_std
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

    reg = alpha * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ labels)
    pred = x @ weights
    ss_res = np.sum((labels - pred) ** 2)
    ss_tot = np.sum((labels - labels.mean(axis=0)) ** 2)
    return max(0.0, float(1 - ss_res / (ss_tot + 1e-8)))


def probe_checkpoint(target, checkpoint_path, grid_size, k, n_episodes):
    loader, kind = MODEL_SPECS[target]
    vec_env = DummyVecEnv([make_env(grid_size, k, train_mode=True)])
    model = loader.load(checkpoint_path, env=vec_env)
    eval_env = FlattenWrapper(
        ShortcutGridWorld(
            grid_size=grid_size,
            num_spurious=k,
            train_mode=False,
        )
    )

    activations, goal_pos, spurious_pos = extract_activations(
        model,
        eval_env,
        kind,
        n_episodes,
    )
    return ridge_r2(activations, goal_pos), ridge_r2(activations, spurious_pos)


def plot_probe_curve(target, steps, goal_r2s, spurious_r2s, k, seed, out_path):
    combined = sorted(zip(steps, goal_r2s, spurious_r2s))
    steps, goal_r2s, spurious_r2s = zip(*combined)
    steps = np.asarray(steps)
    goal_r2s = np.asarray(goal_r2s)
    spurious_r2s = np.asarray(spurious_r2s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(steps, goal_r2s, color="green", linewidth=2, label="Goal position probe R^2")
    ax1.plot(
        steps,
        spurious_r2s,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Spurious feature probe R^2",
    )
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Probe R^2")
    ax1.set_title(f"Linear probe accuracy over training ({target}, seed {seed}, k={k})")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.05)

    diff = goal_r2s - spurious_r2s
    ax2.plot(steps, diff, color="purple", linewidth=2, label="Goal R^2 - Spurious R^2")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.fill_between(
        steps,
        diff,
        0,
        where=diff > 0,
        alpha=0.3,
        color="green",
        label="Goal > Spurious",
    )
    ax2.fill_between(
        steps,
        diff,
        0,
        where=diff < 0,
        alpha=0.3,
        color="red",
        label="Spurious > Goal",
    )
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("R^2 difference")
    ax2.set_title("Difference: Goal probe R^2 minus Spurious probe R^2")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


def run_target(target, grid_size, k, seed, n_episodes, checkpoint_stride):
    checkpoint_dir = f"results/probe_checkpoints/{target}/k{k}_seed{seed}"
    checkpoint_paths = sorted(
        glob.glob(os.path.join(checkpoint_dir, "model_*_steps.zip")),
        key=checkpoint_step,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. "
            "Run experiments/train_probe_checkpoints.py first."
        )
    checkpoint_paths = checkpoint_paths[::checkpoint_stride]

    steps = []
    goal_r2s = []
    spurious_r2s = []
    for path in checkpoint_paths:
        step = checkpoint_step(path)
        print(f"Probing {target} checkpoint step {step}...")
        goal_r2, spurious_r2 = probe_checkpoint(target, path, grid_size, k, n_episodes)
        steps.append(step)
        goal_r2s.append(goal_r2)
        spurious_r2s.append(spurious_r2)
        print(f"  goal={goal_r2:.3f} | spurious={spurious_r2:.3f} | diff={goal_r2 - spurious_r2:.3f}")

    os.makedirs("results/probes", exist_ok=True)
    np.savez(
        f"results/probes/linear_probe_curve_{target}_k{k}_seed{seed}.npz",
        steps=np.asarray(steps),
        goal_r2=np.asarray(goal_r2s),
        spurious_r2=np.asarray(spurious_r2s),
        diff=np.asarray(goal_r2s) - np.asarray(spurious_r2s),
    )
    plot_probe_curve(
        target,
        steps,
        goal_r2s,
        spurious_r2s,
        k,
        seed,
        f"results/figures/linear_probes_{target}_k{k}_seed{seed}.png",
    )


def parse_targets(raw):
    targets = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [target for target in targets if target not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Options: {sorted(MODEL_SPECS)}")
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="a2c,dqn,l2,augmentation,par")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help="Probe every Nth checkpoint. Use 5 for 25k-step spacing when checkpoints are every 5k.",
    )
    args = parser.parse_args()

    for target in parse_targets(args.targets):
        run_target(
            target,
            args.grid_size,
            args.k,
            args.seed,
            args.episodes,
            args.checkpoint_stride,
        )


if __name__ == "__main__":
    main()
