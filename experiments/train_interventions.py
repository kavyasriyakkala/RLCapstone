import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gridworld import ShortcutGridWorld


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
        for ch in [1, 2]:
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


class ShortcutCallback(BaseCallback):
    def __init__(self, train_env, test_env, eval_every=5000, n_eval_episodes=30, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self.test_env = test_env
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.steps_log = []
        self.train_success = []
        self.test_success = []
        self.delta_t = []

    def _on_step(self):
        if self.n_calls % self.eval_every == 0:
            tr = self._evaluate(self.train_env)
            te = self._evaluate(self.test_env)
            gap = tr - te
            self.steps_log.append(self.n_calls)
            self.train_success.append(tr)
            self.test_success.append(te)
            self.delta_t.append(gap)
            if self.verbose:
                print(
                    f"Step {self.n_calls:>7} | "
                    f"Train: {tr:.2f} | Test: {te:.2f} | Gap δ: {gap:.2f}"
                )
        return True

    def _evaluate(self, env):
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            last_info = None
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_info = info
                done = terminated or truncated
            if last_info is not None and last_info.get("reached_goal", False):
                successes += 1
        return successes / self.n_eval_episodes


def build_env(grid_size, num_spurious, train_mode, method):
    env = ShortcutGridWorld(
        grid_size=grid_size,
        num_spurious=num_spurious,
        train_mode=train_mode,
    )

    if train_mode and method == "augmentation":
        env = ColorJitterWrapper(env, jitter_strength=0.2)

    if train_mode and method == "par":
        env = PARRewardWrapper(env, alpha=4.0, reference_reward=0.0)

    env = FlattenWrapper(env)
    return env


def make_train_env(grid_size, num_spurious, method):
    def _init():
        return build_env(grid_size, num_spurious, train_mode=True, method=method)
    return _init


def get_model(method, vec_env, seed):
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

    if method in ["vanilla", "augmentation", "par"]:
        return PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(
                net_arch=[256, 256]
            ),
            **common_kwargs,
        )

    if method == "l2":
        return PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(
                net_arch=[256, 256],
                optimizer_kwargs=dict(weight_decay=1e-4),
            ),
            **common_kwargs,
        )

    raise ValueError(f"Unknown method: {method}")


def plot_results(steps, train_succ, test_succ, delta_t, method, k, seed, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, train_succ, color="green", linewidth=2, label="Train success")
    ax1.plot(steps, test_succ, color="red", linewidth=2, label="Test success")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Success rate")
    ax1.set_title(f"{method}: Train vs Test (k={k}, seed={seed})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(steps, delta_t, color="purple", linewidth=2, label="δ(t)")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("δ(t) = Train - Test")
    ax2.set_title(f"{method}: Shortcut gap (k={k}, seed={seed})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{method}_k{k}_seed{seed}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure to {fig_path}")


def train(method="vanilla", grid_size=10, num_spurious=1, total_steps=500000, eval_every=5000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_eval_env = build_env(grid_size, num_spurious, train_mode=True, method="vanilla")
    test_eval_env = build_env(grid_size, num_spurious, train_mode=False, method="vanilla")
    vec_env = DummyVecEnv([make_train_env(grid_size, num_spurious, method)])

    model = get_model(method, vec_env, seed)

    callback = ShortcutCallback(
        train_env=train_eval_env,
        test_env=test_eval_env,
        eval_every=eval_every,
        n_eval_episodes=30,
        verbose=1,
    )

    print(f"Training method={method} | grid={grid_size} | spurious={num_spurious} | seed={seed}")
    print("-" * 70)

    model.learn(total_timesteps=total_steps, callback=callback)

    out_dir = f"results/interventions/{method}/k{num_spurious}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "final_model"))

    np.savez(
        os.path.join(out_dir, "logs.npz"),
        steps=np.array(callback.steps_log),
        train=np.array(callback.train_success),
        test=np.array(callback.test_success),
        delta=np.array(callback.delta_t),
    )

    plot_results(
        callback.steps_log,
        callback.train_success,
        callback.test_success,
        callback.delta_t,
        method,
        num_spurious,
        seed,
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="vanilla",
                        choices=["vanilla", "l2", "augmentation", "par"])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500000)
    args = parser.parse_args()

    train(
        method=args.method,
        grid_size=10,
        num_spurious=args.k,
        total_steps=args.steps,
        eval_every=5000,
        seed=args.seed,
    )