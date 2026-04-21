import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            low=0, high=1,
            shape=(np.prod(orig.shape),),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()


class ShortcutCallback(BaseCallback):
    def __init__(self, train_env, test_env, eval_every=5000,
                 n_eval_episodes=30, verbose=1):
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
                print(f"Step {self.n_calls:>7} | "
                      f"Train: {tr:.2f} | Test: {te:.2f} | Gap δ: {gap:.2f}")
        return True

    def _evaluate(self, env):
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if reward > 0:
                successes += 1
        return successes / self.n_eval_episodes


def make_env(grid_size, num_spurious, train_mode):
    def _init():
        env = ShortcutGridWorld(
            grid_size=grid_size,
            num_spurious=num_spurious,
            train_mode=train_mode
        )
        return FlattenWrapper(env)
    return _init


def train(grid_size=10, num_spurious=1, total_steps=500000,
          eval_every=5000, seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = FlattenWrapper(ShortcutGridWorld(
        grid_size=grid_size, num_spurious=num_spurious, train_mode=True))
    test_env = FlattenWrapper(ShortcutGridWorld(
        grid_size=grid_size, num_spurious=num_spurious, train_mode=False))

    vec_env = DummyVecEnv([make_env(grid_size, num_spurious, True)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=seed
    )

    callback = ShortcutCallback(
        train_env=train_env,
        test_env=test_env,
        eval_every=eval_every,
        n_eval_episodes=30,
        verbose=1
    )

    print(f"Training | grid={grid_size} | spurious={num_spurious} | seed={seed}")
    print("-" * 60)

    model.learn(total_timesteps=total_steps, callback=callback)

    checkpoint_dir = f"results/checkpoints/sb3_k{num_spurious}_seed{seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(f"{checkpoint_dir}/final_model")
    print(f"Model saved to {checkpoint_dir}/final_model")

    plot_results(callback.steps_log, callback.train_success,
                 callback.test_success, callback.delta_t,
                 num_spurious, seed)

    return model, callback


def plot_results(steps, train_succ, test_succ, delta_t, k, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, train_succ, label='Train success',
             color='green', linewidth=2)
    ax1.plot(steps, test_succ, label='Test success',
             color='red', linewidth=2)
    ax1.set_xlabel('Training steps')
    ax1.set_ylabel('Success rate')
    ax1.set_title(f'Train vs Test Performance (k={k}, seed={seed})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(steps, delta_t, label='δ(t) shortcut gap',
             color='purple', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(steps, delta_t, 0,
                     where=[d > 0 for d in delta_t],
                     alpha=0.2, color='purple', label='Shortcut region')
    ax2.set_xlabel('Training steps')
    ax2.set_ylabel('δ(t) = Train - Test')
    ax2.set_title(f'Shortcut Gap δ(t) (k={k}, seed={seed})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    path = f'results/figures/sb3_training_k{k}_seed{seed}.png'
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    model, callback = train(
        grid_size=10,
        num_spurious=1,
        total_steps=500000,
        eval_every=5000,
        seed=42
    )