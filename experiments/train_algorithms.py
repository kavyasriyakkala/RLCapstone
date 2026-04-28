"""
Train PPO, A2C, and DQN on the shortcut gridworld with shared evaluation logs.

Outputs use the same schema as the existing intervention experiments:
results/algorithms/{algorithm}/k{k}_seed{seed}/logs.npz
with arrays: steps, train, test, delta.
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gridworld import ShortcutGridWorld


ALGORITHMS = ("ppo", "a2c", "dqn")
DEFAULT_SEEDS = (42, 123, 7)
DEFAULT_K_VALUES = (1, 2, 3)


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
            train_rate = evaluate_model(self.model, self.train_env, self.n_eval_episodes)
            test_rate = evaluate_model(self.model, self.test_env, self.n_eval_episodes)
            gap = train_rate - test_rate
            self.steps_log.append(self.n_calls)
            self.train_success.append(train_rate)
            self.test_success.append(test_rate)
            self.delta_t.append(gap)
            if self.verbose:
                print(
                    f"Step {self.n_calls:>7} | "
                    f"Train: {train_rate:.2f} | Test: {test_rate:.2f} | Gap delta: {gap:.2f}"
                )
        return True


def build_env(grid_size, num_spurious, train_mode):
    env = ShortcutGridWorld(
        grid_size=grid_size,
        num_spurious=num_spurious,
        train_mode=train_mode,
    )
    return FlattenWrapper(env)


def make_train_env(grid_size, num_spurious):
    def _init():
        return build_env(grid_size, num_spurious, train_mode=True)

    return _init


def evaluate_model(model, env, n_episodes):
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
        if last_info.get("reached_goal", False):
            successes += 1
    return successes / n_episodes


def get_model(algorithm, vec_env, seed):
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return PPO(
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
            seed=seed,
        )

    if algorithm == "a2c":
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

    if algorithm == "dqn":
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

    raise ValueError(f"Unknown algorithm: {algorithm}")


def plot_results(steps, train_succ, test_succ, delta_t, algorithm, k, seed, out_dir):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, train_succ, color="green", linewidth=2, label="Train success")
    ax1.plot(steps, test_succ, color="red", linewidth=2, label="Test success")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Success rate")
    ax1.set_title(f"{algorithm.upper()}: Train vs Test (k={k}, seed={seed})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(steps, delta_t, color="purple", linewidth=2, label="delta(t)")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("delta(t) = Train - Test")
    ax2.set_title(f"{algorithm.upper()}: Shortcut gap (k={k}, seed={seed})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{algorithm}_k{k}_seed{seed}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure to {fig_path}")


def train_algorithm(
    algorithm="ppo",
    grid_size=10,
    num_spurious=1,
    total_steps=500_000,
    eval_every=5000,
    seed=42,
):
    algorithm = algorithm.lower()
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_eval_env = build_env(grid_size, num_spurious, train_mode=True)
    test_eval_env = build_env(grid_size, num_spurious, train_mode=False)
    vec_env = DummyVecEnv([make_train_env(grid_size, num_spurious)])

    model = get_model(algorithm, vec_env, seed)
    callback = ShortcutCallback(
        train_env=train_eval_env,
        test_env=test_eval_env,
        eval_every=eval_every,
        n_eval_episodes=30,
        verbose=1,
    )

    print(f"Training algorithm={algorithm} | grid={grid_size} | spurious={num_spurious} | seed={seed}")
    print("-" * 78)
    model.learn(total_timesteps=total_steps, callback=callback)

    out_dir = f"results/algorithms/{algorithm}/k{num_spurious}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "final_model"))

    np.savez(
        os.path.join(out_dir, "logs.npz"),
        steps=np.array(callback.steps_log),
        train=np.array(callback.train_success),
        test=np.array(callback.test_success),
        delta=np.array(callback.delta_t),
    )
    print(f"Saved logs to {os.path.join(out_dir, 'logs.npz')}")

    plot_results(
        callback.steps_log,
        callback.train_success,
        callback.test_success,
        callback.delta_t,
        algorithm,
        num_spurious,
        seed,
        out_dir,
    )
    return model, callback


def oracle_action(env):
    agent_row, agent_col = env.unwrapped.agent_pos
    goal_row, goal_col = env.unwrapped.goal_pos
    if agent_row < goal_row:
        return 1
    if agent_row > goal_row:
        return 0
    if agent_col < goal_col:
        return 3
    if agent_col > goal_col:
        return 2
    return 0


def evaluate_baseline(policy, grid_size, num_spurious, train_mode, n_episodes, seed):
    np.random.seed(seed)
    env = ShortcutGridWorld(
        grid_size=grid_size,
        num_spurious=num_spurious,
        train_mode=train_mode,
    )
    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(n_episodes):
        env.reset()
        done = False
        last_info = {}
        while not done:
            if policy == "random":
                action = int(rng.integers(0, env.action_space.n))
            elif policy == "oracle":
                action = oracle_action(env)
            else:
                raise ValueError(f"Unknown baseline policy: {policy}")
            _, _, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
        if last_info.get("reached_goal", False):
            successes += 1

    return successes / n_episodes


def save_baselines(grid_size=10, num_spurious=1, seed=42, n_episodes=100):
    out_dir = f"results/algorithms/baselines/k{num_spurious}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for policy in ("random", "oracle"):
        train_rate = evaluate_baseline(
            policy, grid_size, num_spurious, train_mode=True, n_episodes=n_episodes, seed=seed
        )
        test_rate = evaluate_baseline(
            policy, grid_size, num_spurious, train_mode=False, n_episodes=n_episodes, seed=seed + 1
        )
        results[policy] = {
            "train": train_rate,
            "test": test_rate,
            "delta": train_rate - test_rate,
        }

    np.savez(
        os.path.join(out_dir, "baselines.npz"),
        random_train=results["random"]["train"],
        random_test=results["random"]["test"],
        random_delta=results["random"]["delta"],
        oracle_train=results["oracle"]["train"],
        oracle_test=results["oracle"]["test"],
        oracle_delta=results["oracle"]["delta"],
    )
    print(f"Saved baselines to {os.path.join(out_dir, 'baselines.npz')}")
    for policy, values in results.items():
        print(
            f"{policy:>6} | train={values['train']:.3f} | "
            f"test={values['test']:.3f} | delta={values['delta']:.3f}"
        )
    return results


def parse_csv_ints(raw):
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ppo", choices=ALGORITHMS)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Run PPO/A2C/DQN for k=1,2,3 and seeds 42,123,7.")
    parser.add_argument("--algorithms", type=str, default="ppo,a2c,dqn")
    parser.add_argument("--ks", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=str, default="42,123,7")
    parser.add_argument("--baselines", action="store_true", help="Save random and oracle baselines.")
    parser.add_argument("--baseline-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.baselines and not args.all:
        save_baselines(
            grid_size=args.grid_size,
            num_spurious=args.k,
            seed=args.seed,
            n_episodes=args.baseline_episodes,
        )
        return

    if args.all:
        algorithms = tuple(a.strip().lower() for a in args.algorithms.split(",") if a.strip())
        k_values = parse_csv_ints(args.ks)
        seeds = parse_csv_ints(args.seeds)
        for k in k_values:
            for seed in seeds:
                save_baselines(
                    grid_size=args.grid_size,
                    num_spurious=k,
                    seed=seed,
                    n_episodes=args.baseline_episodes,
                )
                for algorithm in algorithms:
                    train_algorithm(
                        algorithm=algorithm,
                        grid_size=args.grid_size,
                        num_spurious=k,
                        total_steps=args.steps,
                        eval_every=args.eval_every,
                        seed=seed,
                    )
        return

    train_algorithm(
        algorithm=args.algorithm,
        grid_size=args.grid_size,
        num_spurious=args.k,
        total_steps=args.steps,
        eval_every=args.eval_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
