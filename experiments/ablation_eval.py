"""
Evaluate a trained policy under observation ablations (zero out goal or spurious channel).

Obs channels: 0=agent, 1=goal, 2=spurious.
- no_spurious: if train success drops a lot but test does not, the policy relied on
  the spurious map especially where it overlaps the goal (train).
- no_goal: control; success should collapse when the true goal is hidden.
"""
import sys
import os
import re
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gridworld import ShortcutGridWorld

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FlattenWrapper(gym.ObservationWrapper):
    """Match training: (3, H, W) -> flat vector."""

    def __init__(self, env):
        super().__init__(env)
        orig = env.observation_space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(np.prod(orig.shape),), dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()


class ChannelMaskWrapper(gym.ObservationWrapper):
    """Zero out selected channel planes before flatten."""

    def __init__(self, env, zero_channels):
        super().__init__(env)
        self._zero = frozenset(int(c) for c in zero_channels)

    def observation(self, obs):
        o = np.array(obs, copy=True, dtype=np.float32)
        for c in self._zero:
            o[c, :, :] = 0.0
        return o


def make_wrapped_env(grid_size, num_spurious, train_mode, zero_channels=()):
    def _thunk():
        base = ShortcutGridWorld(
            grid_size=grid_size,
            num_spurious=num_spurious,
            train_mode=train_mode,
        )
        if zero_channels:
            base = ChannelMaskWrapper(base, zero_channels)
        return FlattenWrapper(base)

    return _thunk


def success_rate(model, env_fn, n_episodes, reset_seed=0):
    """Fraction of episodes ending in success (same convention as training callback)."""
    env = env_fn()
    rng = np.random.default_rng(reset_seed)
    ok = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            ok += 1
    return ok / n_episodes


CONDITIONS = [
    ("full", ()),
    ("no_goal (ch1)", (1,)),
    ("no_spurious (ch2)", (2,)),
]


def run_ablation_table(
    model_path,
    grid_size=10,
    k=1,
    n_episodes=100,
    base_seed=0,
    conditions=None,
    quiet=False,
):
    """Load model and print success under full / no-goal / no-spurious for train and test envs."""
    if conditions is None:
        conditions = CONDITIONS
    vec = DummyVecEnv(
        [make_wrapped_env(grid_size, k, train_mode=True, zero_channels=())]
    )
    model = PPO.load(model_path, env=vec)

    train_modes = [("train", True), ("test", False)]

    if not quiet:
        print(f"Model: {model_path} | k={k} | n_episodes={n_episodes}")
        print("-" * 72)
        print(f"{'condition':<22} | {'train':>8} | {'test':>8} |")
        print("-" * 72)

    results = {name: {"train": None, "test": None} for name, _ in conditions}

    for cond_name, zc in conditions:
        row = f"{cond_name:<22} |"
        for split_name, tm in train_modes:
            rate = success_rate(
                model,
                make_wrapped_env(grid_size, k, train_mode=tm, zero_channels=zc),
                n_episodes,
                reset_seed=base_seed,
            )
            results[cond_name][split_name] = rate
            row += f" {rate:8.3f} |"
        if not quiet:
            print(row)
    if not quiet:
        print("-" * 72)
    return results, conditions, train_modes


def plot_ablation_bars(
    results,
    conditions,
    k,
    seed_label,
    out_path,
):
    names = [c[0] for c in conditions]
    train_vals = [results[n]["train"] for n in names]
    test_vals = [results[n]["test"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, train_vals, w, label="Train env", color="#2d8f47")
    ax.bar(x + w / 2, test_vals, w, label="Test env", color="#c44e52")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Observation condition")
    ax.set_title(
        f"Channel ablation (k={k}, seed={seed_label})\n"
        "Train: losing ch2 often hurts if the policy used it; test: can rise if ch2 misled"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved: {out_path}")


def plot_ablation_bars_aggregate(
    conditions,
    k,
    train_mean, train_std, test_mean, test_std, out_path
):
    """Grouped bars with yerr=std (over training seeds)."""
    names = [c[0] for c in conditions]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x - w / 2, train_mean, w, yerr=train_std, capsize=4,
        label="Train env", color="#2d8f47", ecolor="black", alpha=0.9,
    )
    ax.bar(
        x + w / 2, test_mean, w, yerr=test_std, capsize=4,
        label="Test env", color="#c44e52", ecolor="black", alpha=0.9,
    )
    ax.set_ylabel("Success rate (mean ± SD over training seeds)")
    ax.set_xlabel("Observation condition")
    ax.set_title(
        f"Channel ablation aggregate (k={k})\n"
        "Train: losing ch2 often hurts if the policy used it; test: can rise if ch2 misled"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved: {out_path}")


def print_aggregate_table(conditions, seed_runs, all_results_tuples):
    """all_results_tuples: list of (train_seed, results_dict)."""
    n = len(all_results_tuples)
    if n == 0:
        return
    print("\n" + "=" * 72)
    print(
        f"Aggregate over training seeds {list(seed_runs)} "
        f"(n={n}) — mean ± std dev"
    )
    print("=" * 72)
    print(f"{'condition':<22} | {'train':>20} | {'test':>20} |")
    print("-" * 72)
    for cond_name, _ in conditions:
        tr = [r[1][cond_name]["train"] for r in all_results_tuples]
        te = [r[1][cond_name]["test"] for r in all_results_tuples]
        m_tr, s_tr = float(np.mean(tr)), float(np.std(tr, ddof=1)) if n > 1 else 0.0
        m_te, s_te = float(np.mean(te)), float(np.std(te, ddof=1)) if n > 1 else 0.0
        if n == 1:
            s_tr = s_te = 0.0
        tr_s = f"{m_tr:.3f} ± {s_tr:.3f}"
        te_s = f"{m_te:.3f} ± {s_te:.3f}"
        print(f"{cond_name:<22} | {tr_s:>20} | {te_s:>20} |")
    print("=" * 72 + "\n")


def resolve_model_path(path_arg):
    """Path relative to project root, or absolute; append .zip if needed."""
    if not os.path.isabs(path_arg):
        path = os.path.join(PROJECT_ROOT, path_arg)
    else:
        path = path_arg
    if not path.endswith(".zip") and os.path.exists(path + ".zip"):
        path = path + ".zip"
    return path


def main():
    p = argparse.ArgumentParser(description="PPO channel ablation eval (goal vs spurious).")
    p.add_argument(
        "--model",
        type=str,
        default="results/checkpoints/sb3_k1_seed42/final_model",
        help="Path under project root, or abs path; .zip added if present (ignored if --seed-runs)",
    )
    p.add_argument(
        "--seed-runs",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="Training run seeds: loads results/checkpoints/sb3_k{K}_seed{S}/final_model for each. "
        "Example: --seed-runs 42 123 7",
    )
    p.add_argument(
        "--quiet-per-seed",
        action="store_true",
        help="With --seed-runs, do not print per-seed tables (only aggregate).",
    )
    p.add_argument("--k", type=int, default=1, help="num_spurious (k)")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="Base seed for eval episode resets")
    p.add_argument(
        "--no-plot", action="store_true", help="Do not save figure(s)"
    )
    p.add_argument(
        "--plot-per-seed",
        action="store_true",
        help="With --seed-runs, also save one bar chart per training seed (default: aggregate only).",
    )
    args = p.parse_args()

    conditions = CONDITIONS

    if args.seed_runs is not None:
        all_tuples = []
        for s in args.seed_runs:
            rel = f"results/checkpoints/sb3_k{args.k}_seed{s}/final_model"
            path = resolve_model_path(rel)
            if not os.path.isfile(path):
                print(f"Error: model not found at {path}", file=sys.stderr)
                sys.exit(1)
            if not args.quiet_per_seed:
                print(f"\n=== Training seed {s} (checkpoint: {rel}) ===\n")
            res, conds, _ = run_ablation_table(
                path,
                grid_size=args.grid_size,
                k=args.k,
                n_episodes=args.n_episodes,
                base_seed=args.seed,
                conditions=conditions,
                quiet=args.quiet_per_seed,
            )
            if args.plot_per_seed and not args.no_plot:
                out_s = os.path.join(
                    PROJECT_ROOT,
                    "results",
                    "figures",
                    f"ablation_k{args.k}_seed{s}.png",
                )
                plot_ablation_bars(res, conds, args.k, str(s), out_s)
            all_tuples.append((s, res))

        print_aggregate_table(conditions, args.seed_runs, all_tuples)

        if not args.no_plot and len(all_tuples) > 0:
            train_mean = []
            train_std = []
            test_mean = []
            test_std = []
            n_seeds = len(all_tuples)
            for cond_name, _ in conditions:
                tr = [r[1][cond_name]["train"] for r in all_tuples]
                te = [r[1][cond_name]["test"] for r in all_tuples]
                train_mean.append(float(np.mean(tr)))
                test_mean.append(float(np.mean(te)))
                if n_seeds > 1:
                    train_std.append(float(np.std(tr, ddof=1)))
                    test_std.append(float(np.std(te, ddof=1)))
                else:
                    train_std.append(0.0)
                    test_std.append(0.0)
            out_agg = os.path.join(
                PROJECT_ROOT,
                "results",
                "figures",
                f"ablation_k{args.k}_aggregate.png",
            )
            plot_ablation_bars_aggregate(
                conditions,
                args.k,
                np.array(train_mean),
                np.array(train_std),
                np.array(test_mean),
                np.array(test_std),
                out_agg,
            )
    else:
        path = resolve_model_path(args.model)
        if not os.path.isfile(path):
            print(f"Error: model not found at {path}")
            sys.exit(1)

        m = re.search(r"seed(\d+)", path)
        seed_tag = m.group(1) if m else "unknown"

        results, conditions, _ = run_ablation_table(
            path,
            grid_size=args.grid_size,
            k=args.k,
            n_episodes=args.n_episodes,
            base_seed=args.seed,
            conditions=conditions,
            quiet=False,
        )
        if not args.no_plot:
            out = os.path.join(
                PROJECT_ROOT,
                "results",
                "figures",
                f"ablation_k{args.k}_seed{seed_tag}.png",
            )
            plot_ablation_bars(results, conditions, args.k, str(seed_tag), out)


if __name__ == "__main__":
    main()
