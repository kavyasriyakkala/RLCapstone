"""
Plot Figure 4: PPO vs A2C vs DQN shortcut-gap comparison.

Expected inputs:
results/algorithms/{algorithm}/k{k}_seed{seed}/logs.npz

For PPO only, the script also falls back to the original baseline path:
results/checkpoints/sb3_k{k}_seed{seed}/logs.npz
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("ppo", "a2c", "dqn")
COLORS = {
    "ppo": "#4C4C9D",
    "a2c": "#2E8B57",
    "dqn": "#D96C06",
}


def parse_csv_ints(raw):
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_csv_strings(raw):
    return tuple(x.strip().lower() for x in raw.split(",") if x.strip())


def algorithm_log_path(base_dir, algorithm, k, seed):
    primary = os.path.join(base_dir, algorithm, f"k{k}_seed{seed}", "logs.npz")
    if os.path.exists(primary):
        return primary

    if algorithm == "ppo":
        fallback = os.path.join("results", "checkpoints", f"sb3_k{k}_seed{seed}", "logs.npz")
        if os.path.exists(fallback):
            return fallback

    return primary


def load_run(path):
    logs = np.load(path)
    return {
        "steps": logs["steps"].astype(float),
        "train": logs["train"].astype(float),
        "test": logs["test"].astype(float),
        "delta": logs["delta"].astype(float),
    }


def collect_runs(base_dir, algorithms, k_values, seeds):
    runs = {algorithm: [] for algorithm in algorithms}
    by_k = {k: {algorithm: [] for algorithm in algorithms} for k in k_values}
    missing = []

    for k in k_values:
        for seed in seeds:
            for algorithm in algorithms:
                path = algorithm_log_path(base_dir, algorithm, k, seed)
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                run = load_run(path)
                run["k"] = k
                run["seed"] = seed
                runs[algorithm].append(run)
                by_k[k][algorithm].append(run)

    return runs, by_k, missing


def aggregate_runs(runs, metric):
    if not runs:
        return None

    shortest = min(len(run["steps"]) for run in runs)
    if shortest == 0:
        return None

    common_steps = runs[0]["steps"][:shortest]
    values = []
    for run in runs:
        values.append(np.interp(common_steps, run["steps"], run[metric]))

    arr = np.vstack(values)
    return {
        "steps": common_steps,
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "n": arr.shape[0],
    }


def load_baselines(k_values, seeds):
    rows = {"random": [], "oracle": []}
    for k in k_values:
        for seed in seeds:
            path = os.path.join("results", "algorithms", "baselines", f"k{k}_seed{seed}", "baselines.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            for policy in rows:
                rows[policy].append(
                    {
                        "k": k,
                        "seed": seed,
                        "train": float(data[f"{policy}_train"]),
                        "test": float(data[f"{policy}_test"]),
                        "delta": float(data[f"{policy}_delta"]),
                    }
                )
    return rows


def baseline_summary(baselines, policy, metric):
    rows = baselines.get(policy, [])
    if not rows:
        return None
    values = np.array([row[metric] for row in rows], dtype=float)
    return float(values.mean()), float(values.std()), len(values)


def draw_metric_panel(ax, runs, baselines, algorithms, metric, title, ylabel):
    for algorithm in algorithms:
        agg = aggregate_runs(runs[algorithm], metric)
        if agg is None:
            continue
        color = COLORS.get(algorithm)
        label = f"{algorithm.upper()} (n={agg['n']})"
        ax.plot(agg["steps"], agg["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            agg["steps"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=color,
            alpha=0.18,
        )

    if metric in ("test", "delta"):
        for policy, linestyle in (("random", ":"), ("oracle", "--")):
            summary = baseline_summary(baselines, policy, metric)
            if summary is None:
                continue
            mean, std, n = summary
            ax.axhline(mean, color="black", linestyle=linestyle, alpha=0.7, linewidth=1.5)
            ax.text(
                0.99,
                mean,
                f" {policy} {mean:.2f}+/-{std:.2f} (n={n})",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=8,
                backgroundcolor="white",
            )

    if metric == "delta":
        ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)


def plot_aggregate(runs, baselines, algorithms, k_values, seeds, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    draw_metric_panel(
        axes[0],
        runs,
        baselines,
        algorithms,
        metric="test",
        title=f"Test Success by Algorithm (k={list(k_values)}, seeds={list(seeds)})",
        ylabel="Test success",
    )
    axes[0].set_ylim(-0.05, 1.05)

    draw_metric_panel(
        axes[1],
        runs,
        baselines,
        algorithms,
        metric="delta",
        title="Shortcut Gap by Algorithm",
        ylabel="delta(t) = Train - Test",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_by_k(by_k, baselines, algorithms, k_values, seeds, out_path):
    fig, axes = plt.subplots(len(k_values), 2, figsize=(14, 4 * len(k_values)), squeeze=False)

    for row, k in enumerate(k_values):
        k_runs = by_k[k]
        k_baselines = {
            policy: [item for item in values if item["k"] == k]
            for policy, values in baselines.items()
        }
        draw_metric_panel(
            axes[row][0],
            k_runs,
            k_baselines,
            algorithms,
            metric="test",
            title=f"k={k}: Test Success (seeds={list(seeds)})",
            ylabel="Test success",
        )
        axes[row][0].set_ylim(-0.05, 1.05)

        draw_metric_panel(
            axes[row][1],
            k_runs,
            k_baselines,
            algorithms,
            metric="delta",
            title=f"k={k}: Shortcut Gap",
            ylabel="delta(t) = Train - Test",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def print_missing(missing):
    if not missing:
        return
    print("\nMissing logs:")
    for path in missing:
        print(f"  {path}")
    print("\nRun experiments with experiments/train_algorithms.py before plotting missing curves.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.join("results", "algorithms"))
    parser.add_argument("--algorithms", type=str, default="ppo,a2c,dqn")
    parser.add_argument("--ks", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=str, default="42,123,7")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("results", "figures", "algorithm_comparison_aggregate.png"),
    )
    parser.add_argument(
        "--by-k-out",
        type=str,
        default=os.path.join("results", "figures", "algorithm_comparison_by_k.png"),
    )
    args = parser.parse_args()

    algorithms = parse_csv_strings(args.algorithms)
    k_values = parse_csv_ints(args.ks)
    seeds = parse_csv_ints(args.seeds)

    runs, by_k, missing = collect_runs(args.base_dir, algorithms, k_values, seeds)
    baselines = load_baselines(k_values, seeds)
    print_missing(missing)

    plot_aggregate(runs, baselines, algorithms, k_values, seeds, args.out)
    plot_by_k(by_k, baselines, algorithms, k_values, seeds, args.by_k_out)


if __name__ == "__main__":
    main()
