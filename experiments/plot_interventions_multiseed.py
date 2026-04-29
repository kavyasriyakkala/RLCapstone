import os
import csv
import numpy as np
import matplotlib.pyplot as plt

methods = ["vanilla", "l2", "augmentation", "par"]
ks = [1, 2, 3]
seeds = [7, 42, 123]

base_dir = "results/interventions"
out_dir = "results/figures/interventions_multiseed"
os.makedirs(out_dir, exist_ok=True)

colors = {
    "vanilla": "#4C4C9D",
    "l2": "#2E8B57",
    "augmentation": "#D96C06",
    "par": "#C13C7A",
}

summary_rows = []

for k in ks:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        train_runs = []
        test_runs = []
        delta_runs = []
        steps_ref = None

        for seed in seeds:
            path = f"{base_dir}/{method}/k{k}_seed{seed}/logs.npz"
            logs = np.load(path)

            steps = logs["steps"]
            train = logs["train"]
            test = logs["test"]
            delta = logs["delta"]

            if steps_ref is None:
                steps_ref = steps

            train_runs.append(train)
            test_runs.append(test)
            delta_runs.append(delta)

        train_runs = np.array(train_runs)
        test_runs = np.array(test_runs)
        delta_runs = np.array(delta_runs)

        train_mean = train_runs.mean(axis=0)
        train_std = train_runs.std(axis=0)
        test_mean = test_runs.mean(axis=0)
        test_std = test_runs.std(axis=0)
        delta_mean = delta_runs.mean(axis=0)
        delta_std = delta_runs.std(axis=0)

        ax1.plot(
            steps_ref, test_mean,
            color=colors[method],
            linewidth=2,
            label=method
        )
        ax1.fill_between(
            steps_ref,
            test_mean - test_std,
            test_mean + test_std,
            color=colors[method],
            alpha=0.15
        )

        ax2.plot(
            steps_ref, delta_mean,
            color=colors[method],
            linewidth=2,
            label=method
        )
        ax2.fill_between(
            steps_ref,
            delta_mean - delta_std,
            delta_mean + delta_std,
            color=colors[method],
            alpha=0.15
        )

        summary_rows.append([
            method,
            k,
            float(train_mean[-1]),
            float(train_std[-1]),
            float(test_mean[-1]),
            float(test_std[-1]),
            float(delta_mean[-1]),
            float(delta_std[-1]),
        ])

    ax1.set_title(f"Test Success by Intervention (k={k}, mean ± std over 3 seeds)")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Test success")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title(f"Shortcut Gap by Intervention (k={k}, mean ± std over 3 seeds)")
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("δ(t) = Train - Test")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/interventions_k{k}_mean_std.png", dpi=150)
    plt.close()

# Final-gap summary plot across k values
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(ks))
width = 0.18
offsets = {
    "vanilla": -1.5 * width,
    "l2": -0.5 * width,
    "augmentation": 0.5 * width,
    "par": 1.5 * width,
}

for method in methods:
    means = []
    stds = []
    for k in ks:
        row = [r for r in summary_rows if r[0] == method and r[1] == k][0]
        means.append(row[6])
        stds.append(row[7])

    ax.bar(
        x + offsets[method],
        means,
        width=width,
        yerr=stds,
        capsize=4,
        color=colors[method],
        alpha=0.85,
        label=method
    )

ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_ylabel("Final shortcut gap")
ax.set_title("Final shortcut gap by intervention and k (mean ± std)")
ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/final_gap_summary.png", dpi=150)
plt.close()

# Save csv summary
csv_path = f"{out_dir}/intervention_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method", "k",
        "final_train_mean", "final_train_std",
        "final_test_mean", "final_test_std",
        "final_gap_mean", "final_gap_std"
    ])
    writer.writerows(summary_rows)

print("Saved:")
for k in ks:
    print(f"  {out_dir}/interventions_k{k}_mean_std.png")
print(f"  {out_dir}/final_gap_summary.png")
print(f"  {csv_path}")