import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = "results/interventions"
methods = ["vanilla", "l2", "augmentation", "par"]
k = 1
seed = 42

data = {}

for method in methods:
    path = f"{base_dir}/{method}/k{k}_seed{seed}/logs.npz"
    logs = np.load(path)
    data[method] = {
        "steps": logs["steps"],
        "train": logs["train"],
        "test": logs["test"],
        "delta": logs["delta"],
    }

colors = {
    "vanilla": "#4C4C9D",
    "l2": "#2E8B57",
    "augmentation": "#D96C06",
    "par": "#C13C7A",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1, ax2 = axes

for method in methods:
    ax1.plot(
        data[method]["steps"],
        data[method]["test"],
        linewidth=2,
        color=colors[method],
        label=method,
    )

ax1.set_title("Test Success by Intervention (k=1, seed=42)")
ax1.set_xlabel("Training steps")
ax1.set_ylabel("Test success")
ax1.grid(True, alpha=0.3)
ax1.legend()

for method in methods:
    ax2.plot(
        data[method]["steps"],
        data[method]["delta"],
        linewidth=2,
        color=colors[method],
        label=method,
    )

ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
ax2.set_title("Shortcut Gap by Intervention (k=1, seed=42)")
ax2.set_xlabel("Training steps")
ax2.set_ylabel("δ(t) = Train - Test")
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/intervention_comparison_k1_seed42.png", dpi=150)
plt.show()
plt.close()

print("Saved results/figures/intervention_comparison_k1_seed42.png")