import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

steps = np.load("results/probe_steps.npy", allow_pickle=True).tolist()
goal_r2s = np.load("results/probe_goal_r2.npy", allow_pickle=True).tolist()
spurious_r2s = np.load("results/probe_spurious_r2.npy", allow_pickle=True).tolist()

# Flatten if nested
if isinstance(steps[0], list):
    steps = [s for sub in steps for s in sub]
    goal_r2s = [s for sub in goal_r2s for s in sub]
    spurious_r2s = [s for sub in spurious_r2s for s in sub]

# Sort by step
combined = sorted(zip(steps, goal_r2s, spurious_r2s))
steps, goal_r2s, spurious_r2s = zip(*combined)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(steps, goal_r2s, color='green', linewidth=2,
         label='Goal position probe R²')
ax1.plot(steps, spurious_r2s, color='red', linewidth=2,
         linestyle='--', label='Spurious feature probe R²')
ax1.set_xlabel('Training steps')
ax1.set_ylabel('Probe R²')
ax1.set_title('Linear probe accuracy over training (seed 42, k=1)')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.7, 1.05)

diff = np.array(goal_r2s) - np.array(spurious_r2s)
ax2.plot(steps, diff, color='purple', linewidth=2,
         label='Goal R² − Spurious R²')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.fill_between(steps, diff, 0,
                 where=[d > 0 for d in diff],
                 alpha=0.3, color='green', label='Goal > Spurious')
ax2.fill_between(steps, diff, 0,
                 where=[d < 0 for d in diff],
                 alpha=0.3, color='red', label='Spurious > Goal')
ax2.set_xlabel('Training steps')
ax2.set_ylabel('R² difference')
ax2.set_title('Difference: Goal probe R² minus Spurious probe R²')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/linear_probes_k1_seed42.png", dpi=150)
plt.show()
plt.close()
print("Done.")