import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gridworld import ShortcutGridWorld
import gymnasium as gym
from gymnasium import spaces

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import glob
import matplotlib.pyplot as plt


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


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



def extract_activations(model, env, n_episodes=200):
    """
    Run the agent and collect internal activations, goal positions, spurious positions.
    """
    activations = []
    goal_positions = []
    spurious_positions = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                # Extract features using SB3's built-in feature extractor
                features = model.policy.extract_features(
                    obs_tensor,
                    model.policy.features_extractor
                )
                # Pass through first layer of MLP to get activations
                act = model.policy.mlp_extractor.policy_net[0](features)
                act = torch.relu(act)
                act = act.squeeze(0).numpy()

            activations.append(act)
            goal_positions.append(list(env.env.goal_pos))
            spurious_positions.append(list(env.env.spurious_positions[0]))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return (
        np.array(activations),
        np.array(goal_positions),
        np.array(spurious_positions)
    )

def train_probe(activations, labels):
    """Train a linear probe using Ridge regression — solves analytically."""
    probe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    probe.fit(activations, labels)
    
    pred = probe.predict(activations)
    ss_res = np.sum((labels - pred) ** 2)
    ss_tot = np.sum((labels - labels.mean(axis=0)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    return max(0.0, r2)


def run_probe_on_checkpoint(checkpoint_path, step_number):
    print(f"\nLoading checkpoint: step {step_number}")

    env_fn = lambda: FlattenWrapper(
        ShortcutGridWorld(grid_size=10, num_spurious=1, train_mode=True)
    )
    vec_env = DummyVecEnv([env_fn])
    model = PPO.load(checkpoint_path, env=vec_env)

    # Use TEST environment — here goal and spurious are separated
    # This lets us ask: does the agent track goal or spurious?
    raw_env = FlattenWrapper(
        ShortcutGridWorld(grid_size=10, num_spurious=1, train_mode=False)
    )

    print("Extracting activations from test environment...")
    activations, goal_pos, spurious_pos = extract_activations(model, raw_env)

    print(f"Activation shape: {activations.shape}")
    print(f"Activation stats: mean={activations.mean():.3f}, std={activations.std():.3f}")
    print(f"Goal pos std: {goal_pos.std(axis=0)}")
    print(f"Spurious pos std: {spurious_pos.std(axis=0)}")
    print(f"Are they identical? {np.allclose(goal_pos, spurious_pos)}")

    print("Training goal position probe...")
    goal_r2 = train_probe(activations, goal_pos)

    print("Training spurious feature probe...")
    spurious_r2 = train_probe(activations, spurious_pos)

    print(f"\nResults for step {step_number}:")
    print(f"  Goal position probe R²:     {goal_r2:.3f}")
    print(f"  Spurious position probe R²: {spurious_r2:.3f}")

    if spurious_r2 > goal_r2:
        print(f"  → Agent is tracking the SHORTCUT (red) more than the goal")
    else:
        print(f"  → Agent is tracking the GOAL more than the shortcut")

    return goal_r2, spurious_r2


if __name__ == "__main__":

    checkpoint_dir = "results/checkpoints/sb3_probes_k1_seed42"
    checkpoint_files = sorted(glob.glob(f"{checkpoint_dir}/model_*_steps.zip"))

    if not checkpoint_files:
        print("No checkpoints found. Make sure training with CheckpointCallback finished.")
    else:
        print(f"Found {len(checkpoint_files)} checkpoints")

        steps = []
        goal_r2s = []
        spurious_r2s = []

        for path in checkpoint_files:
            step = int(path.split("_steps")[0].split("model_")[-1])
            g, s = run_probe_on_checkpoint(path, step)
            steps.append(step)
            goal_r2s.append(g)
            spurious_r2s.append(s)

        # Save raw results
        os.makedirs("results", exist_ok=True)
        np.save("results/probe_steps.npy", steps)
        np.save("results/probe_goal_r2.npy", goal_r2s)
        np.save("results/probe_spurious_r2.npy", spurious_r2s)

        # Plot
        # Sort by step to ensure clean lines
        combined = sorted(zip(steps, goal_r2s, spurious_r2s))
        steps, goal_r2s, spurious_r2s = zip(*combined)

        # Save raw results
        np.save("results/probe_steps.npy", steps)
        np.save("results/probe_goal_r2.npy", goal_r2s)
        np.save("results/probe_spurious_r2.npy", spurious_r2s)

        # Plot
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
        print("\nProbe analysis complete. Figure saved.")