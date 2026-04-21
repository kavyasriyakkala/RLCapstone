import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
from gymnasium import spaces
import os


class ShortcutGridWorld(gym.Env):
    def __init__(self, grid_size=10, num_spurious=1, train_mode=True,
                 correlation_strength=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_spurious = num_spurious
        self.train_mode = train_mode
        self.correlation_strength = correlation_strength

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, grid_size, grid_size),
            dtype=np.float32
        )

        self.agent_pos = None
        self.goal_pos = None
        self.spurious_positions = []
        self.steps = 0
        self.max_steps = grid_size * grid_size * 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        self.agent_pos = [
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ]

        self.goal_pos = [
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ]
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ]

        self.spurious_positions = []
        for _ in range(self.num_spurious):
            if self.train_mode:
                # Spurious always on top of goal during training
                # Agent cannot distinguish channels 1 and 2
                spurious = list(self.goal_pos)
            else:
                # Test: spurious placed randomly away from goal
                spurious = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ]
                while spurious == self.goal_pos or spurious == self.agent_pos:
                    spurious = [
                        np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size)
                    ]
            self.spurious_positions.append(spurious)

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        new_pos[0] = max(0, min(self.grid_size - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.grid_size - 1, new_pos[1]))
        self.agent_pos = new_pos

        reached_goal = (self.agent_pos == self.goal_pos)
        near_spurious = any(
            self.agent_pos == s for s in self.spurious_positions)

        if reached_goal:
            reward = 1.0
        else:
            reward = -0.01

        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            'reached_goal': reached_goal,
            'near_spurious': near_spurious,
            'steps': self.steps
        }

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[1, self.goal_pos[0], self.goal_pos[1]] = 1.0
        for s in self.spurious_positions:
            obs[2, s[0], s[1]] = 1.0
        return obs

    def render(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        goal_rect = patches.Rectangle(
            (self.goal_pos[1], self.grid_size - self.goal_pos[0] - 1),
            1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.7)
        ax.add_patch(goal_rect)

        for s in self.spurious_positions:
            spur_rect = patches.Rectangle(
                (s[1], self.grid_size - s[0] - 1),
                1, 1, linewidth=1, edgecolor='black', facecolor='red', alpha=0.5)
            ax.add_patch(spur_rect)

        agent_rect = patches.Rectangle(
            (self.agent_pos[1], self.grid_size - self.agent_pos[0] - 1),
            1, 1, linewidth=1, edgecolor='black', facecolor='blue', alpha=0.9)
        ax.add_patch(agent_rect)

        mode = "TRAIN" if self.train_mode else "TEST"
        ax.set_title(f'ShortcutGridWorld [{mode}] — Step {self.steps}')
        plt.tight_layout()
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(f'results/figures/gridworld_{mode.lower()}.png', dpi=100)
        plt.show()
        plt.close()

    def get_info(self):
        return {
            'grid_size': self.grid_size,
            'num_spurious': self.num_spurious,
            'train_mode': self.train_mode,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'spurious_positions': self.spurious_positions
        }