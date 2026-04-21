import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces

class ShortcutGridWorld(gym.Env):
    def __init__(self, grid_size=10, num_spurious=1, train_mode=True, 
                 correlation_strength=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_spurious = num_spurious
        self.train_mode = train_mode
        self.correlation_strength = correlation_strength
        
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, grid_size, grid_size),
            dtype=np.float32
        )
        
        # Colors: 0=empty(white), 1=agent(blue), 2=goal(green), 3=spurious(red)
        self.colors = {
            'empty': np.array([1.0, 1.0, 1.0]),
            'agent': np.array([0.0, 0.0, 1.0]),
            'goal':  np.array([0.0, 1.0, 0.0]),
            'spurious': np.array([1.0, 0.0, 0.0])
        }
        
        self.agent_pos = None
        self.goal_pos = None
        self.spurious_positions = []
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Place agent randomly
        self.agent_pos = [
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ]
        
        # Place goal
        if self.train_mode:
            # Training: goal always in top-right quadrant
            self.goal_pos = [
                np.random.randint(0, self.grid_size // 2),
                np.random.randint(self.grid_size // 2, self.grid_size)
            ]
        else:
            # Test: goal anywhere
            self.goal_pos = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ]
        
        # Make sure goal and agent aren't on same cell
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ]
        
        # Place spurious features (red cells)
        self.spurious_positions = []
        for _ in range(self.num_spurious):
            use_correlation = np.random.random() < self.correlation_strength
            if self.train_mode and use_correlation:
                # Training: spurious feature near goal (the shortcut)
                spurious = [
                    max(0, min(self.grid_size-1, self.goal_pos[0] + np.random.randint(-1, 2))),
                    max(0, min(self.grid_size-1, self.goal_pos[1] + np.random.randint(-1, 2)))
                ]
            else:
                # Test: spurious feature placed randomly
                spurious = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ]
            
            while spurious == self.agent_pos or spurious == self.goal_pos:
                spurious = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ]
            self.spurious_positions.append(spurious)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.steps += 1
        
        # Move agent: 0=up, 1=down, 2=left, 3=right
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        # Stay in bounds
        new_pos[0] = max(0, min(self.grid_size - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.grid_size - 1, new_pos[1]))
        self.agent_pos = new_pos
        
        # Check if reached goal
        reached_goal = (self.agent_pos == self.goal_pos)
        
        # Check if near spurious feature
        near_spurious = any(self.agent_pos == s for s in self.spurious_positions)
        
        # Reward
        if reached_goal:
            reward = 1.0
        else:
            reward = -0.01  # small penalty per step to encourage efficiency
            
        terminated = reached_goal
        truncated = self.steps >= self.max_steps
        
        info = {
            'reached_goal': reached_goal,
            'near_spurious': near_spurious,
            'steps': self.steps
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: agent position
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Channel 1: goal position
        obs[1, self.goal_pos[0], self.goal_pos[1]] = 1.0
        
        # Channel 2: spurious features
        for s in self.spurious_positions:
            obs[2, s[0], s[1]] = 1.0
            
        return obs
    
    def render(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw goal (green)
        goal_rect = patches.Rectangle(
            (self.goal_pos[1], self.grid_size - self.goal_pos[0] - 1),
            1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.7
        )
        ax.add_patch(goal_rect)
        
        # Draw spurious features (red)
        for s in self.spurious_positions:
            spur_rect = patches.Rectangle(
                (s[1], self.grid_size - s[0] - 1),
                1, 1, linewidth=1, edgecolor='black', facecolor='red', alpha=0.7
            )
            ax.add_patch(spur_rect)
        
        # Draw agent (blue)
        agent_rect = patches.Rectangle(
            (self.agent_pos[1], self.grid_size - self.agent_pos[0] - 1),
            1, 1, linewidth=1, edgecolor='black', facecolor='blue', alpha=0.9
        )
        ax.add_patch(agent_rect)
        
        mode = "TRAIN" if self.train_mode else "TEST"
        ax.set_title(f'ShortcutGridWorld [{mode}] — Step {self.steps}')
        plt.tight_layout()
        plt.savefig(f'results/figures/gridworld_{mode.lower()}.png', dpi=100)
        plt.show()
        plt.close()
        
    def get_info(self):
        return {
            'grid_size': self.grid_size,
            'num_spurious': self.num_spurious,
            'train_mode': self.train_mode,
            'correlation_strength': self.correlation_strength,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'spurious_positions': self.spurious_positions
        }