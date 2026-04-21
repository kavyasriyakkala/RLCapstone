import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, *obs_shape)
        self.feature_dim = self.conv(dummy).shape[1]

    def forward(self, x):
        return self.conv(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.actor = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.actor(features), self.critic(features)

    def get_action(self, x):
        features = self.encoder(x)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

    def get_features(self, x):
        with torch.no_grad():
            return self.encoder(x)


class PPOAgent:
    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, epochs=4, batch_size=64):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Storage
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []

    def store(self, obs, act, logp, rew, val, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def clear(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.done_buf.clear()

    def compute_returns(self, last_val=0):
        returns = []
        R = last_val
        for r, d in zip(reversed(self.rew_buf), reversed(self.done_buf)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def update(self):
        returns = self.compute_returns()

        obs = torch.FloatTensor(np.array(self.obs_buf)).to(self.device)
        acts = torch.LongTensor(self.act_buf).to(self.device)
        old_logps = torch.FloatTensor(self.logp_buf).to(self.device)
        rets = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        rets = (rets - rets.mean()) / (rets.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs):
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                b_obs = obs[batch_idx]
                b_acts = acts[batch_idx]
                b_old_logps = old_logps[batch_idx]
                b_rets = rets[batch_idx]

                logits, values = self.model(b_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()

                # PPO clip loss
                ratio = (new_logps - b_old_logps).exp()
                adv = b_rets - values.squeeze()
                loss1 = ratio * adv
                loss2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(loss1, loss2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), b_rets)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.clear()
        return total_loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")