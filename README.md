# Reward Hacking in Shortcut Gridworlds

This repository studies reward hacking and shortcut learning in reinforcement
learning. The central question is:

> When a training environment contains a spurious feature that is perfectly
> correlated with reward, do RL agents learn the real task or exploit the
> shortcut?

The project uses a custom gridworld environment where the true goal and a
spurious marker overlap during training but separate during testing. This lets us
measure whether an agent learned to navigate to the real goal or whether it
learned a brittle shortcut that only works in the training distribution.

## Project Summary

The environment provides three observation channels:

1. Agent position.
2. True goal position.
3. Spurious shortcut-marker position.

During training, the spurious marker is placed directly on top of the true goal.
During testing, the spurious marker is moved away from the true goal. The reward
is always tied only to the true goal:

```text
reward = +1.0   if the agent reaches the true goal
reward = -0.01  otherwise
```

The spurious marker never directly gives reward. It only becomes useful because
it is correlated with the true goal during training. If an agent follows the
spurious marker during testing, it will fail unless it eventually reaches the
actual goal.

The main behavioral metric is the shortcut gap:

```text
delta(t) = train_success(t) - test_success(t)
```

A large positive shortcut gap means the agent performs well when the shortcut is
valid but fails when the shortcut is broken.

## Repository Structure

```text
RLCapstone/
  envs/
    gridworld.py                  Custom shortcut gridworld environment
  agents/
    ppo_agent.py                  Earlier custom PPO implementation
  experiments/
    train.py                      Baseline PPO training
    train_interventions.py        PPO interventions: vanilla, L2, augmentation, PAR
    train_algorithms.py           PPO, A2C, DQN algorithm-comparison training
    plot_algorithm_comparison.py  Aggregate PPO/A2C/DQN comparison plots
    linear_probes.py              Original PPO checkpoint probing
    train_probe_checkpoints.py    Checkpointed A2C/DQN training for probe curves
    probe_checkpoint_curves.py    A2C/DQN probe curves over training steps
    ablation_eval.py              Observation-channel ablation evaluation
  results/
    figures/                      Saved figures used in the report
    algorithms/                   A2C/DQN/PPO logs and trained models
    interventions/                PPO intervention logs and models
  RESULTS.md                      Concise result summary
  requirements.txt                Python dependencies
```

## Environment Design

The core environment is `ShortcutGridWorld` in `envs/gridworld.py`.

For a `10x10` grid, the observation has shape:

```text
(3, 10, 10)
```

The three channels are:

```text
channel 0: agent location
channel 1: true goal location
channel 2: spurious marker location
```

For Stable-Baselines3 training, the observation is flattened before being passed
to an MLP policy:

```text
(3, 10, 10) -> 300-dimensional vector
```

At reset time, the environment randomly samples an agent position and a true goal
position. The spurious marker is then placed differently depending on the split:

```python
if train_mode:
    spurious = goal_pos
else:
    spurious = random_position_away_from_goal
```

The episode ends when either:

1. The agent reaches the true goal.
2. The maximum step limit is reached.

The maximum episode length is:

```text
max_steps = grid_size * grid_size * 4
```

For a `10x10` grid this is `400` steps.

## Why This Creates Reward Hacking

The true reward function depends only on the goal:

```text
reward = 1 if agent_pos == goal_pos
```

However, during training:

```text
spurious_pos == goal_pos
```

So the spurious marker is not causal, but it is perfectly predictive. An agent
can maximize training reward by learning:

```text
go to the spurious marker
```

instead of learning:

```text
go to the true goal channel
```

Testing breaks this shortcut by moving the spurious marker away from the goal. If
the policy learned the shortcut, test success drops and the shortcut gap becomes
positive.

## Algorithms

The project uses Stable-Baselines3 implementations of:

- PPO
- A2C
- DQN

All three use `MlpPolicy` over the flattened gridworld observation.

### PPO

PPO learns a stochastic policy:

```text
pi_theta(a | s)
```

and a value function:

```text
V_phi(s)
```

The advantage estimate measures whether an action was better or worse than
expected:

```text
A_t = return_t - V(s_t)
```

Stable-Baselines3 PPO uses generalized advantage estimation in practice, based on
temporal-difference errors:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

PPO compares the new policy to the old policy using:

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

The clipped PPO objective is:

```text
L_clip(theta) = E[min(r_t(theta) * A_t,
                      clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_t)]
```

In this project, if moving toward the spurious marker leads to reward during
training, those actions receive positive advantage and PPO increases their
probability.

### A2C

A2C is also an actor-critic method. It learns:

```text
actor:  pi_theta(a | s)
critic: V_phi(s)
```

The actor loss is:

```text
L_actor = -log pi_theta(a_t | s_t) * A_t
```

The critic loss is:

```text
L_critic = (V_phi(s_t) - return_t)^2
```

A2C differs from PPO because it does not use PPO's clipped probability-ratio
objective. It directly pushes actions up or down according to the advantage. In
our experiments, A2C did not develop a large shortcut gap, but it also had low
train and test success, meaning it mostly failed to learn the task well.

### DQN

DQN is value-based. Instead of directly learning a policy, it learns:

```text
Q_theta(s, a)
```

which estimates the expected future reward from taking action `a` in state `s`.
The policy chooses:

```text
a = argmax_a Q(s, a)
```

DQN is trained using the Bellman target:

```text
y_t = r_t + gamma * max_a Q_target(s_{t+1}, a)
```

and minimizes:

```text
L_DQN = (Q_theta(s_t, a_t) - y_t)^2
```

DQN also uses a replay buffer and a target network. In this project, DQN strongly
learned the shortcut: it achieved high training success but low test success.

## PPO Interventions

The project also compares several PPO variants:

- `vanilla`: standard PPO baseline.
- `l2`: PPO with weight decay.
- `augmentation`: PPO with observation-channel jitter during training.
- `par`: PPO with reward transformation through `PARRewardWrapper`.

These methods are intended to test whether regularization, augmentation, or
reward transformation reduces shortcut reliance.

## Experiments

### Baseline PPO

```bash
python experiments/train.py
```

This trains PPO on the shortcut gridworld and saves:

```text
results/checkpoints/
results/figures/
```

### PPO Interventions

```bash
python experiments/train_interventions.py --method vanilla --k 1 --seed 42
python experiments/train_interventions.py --method l2 --k 1 --seed 42
python experiments/train_interventions.py --method augmentation --k 1 --seed 42
python experiments/train_interventions.py --method par --k 1 --seed 42
```

Plot the intervention comparison with:

```bash
python experiments/plot_interventions.py
```

### Algorithm Comparison

Train A2C and DQN across `k=1,2,3` and seeds `42,123,7`:

```bash
python experiments/train_algorithms.py --all --algorithms a2c,dqn
```

Plot aggregate comparisons:

```bash
python experiments/plot_algorithm_comparison.py
```

The aggregate plot is saved to:

```text
results/figures/algorithm_comparison_aggregate.png
```

### Linear Probes

Linear probes test whether hidden activations encode the true goal position or
the spurious marker position.

For A2C/DQN probe curves, first train checkpointed models:

```bash
python experiments/train_probe_checkpoints.py --targets a2c,dqn --k 1 --seed 42
```

Then generate probe curves:

```bash
python experiments/probe_checkpoint_curves.py \
  --targets a2c,dqn \
  --k 1 \
  --seed 42 \
  --episodes 20 \
  --checkpoint-stride 5
```

This saves:

```text
results/figures/linear_probes_a2c_k1_seed42.png
results/figures/linear_probes_dqn_k1_seed42.png
```

## Results

See `RESULTS.md` for the full summary. The main results are:

- PPO shows a positive shortcut gap, meaning the policy partially relies on the
  training shortcut.
- DQN shows the strongest shortcut exploitation: near-perfect training success but
  poor test success.
- A2C does not show a strong shortcut gap, but it also fails to learn the task
  well in this setup.
- PAR PPO reduces the shortcut gap compared with vanilla PPO for `k=1`,
  `seed=42`.
- Linear probes show that both goal and spurious information are decodable from
  A2C/DQN hidden activations, but probes alone do not prove which feature the
  policy uses for action selection.

## Key Figures

```text
results/figures/sb3_training_k1_seed42.png
results/figures/intervention_comparison_k1_seed42.png
results/figures/algorithm_comparison_aggregate.png
results/figures/linear_probes_k1_seed42.png
results/figures/linear_probes_a2c_k1_seed42.png
results/figures/linear_probes_dqn_k1_seed42.png
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If Matplotlib cache warnings appear, use a local cache directory:

```bash
MPLCONFIGDIR=.matplotlib-cache python experiments/plot_algorithm_comparison.py
```

## Reproducing the Main Algorithm Comparison

From the repository root:

```bash
source .venv/bin/activate

python experiments/train_algorithms.py --all --algorithms a2c,dqn
python experiments/plot_algorithm_comparison.py
```

For probe curves:

```bash
python experiments/train_probe_checkpoints.py --targets a2c,dqn --k 1 --seed 42
python experiments/probe_checkpoint_curves.py \
  --targets a2c,dqn \
  --k 1 \
  --seed 42 \
  --episodes 20 \
  --checkpoint-stride 5
```

## Interpretation

This project demonstrates that shortcut learning is a distribution-shift problem:
the training reward can be maximized using a feature that is predictive in
training but not reliable in testing. The DQN results show that this failure mode
is not specific to PPO. The probe results show that hidden representations can
contain both true-goal and shortcut information, so behavioral evaluation under
test-time shortcut breaks is essential.