# Reward Hacking in Shortcut Gridworlds

This project studies **reward hacking** and **shortcut learning** in reinforcement learning.

The core question is:

> When a training environment contains a spurious feature that is perfectly correlated with reward, does an RL agent learn the real task or exploit the shortcut?

The project uses a custom gridworld where the true goal and a shortcut marker overlap during training, but separate during testing. This setup makes it possible to measure whether agents learned the actual goal or simply followed the shortcut.

---

## Project Flow

### 1. Environment

The main environment is:

```text
envs/gridworld.py
```

It defines a shortcut gridworld with three observation channels:

```text
channel 0: agent position
channel 1: true goal position
channel 2: spurious shortcut marker position
```

During training:

```text
spurious marker = true goal
```

During testing:

```text
spurious marker != true goal
```

The reward is always based only on reaching the true goal. The shortcut marker does not directly give reward, but it is perfectly correlated with reward during training.

This creates a reward-hacking setting: an agent can perform well in training by following the marker, but fail at test time when the marker is moved.

---

## Main Metric

The main behavioral metric is the **shortcut gap**:

```text
shortcut gap = train success - test success
```

A large positive shortcut gap means the agent succeeds when the shortcut is valid, but fails when the shortcut is broken.

---

## Repository Structure

```text
RLCapstone/
│
├── envs/
│   ├── gridworld.py
│   └── __init__.py
│
├── agents/
│   ├── ppo_agent.py
│   └── __init__.py
│
├── experiments/
│   ├── train.py
│   ├── train_algorithms.py
│   ├── train_interventions.py
│   ├── train_probe_checkpoints.py
│   ├── linear_probes.py
│   ├── probe_checkpoint_curves.py
│   ├── ablation_eval.py
│   ├── plot_algorithm_comparison.py
│   ├── plot_interventions.py
│   ├── plot_interventions_multiseed.py
│   ├── plot_k_comparison.py
│   ├── plot_seeds.py
│   └── replot_probes.py
│
├── results/
│   ├── figures/
│   ├── algorithms/
│   └── interventions/
│
├── utils/
├── test_env.py
├── requirements.txt
└── README.md
```

---

## Important Files

### Environment

```text
envs/gridworld.py
```

Defines the custom shortcut gridworld environment. This is the core environment used by all training and evaluation scripts.

---

### Baseline PPO Training

```text
experiments/train.py
```

Runs the baseline PPO experiment on the shortcut gridworld.

Use this first if you want to understand the simplest training setup.

---

### Algorithm Comparison

```text
experiments/train_algorithms.py
```

Trains and compares multiple RL algorithms:

```text
PPO
A2C
DQN
```

This is the main script for testing whether shortcut learning appears across different RL algorithms.

Plot the comparison with:

```text
experiments/plot_algorithm_comparison.py
```

---

### PPO Interventions

```text
experiments/train_interventions.py
```

Runs PPO with different intervention methods designed to reduce shortcut reliance:

```text
vanilla       standard PPO
l2            PPO with weight decay
augmentation  observation augmentation
par           reward transformation wrapper
```

Plot intervention results with:

```text
experiments/plot_interventions.py
experiments/plot_interventions_multiseed.py
```

---

### Linear Probes

```text
experiments/linear_probes.py
experiments/train_probe_checkpoints.py
experiments/probe_checkpoint_curves.py
```

These scripts analyze whether trained models encode information about:

```text
true goal position
spurious marker position
```

The probe experiments help inspect internal representations, but the main evidence for reward hacking comes from behavior under train/test distribution shift.

---

### Ablation Evaluation

```text
experiments/ablation_eval.py
```

Evaluates trained agents while modifying or removing observation channels. This helps test which parts of the observation the policy depends on.

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline PPO experiment:

```bash
python experiments/train.py
```

Run algorithm comparison experiments:

```bash
python experiments/train_algorithms.py --all --algorithms a2c,dqn
```

Plot aggregate algorithm results:

```bash
python experiments/plot_algorithm_comparison.py
```

Run PPO intervention experiments:

```bash
python experiments/train_interventions.py --method vanilla --k 1 --seed 42
python experiments/train_interventions.py --method l2 --k 1 --seed 42
python experiments/train_interventions.py --method augmentation --k 1 --seed 42
python experiments/train_interventions.py --method par --k 1 --seed 42
```

Plot intervention results:

```bash
python experiments/plot_interventions.py
```

Run probe checkpoint training:

```bash
python experiments/train_probe_checkpoints.py --targets a2c,dqn --k 1 --seed 42
```

Generate probe curves:

```bash
python experiments/probe_checkpoint_curves.py \
  --targets a2c,dqn \
  --k 1 \
  --seed 42 \
  --episodes 20 \
  --checkpoint-stride 5
```

---

## How to Read the Project

A good order for newcomers is:

1. Start with `envs/gridworld.py` to understand the environment.
2. Read `experiments/train.py` for the baseline PPO setup.
3. Read `experiments/train_algorithms.py` to see how PPO, A2C, and DQN are compared.
4. Read `experiments/train_interventions.py` to understand attempted fixes.
5. Read the probe scripts if you want to understand representation-level analysis.

---

## Summary

This project shows how reinforcement learning agents can exploit a shortcut feature that is predictive during training but unreliable during testing.

The key idea is simple:

```text
Training: shortcut marker overlaps with the true goal.
Testing: shortcut marker moves away from the true goal.
```

If an agent learned the real task, it should still go to the true goal.

If it learned the shortcut, test performance drops.

The shortcut gap captures this failure mode.
