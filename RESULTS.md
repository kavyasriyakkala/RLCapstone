# Algorithm Comparison Results

This section summarizes the algorithm comparison experiment for the shortcut gridworld.
The question is whether shortcut learning is specific to PPO or also appears in other
RL algorithms.

## Setup

The environment contains a true goal channel and a spurious shortcut channel. During
training, the spurious marker is placed on the true goal. During testing, the spurious
marker is placed away from the true goal.

Success is measured in both environments:

- `train_success`: success when the shortcut remains aligned with the goal.
- `test_success`: success when the shortcut is broken.
- `delta(t) = train_success - test_success`: the shortcut gap.

A large positive `delta(t)` means the agent performs well when the shortcut is valid
but fails to transfer when the shortcut is moved.

## Behavioral Results

The aggregate comparison is saved in:

`results/figures/algorithm_comparison_aggregate.png`

The per-run PPO, A2C, and DQN plots are saved under:

`results/algorithms/ppo/`

`results/algorithms/a2c/`

`results/algorithms/dqn/`

Across the available algorithm-comparison runs, the final-step summaries were:

| Algorithm | Final train success | Final test success | Final shortcut gap |
| --- | ---: | ---: | ---: |
| PPO | 0.650 +/- 0.076 | 0.233 +/- 0.122 | 0.417 +/- 0.090 |
| A2C | 0.052 +/- 0.036 | 0.041 +/- 0.041 | 0.011 +/- 0.038 |
| DQN | 0.996 +/- 0.010 | 0.200 +/- 0.075 | 0.796 +/- 0.079 |

The A2C and DQN rows use all nine runs across `k=1,2,3` and seeds
`42,123,7`. The PPO row uses the available algorithm-comparison PPO logs
present in `results/algorithms/ppo/` (`n=4`).

Interpretation:

- PPO shows a meaningful shortcut gap: train success is substantially higher than
  test success, so the baseline PPO agent partially relies on the training
  shortcut.
- DQN strongly learns the training distribution but fails to transfer well. Its large
  shortcut gap suggests it is maximizing reward through the spurious shortcut.
- A2C does not show a large shortcut gap, but its train and test success are both low.
  This means A2C mostly failed to learn the task well in this setup, rather than
  learning a robust goal-directed policy.
- The main evidence for shortcut learning should come from the behavioral train/test
  curves, not from probes alone.

## PPO Intervention Results

The PPO intervention comparison is saved in:

`results/figures/intervention_comparison_k1_seed42.png`

For `k=1`, `seed=42`, the final-step results were:

| PPO variant | Final train success | Final test success | Final shortcut gap |
| --- | ---: | ---: | ---: |
| Vanilla PPO | 0.400 | 0.067 | 0.333 |
| PAR PPO | 0.400 | 0.267 | 0.133 |

PAR PPO keeps the same final train success as vanilla PPO in this run, but improves
test success and reduces the shortcut gap from `0.333` to `0.133`. This suggests
PAR reduces shortcut reliance compared with vanilla PPO, although it does not remove
the gap completely.

## Probe Results

Checkpoint-based linear probe curves were generated for A2C and DQN at `k=1`,
`seed=42`. These match the format of the original PPO probe figure:

`results/figures/linear_probes_a2c_k1_seed42.png`

`results/figures/linear_probes_dqn_k1_seed42.png`

The probes predict goal position and spurious-marker position from hidden-layer
activations across training checkpoints. For both A2C and DQN, goal and spurious
positions were almost perfectly decodable:

| Algorithm | Mean goal probe R^2 | Mean spurious probe R^2 | Difference scale |
| --- | ---: | ---: | ---: |
| A2C | 0.9999997 | 0.9999997 | about 1e-7 |
| DQN | 0.9999996 | 0.9999996 | about 1e-7 |

Interpretation:

- The probes show that both goal and spurious information are present in the hidden
  representations.
- The probes do not show a meaningful representation-level preference for either the
  goal or the shortcut in A2C/DQN.
- Therefore, the behavioral plots are more important for deciding whether an agent
  learned the real task or exploited the shortcut.

## Conclusion

DQN provides the clearest evidence that shortcut learning is not PPO-specific. It
achieves near-perfect training success while maintaining low test success, producing
a large shortcut gap. PPO also shows a positive shortcut gap, while PAR PPO reduces
that gap in the `k=1`, `seed=42` intervention setting. A2C does not show the same
gap, but this is mainly because it does not learn the task effectively under the
current setup.

Overall, the results suggest that reward maximization under a shortcut-correlated
training distribution can produce shortcut reliance beyond PPO, especially for DQN.
