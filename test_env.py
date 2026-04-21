from envs.gridworld import ShortcutGridWorld

# Test training environment
print("Testing training environment...")
env = ShortcutGridWorld(grid_size=10, num_spurious=1, train_mode=True)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Grid info: {env.get_info()}")

# Take a few random steps
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={terminated}")

# Render it
env.render()
print("\nTraining env test PASSED")

# Test eval environment  
print("\nTesting eval environment...")
env_test = ShortcutGridWorld(grid_size=10, num_spurious=1, train_mode=False)
obs, info = env_test.reset()
env_test.render()
print("Eval env test PASSED")