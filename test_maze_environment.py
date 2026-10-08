from maze_environment import CustomMazeEnv


# 1. Create an environment for the goal-seeking expert.
env = CustomMazeEnv(expert_type="goal")
print(f"Observation Space: {env.observation_space.n}")
print(f"Action Space: {env.action_space.n}")

# 2. Reset and render the environment.
obs, info = env.reset()
env.render()

# 3. Take up to ten random actions as a smoke test.
for step in range(10):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step: {step + 1}, Action: {action}, Reward: {reward}, "
        f"Terminated: {terminated}"
    )
    env.render()

    if terminated or truncated:
        print("Episode finished!")
        break
