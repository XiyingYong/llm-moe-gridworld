import os
import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf

from maze_environment import CustomMazeEnv


warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# 1. Conflict-map definition.
conflict_map = [
    "SFFFFFG",
    "FTFPFTF",
    "FFFTFFF",
]


class GlobalPOMDPWrapper(gym.Wrapper):
    """Add prize-memory state to the environment used by the global benchmark.

    The wrapper expands the original 21 states to 42 states:
    - States 0-20 represent positions before the prize is collected.
    - States 21-41 represent positions after the prize is collected.
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_states_base = env.observation_space.n
        self.observation_space = gym.spaces.Discrete(self.n_states_base * 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.prize_collected:
            obs += self.n_states_base
        return obs, reward, terminated, truncated, info


def build_dqn_model(n_states, n_actions):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            # The embedding represents both the pre-prize and post-prize states.
            tf.keras.layers.Embedding(input_dim=n_states, output_dim=64),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_actions),
        ]
    )


def epsilon_greedy_policy(state, model, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_values = model.predict(np.array([[state]]), verbose=0)
    return np.argmax(q_values[0])


def sample_experiences(replay_buffer, batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ]


def training_step(
    batch_size,
    model,
    replay_buffer,
    optimizer,
    loss_fn,
    discount_factor,
    n_actions,
):
    states, actions, rewards, next_states, dones, truncateds = sample_experiences(
        replay_buffer, batch_size
    )

    states = states.reshape(-1, 1)
    next_states = next_states.reshape(-1, 1)

    next_q_values = model.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)
    runs = 1.0 - (dones | truncateds)
    target_q_values = rewards + runs * discount_factor * max_next_q_values
    target_q_values = target_q_values.reshape(-1, 1)

    mask = tf.one_hot(actions, n_actions)

    with tf.GradientTape() as tape:
        all_q_values = model(states)
        q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q_values, q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def play_one_step(env, state, model, replay_buffer, epsilon, n_actions):
    action = epsilon_greedy_policy(state, model, epsilon, n_actions)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info


def train_agent(
    env,
    model,
    num_iterations,
    optimizer,
    loss_fn,
    batch_size,
    discount_factor,
    epsilon_decay_steps=3000,
):
    replay_buffer = deque(maxlen=20000)
    rewards_history = []

    print(f"Starting training. Total episodes: {num_iterations}")

    for episode in range(num_iterations):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(150):
            # Trust the learned policy increasingly as training progresses.
            epsilon = max(1 - episode / epsilon_decay_steps, 0.01)

            obs, reward, done, truncated, info = play_one_step(
                env, obs, model, replay_buffer, epsilon, env.action_space.n
            )
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                training_step(
                    batch_size,
                    model,
                    replay_buffer,
                    optimizer,
                    loss_fn,
                    discount_factor,
                    env.action_space.n,
                )

            if done or truncated:
                break

        rewards_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode: {episode + 1:04d}/{num_iterations} | "
                f"Avg Reward (last 100): {avg_reward:7.2f} | "
                f"epsilon: {epsilon:.3f}"
            )

    return model


if __name__ == "__main__":
    print("=== Initializing the Global Benchmark Environment ===")

    base_env = CustomMazeEnv(expert_type="global", custom_map=conflict_map)
    env_global = GlobalPOMDPWrapper(base_env)

    n_states = env_global.observation_space.n
    n_actions = env_global.action_space.n

    print(f"=== Building the DQN Model (state space: {n_states}) ===")
    model_global = build_dqn_model(n_states, n_actions)

    # Use a lower learning rate and gradient clipping for stable training.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()
    batch_size = 128
    discount_factor = 0.99
    num_iterations = 4000

    train_agent(
        env=env_global,
        model=model_global,
        num_iterations=num_iterations,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_size=batch_size,
        discount_factor=discount_factor,
        epsilon_decay_steps=3000,
    )

    save_path = "global_benchmark_conflict_map.weights.h5"
    model_global.save_weights(save_path)
    print(f"\nGlobal benchmark training complete. Weights saved to: {save_path}")
