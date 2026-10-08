"""Verify the translated evaluation assets without making external API calls."""

from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
N_STATES = 21
N_ACTIONS = 4


def build_expert_model():
    """Build the architecture used by the goal, prize, and trap experts."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Embedding(input_dim=N_STATES, output_dim=32),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(N_ACTIONS),
        ]
    )


def build_global_model():
    """Build the expanded architecture used by the global benchmark."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Embedding(input_dim=N_STATES * 2, output_dim=64),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(N_ACTIONS),
        ]
    )


def load_and_check_models():
    model_specs = {
        "Goal Expert": (build_expert_model(), "goal_expert_open_map.weights.h5"),
        "Prize Expert": (build_expert_model(), "prize_expert_open_map.weights.h5"),
        "Trap Expert": (build_expert_model(), "trap_expert_open_map.weights.h5"),
        "Global Benchmark": (
            build_global_model(),
            "global_benchmark_open_map.weights.h5",
        ),
    }

    print("Weight verification")
    print("-------------------")
    for name, (model, filename) in model_specs.items():
        path = ROOT / filename
        model.load_weights(path)
        q_values = model(np.array([[0]], dtype=np.int32), training=False).numpy()[0]
        if q_values.shape != (N_ACTIONS,) or not np.isfinite(q_values).all():
            raise ValueError(f"Invalid Q-values produced by {name}: {q_values}")
        print(
            f"OK  {name:<18} {filename:<43} "
            f"Q(state=0)={np.round(q_values, 3)}"
        )


def load_and_summarize_evaluation():
    results_path = ROOT / "moe_evaluation_results.pkl"
    with results_path.open("rb") as file:
        results = pickle.load(file)

    print("\nCached evaluation summary")
    print("-------------------------")
    print(
        f"{'Strategy':<18} {'Episodes':>8} {'Success':>9} "
        f"{'Prize':>9} {'Trap':>9} {'Avg reward':>12} {'Avg steps':>10}"
    )

    for strategy, stats in results.items():
        rewards = np.asarray(stats["total_rewards"], dtype=float)
        steps = np.asarray(stats["steps_taken"], dtype=float)
        episodes = len(rewards)
        if episodes == 0:
            raise ValueError(f"No cached episodes found for {strategy}")

        success_rate = stats["success_count"] / episodes * 100
        prize_rate = stats["prize_collected_count"] / episodes * 100
        trap_rate = stats["trap_hit_count"] / episodes * 100
        average_steps = float(np.mean(steps)) if len(steps) else float("nan")

        print(
            f"{strategy:<18} {episodes:>8} {success_rate:>8.1f}% "
            f"{prize_rate:>8.1f}% {trap_rate:>8.1f}% "
            f"{np.mean(rewards):>12.2f} {average_steps:>10.2f}"
        )


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    load_and_check_models()
    load_and_summarize_evaluation()
    print("\nAll local evaluation assets passed verification.")
