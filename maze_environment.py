import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, expert_type="goal", custom_map=None):
        super().__init__()

        # 1. Validate the expert type (including the global benchmark agent).
        assert expert_type in ["goal", "prize", "trap", "global"], (
            "expert_type must be 'goal', 'prize', 'trap', or 'global'"
        )
        self.expert_type = expert_type

        # 2. Define the maze layout.
        if custom_map is not None:
            self.desc = custom_map
        else:
            self.desc = [
                "SFFFFFF",
                "FFTFFPF",
                "FFFFTFG",
            ]

        self.nrow = len(self.desc)
        self.ncol = len(self.desc[0])
        self.n_states = self.nrow * self.ncol

        # 3. Define the action and observation spaces.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)

        self.current_state = 0
        self.prize_collected = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = 0
        self.prize_collected = False
        return self.current_state, {}

    def step(self, action):
        row = self.current_state // self.ncol
        col = self.current_state % self.ncol

        if action == 0:  # Left
            col = max(col - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # Right
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # Up
            row = max(row - 1, 0)

        self.current_state = row * self.ncol + col
        cell_type = self.desc[row][col]

        reward = 0.0
        terminated = False
        truncated = False
        info = {"cell_type": cell_type}

        # Core reward logic.
        if cell_type == "G":  # Goal
            terminated = True
            if self.expert_type == "goal":
                reward = 10.0
            elif self.expert_type == "prize":
                reward = 1.0
            elif self.expert_type == "trap":
                reward = 10.0
            elif self.expert_type == "global":
                reward = 100.0

        elif cell_type == "T":  # Trap
            terminated = True
            if self.expert_type == "goal":
                reward = -5.0
            elif self.expert_type == "prize":
                reward = -5.0
            elif self.expert_type == "trap":
                reward = -20.0
            elif self.expert_type == "global":
                reward = -40.0

        elif cell_type == "P":  # Prize
            if not self.prize_collected:
                self.prize_collected = True
                if self.expert_type == "goal":
                    reward = 0.0
                elif self.expert_type == "prize":
                    reward = 50.0
                    terminated = True
                elif self.expert_type == "trap":
                    reward = 0.0
                elif self.expert_type == "global":
                    reward = 35.0
                    # The global agent must continue to the goal after collecting the prize.
                    terminated = False
            else:
                # Handle revisits after the prize has already been collected.
                if self.expert_type in ["goal", "prize", "trap"]:
                    reward = 0.0
                elif self.expert_type == "global":
                    # Treat an empty prize cell as an ordinary floor cell.
                    reward = -0.1

        else:  # Ordinary floor: "F" or "S"
            reward = -0.1

        return self.current_state, reward, terminated, truncated, info

    def render(self):
        row = self.current_state // self.ncol
        col = self.current_state % self.ncol

        map_list = [list(r) for r in self.desc]
        map_list[row][col] = "X"

        print("-" * (self.ncol * 2 + 1))
        for r in map_list:
            print("|" + "|".join(r) + "|")
        print("-" * (self.ncol * 2 + 1))
        print(
            f"Current State: {self.current_state}, "
            f"Prize Collected: {self.prize_collected}"
        )
        print()
