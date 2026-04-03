"""
Obstacle Taxi Environment
Taxi-v3 with danger zones that give -20 penalty.
Used to compare Q-Learning (off-policy, risky) vs SARSA (on-policy, safe).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

# Danger zones: scattered across the grid to block multiple routes
# (1,1) blocks R→Y path, (2,3) blocks G→B path, (3,1) blocks Y→B path
DANGER_ZONES = [(1, 1), (2, 3), (3, 1)]
DANGER_PENALTY = -100


class ObstacleTaxiEnv(gym.Env):
    """
    Taxi-v3 with danger zones.

    Same state encoding as Taxi-v3:
        (taxi_row, taxi_col, passenger_location, destination) -> 500 states
        passenger_location: 0-3 = R/G/Y/B, 4 = in taxi
        destination: 0-3 = R/G/Y/B

    Danger zones give -20 reward when the taxi moves onto them.
    This makes SARSA (on-policy) learn to AVOID them,
    while Q-Learning (off-policy) walks right next to them.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Discrete(500)
        self.action_space = spaces.Discrete(6)
        self.render_mode = render_mode

        self._walls = self._parse_walls()
        self.taxi_row = 0
        self.taxi_col = 0
        self.pass_loc = 0
        self.dest = 0
        self.steps = 0
        self.max_steps = 200

    def _parse_walls(self):
        walls = set()
        for r in range(5):
            map_row = MAP[r + 1]
            for c in range(5):
                char_idx = 1 + c * 2
                if c < 4:
                    wall_char = map_row[char_idx + 1]
                    if wall_char == '|':
                        walls.add((r, c, 2))
                        walls.add((r, c + 1, 3))
        return walls

    def _can_move(self, row, col, direction):
        if direction == 0:
            return row < 4
        elif direction == 1:
            return row > 0
        elif direction == 2:
            return col < 4 and (row, col, 2) not in self._walls
        elif direction == 3:
            return col > 0 and (row, col, 3) not in self._walls
        return False

    def _encode_state(self):
        return ((self.taxi_row * 5 + self.taxi_col) * 5 + self.pass_loc) * 4 + self.dest

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.taxi_row = self.np_random.integers(0, 5)
        self.taxi_col = self.np_random.integers(0, 5)

        self.pass_loc = self.np_random.integers(0, 4)
        self.dest = self.np_random.integers(0, 4)
        while self.dest == self.pass_loc:
            self.dest = self.np_random.integers(0, 4)

        self.steps = 0
        return self._encode_state(), {}

    def step(self, action):
        self.steps += 1
        reward = -1
        done = False

        taxi_pos = (self.taxi_row, self.taxi_col)

        if action < 4:  # movement
            if self._can_move(self.taxi_row, self.taxi_col, action):
                if action == 0:
                    self.taxi_row += 1
                elif action == 1:
                    self.taxi_row -= 1
                elif action == 2:
                    self.taxi_col += 1
                elif action == 3:
                    self.taxi_col -= 1

            # DANGER ZONE penalty
            if (self.taxi_row, self.taxi_col) in DANGER_ZONES:
                reward = DANGER_PENALTY

        elif action == 4:  # pickup
            if self.pass_loc < 4 and LOCS[self.pass_loc] == taxi_pos:
                self.pass_loc = 4
            else:
                reward = -10

        elif action == 5:  # dropoff
            if self.pass_loc == 4 and LOCS[self.dest] == taxi_pos:
                self.pass_loc = self.dest
                reward = 20
                done = True
            else:
                reward = -10

        truncated = self.steps >= self.max_steps
        return self._encode_state(), reward, done, truncated, {}

    def get_danger_zones(self):
        """Return list of danger zone positions for visualization."""
        return DANGER_ZONES

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self):
        grid = [list(row) for row in MAP]

        # mark danger zones with X
        for dr, dc in DANGER_ZONES:
            r_idx = dr + 1
            c_idx = 1 + dc * 2
            grid[r_idx][c_idx] = 'X'

        # place taxi
        r_idx = self.taxi_row + 1
        c_idx = 1 + self.taxi_col * 2
        if self.pass_loc == 4:
            grid[r_idx][c_idx] = '@'
        else:
            grid[r_idx][c_idx] = 'T'

        lines = [''.join(row) for row in grid]

        p_status = "in taxi" if self.pass_loc == 4 else f"at {['R','G','Y','B'][self.pass_loc]}"
        lines.append(f"Taxi: ({self.taxi_row},{self.taxi_col})")
        lines.append(f"Pass: {p_status} -> {['R','G','Y','B'][self.dest]}")
        lines.append(f"Danger zones: {DANGER_ZONES}")

        return '\n'.join(lines)
