"""
Multi-Passenger Taxi Environment
Extension of Taxi-v3 where the taxi must pick up and drop off 2 passengers.
Each passenger has a random start location and a random destination among R, G, Y, B.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# The 4 fixed locations in Taxi-v3: R, G, Y, B
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# Taxi-v3 map walls (for movement logic)
# MAP[row][col] = set of blocked directions
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class MultiPassengerTaxiEnv(gym.Env):
    """
    Taxi environment with 2 passengers.

    State encoding:
        taxi_row (0-4) * (5 * 6 * 4 * 6 * 4) +
        taxi_col (0-4) * (6 * 4 * 6 * 4) +
        pass1_loc (0-5) * (4 * 6 * 4) +
        dest1 (0-3) * (6 * 4) +
        pass2_loc (0-5) * 4 +
        dest2 (0-3)

    pass_loc: 0-3 = at location R/G/Y/B, 4 = in taxi, 5 = delivered
    dest: 0-3 = R/G/Y/B

    Total states: 5 * 5 * 6 * 4 * 6 * 4 = 14400
    Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Discrete(14400)
        self.action_space = spaces.Discrete(6)
        self.render_mode = render_mode

        # parse walls from map
        self._walls = self._parse_walls()

        self.taxi_row = 0
        self.taxi_col = 0
        self.pass1_loc = 0
        self.dest1 = 0
        self.pass2_loc = 0
        self.dest2 = 0
        self.steps = 0
        self.max_steps = 400

    def _parse_walls(self):
        """Parse walls: returns set of (row, col, direction) that are blocked."""
        walls = set()
        for r in range(5):
            map_row = MAP[r + 1]  # skip top border
            for c in range(5):
                char_idx = 1 + c * 2  # position in string
                if c < 4:
                    # check wall to the right: character at char_idx + 1
                    wall_char = map_row[char_idx + 1]
                    if wall_char == '|':
                        walls.add((r, c, 2))      # east blocked from (r,c)
                        walls.add((r, c + 1, 3))   # west blocked from (r,c+1)
        return walls

    def _can_move(self, row, col, direction):
        """Check if movement is possible. direction: 0=S,1=N,2=E,3=W"""
        if direction == 0:  # south
            return row < 4
        elif direction == 1:  # north
            return row > 0
        elif direction == 2:  # east
            return col < 4 and (row, col, 2) not in self._walls
        elif direction == 3:  # west
            return col > 0 and (row, col, 3) not in self._walls
        return False

    def _encode_state(self):
        return (self.taxi_row * 5 * 6 * 4 * 6 * 4 +
                self.taxi_col * 6 * 4 * 6 * 4 +
                self.pass1_loc * 4 * 6 * 4 +
                self.dest1 * 6 * 4 +
                self.pass2_loc * 4 +
                self.dest2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.taxi_row = self.np_random.integers(0, 5)
        self.taxi_col = self.np_random.integers(0, 5)

        # passenger 1: random start and destination (different)
        self.pass1_loc = self.np_random.integers(0, 4)
        self.dest1 = self.np_random.integers(0, 4)
        while self.dest1 == self.pass1_loc:
            self.dest1 = self.np_random.integers(0, 4)

        # passenger 2: random start and destination (different from each other)
        self.pass2_loc = self.np_random.integers(0, 4)
        self.dest2 = self.np_random.integers(0, 4)
        while self.dest2 == self.pass2_loc:
            self.dest2 = self.np_random.integers(0, 4)

        self.steps = 0
        return self._encode_state(), {}

    def step(self, action):
        self.steps += 1
        reward = -1  # time penalty
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

        elif action == 4:  # pickup
            picked = False
            # try pickup passenger 1
            if self.pass1_loc < 4 and LOCS[self.pass1_loc] == taxi_pos:
                self.pass1_loc = 4  # in taxi
                picked = True
            # try pickup passenger 2 (only if pass1 is delivered)
            elif self.pass2_loc < 4 and self.pass1_loc == 5 and LOCS[self.pass2_loc] == taxi_pos:
                self.pass2_loc = 4  # in taxi
                picked = True

            if not picked:
                reward = -10

        elif action == 5:  # dropoff
            dropped = False
            # try dropoff passenger 1
            if self.pass1_loc == 4 and LOCS[self.dest1] == taxi_pos:
                self.pass1_loc = 5  # delivered
                reward = 20
                dropped = True
            # try dropoff passenger 2
            elif self.pass2_loc == 4 and LOCS[self.dest2] == taxi_pos:
                self.pass2_loc = 5  # delivered
                reward = 20
                dropped = True

            if not dropped:
                reward = -10

        # check if both passengers delivered
        if self.pass1_loc == 5 and self.pass2_loc == 5:
            done = True

        # truncate if too many steps
        truncated = self.steps >= self.max_steps

        return self._encode_state(), reward, done, truncated, {}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self):
        grid = [list(row) for row in MAP]

        # place taxi
        r_idx = self.taxi_row + 1
        c_idx = 1 + self.taxi_col * 2

        # determine taxi character
        if self.pass1_loc == 4 or self.pass2_loc == 4:
            taxi_char = '@'  # has passenger
        else:
            taxi_char = 'T'

        # check if taxi is on a location letter
        loc_letters = {(0, 0): 'R', (0, 4): 'G', (4, 0): 'Y', (4, 3): 'B'}
        if (self.taxi_row, self.taxi_col) not in loc_letters:
            grid[r_idx][c_idx] = taxi_char

        lines = [''.join(row) for row in grid]

        # status
        p1_status = "delivered" if self.pass1_loc == 5 else ("in taxi" if self.pass1_loc == 4 else f"at {['R','G','Y','B'][self.pass1_loc]}")
        p2_status = "delivered" if self.pass2_loc == 5 else ("in taxi" if self.pass2_loc == 4 else f"at {['R','G','Y','B'][self.pass2_loc]}")

        lines.append(f"Taxi: ({self.taxi_row},{self.taxi_col})")
        lines.append(f"Pass1: {p1_status} -> {['R','G','Y','B'][self.dest1]}")
        lines.append(f"Pass2: {p2_status} -> {['R','G','Y','B'][self.dest2]}")

        return '\n'.join(lines)
