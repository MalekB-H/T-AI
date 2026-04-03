"""
Shaped Reward Taxi Environment
Taxi-v3 wrapper with custom reward shaping to speed up learning.
Compares default rewards vs shaped rewards.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# The 4 fixed locations in Taxi-v3: R(0,0), G(0,4), Y(4,0), B(4,3)
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]


def _decode_taxi_state(state):
    """Decode Taxi-v3 state into (taxi_row, taxi_col, pass_loc, dest)."""
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return taxi_row, taxi_col, pass_loc, dest


def _manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class ShapedRewardTaxiEnv(gym.Wrapper):
    """
    Wrapper around Taxi-v3 that modifies the reward signal.

    Reward modes:
    - "default": original Taxi-v3 rewards (-1 step, -10 illegal, +20 dropoff)
    - "distance": adds distance-based shaping (closer to target = bonus)
    - "milestone": gives intermediate rewards for pickup
    - "aggressive": heavy penalties for wrong moves, big bonus for speed
    """

    MODES = ["default", "distance", "milestone", "aggressive"]

    def __init__(self, reward_mode="distance", render_mode=None):
        env = gym.make("Taxi-v3", render_mode=render_mode)
        super().__init__(env)
        self.reward_mode = reward_mode
        self._prev_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = self._get_target_distance(obs)
        return obs, info

    def _get_target_distance(self, state):
        """Distance from taxi to current target (passenger or destination)."""
        taxi_row, taxi_col, pass_loc, dest = _decode_taxi_state(state)
        if pass_loc < 4:
            # passenger not picked up yet -> target is passenger
            target = LOCS[pass_loc]
        else:
            # passenger in taxi -> target is destination
            target = LOCS[dest]
        return _manhattan((taxi_row, taxi_col), target)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.reward_mode == "default":
            # no modification
            pass

        elif self.reward_mode == "distance":
            # add distance-based shaping: reward for getting closer
            curr_dist = self._get_target_distance(obs)
            if self._prev_dist is not None and action < 4:
                # moving action: reward based on distance change
                if curr_dist < self._prev_dist:
                    reward += 2    # getting closer
                elif curr_dist > self._prev_dist:
                    reward -= 1.5  # getting farther
            self._prev_dist = curr_dist

        elif self.reward_mode == "milestone":
            # intermediate reward for pickup
            taxi_row, taxi_col, pass_loc, dest = _decode_taxi_state(obs)
            if action == 4 and reward >= 0 and pass_loc == 4:
                reward += 10  # pickup bonus
            # small distance bonus
            curr_dist = self._get_target_distance(obs)
            if self._prev_dist is not None and action < 4:
                if curr_dist < self._prev_dist:
                    reward += 1
            self._prev_dist = curr_dist

        elif self.reward_mode == "aggressive":
            # heavy penalties, big bonuses
            if done and reward >= 0:
                reward = 50     # big dropoff bonus
            elif action == 4 or action == 5:
                if reward == -10:
                    reward = -25  # harsh illegal penalty
            else:
                reward = -2     # double time penalty

        return obs, reward, done, truncated, info
