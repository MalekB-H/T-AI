"""Évaluation d'une Q-table en mode greedy sans exploration."""

import numpy as np
import gymnasium as gym


def test_policy(
    Q            : np.ndarray,
    test_episodes: int,
    verbose      : bool = False,
) -> tuple[list, list]:
    """
    Évalue une Q-table en mode greedy (sans exploration).

    Paramètres
    ----------
    Q             : Q-table à évaluer (500 × 6)
    test_episodes : nombre d'épisodes de test
    verbose       : affiche la progression

    Retourne
    --------
    rewards    : liste des récompenses par épisode
    steps_list : liste des steps par épisode
    """
    env = gym.make("Taxi-v3")
    rewards, steps_list = [], []

    for ep in range(test_episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            action = int(np.argmax(Q[state]))            # greedy pur
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, test_episodes // 5) == 0:
            print(f"  Test Ep {ep+1:>4}/{test_episodes} | Steps: {steps:>4} | Reward: {total_reward:>7.1f}")

    env.close()
    return rewards, steps_list
