"""Monte Carlo First-Visit on-policy avec politique epsilon-greedy."""

import numpy as np
import gymnasium as gym

STATE_SIZE  = 500
ACTION_SIZE = 6


def monte_carlo(
    env       : gym.Env,
    episodes  : int,
    gamma     : float = 0.99,
    eps_start : float = 1.0,
    eps_min   : float = 0.01,
    eps_decay : float = 0.995,
    verbose   : bool  = True,
                 # SharedState (optionnel, threading)
):
    """
    Monte Carlo First-Visit avec politique epsilon-greedy.

    Retourne
    --------
    Q             : Q-table finale (500 × 6)
    rewards_hist  : récompenses par épisode
    steps_list    : steps par épisode
    """
    Q      = np.zeros((STATE_SIZE, ACTION_SIZE))
    counts = np.zeros((STATE_SIZE, ACTION_SIZE))   # nb de visites par (s,a)
    rewards_hist, steps_list = [], []
    eps = eps_start

    for ep in range(episodes):
        state, _ = env.reset()
        done, episode, total_reward, steps = False, [], 0, 0

        # Génération de l'épisode complet
        while not done:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            ns, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode.append((state, action, reward))
            state        = ns
            total_reward += reward
            steps        += 1

        # Mise à jour First-Visit en arrière
        visited, G = set(), 0.0
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                counts[s, a] += 1
                # Moyenne incrémentale (évite de stocker tous les retours)
                Q[s, a] += (G - Q[s, a]) / counts[s, a]

        eps = max(eps_min, eps * eps_decay)
        rewards_hist.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  Ep {ep+1:>5}/{episodes} | ε={eps:.3f} | Steps: {steps:>4} | Reward: {total_reward:>7.1f}")

    return Q, rewards_hist, steps_list
