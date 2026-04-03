"""Q-Learning classique (TD(0)) avec epsilon-greedy."""

import numpy as np
import gymnasium as gym

STATE_SIZE  = 500
ACTION_SIZE = 6


def q_learning(
    env       : gym.Env,
    episodes  : int,
    alpha     : float = 0.15,      # taux d'apprentissage
    gamma     : float = 0.99,      # facteur de discount
    eps_start : float = 1.0,       # exploration initiale
    eps_min   : float = 0.01,      # exploration minimale
    eps_decay : float = 0.995,     # décroissance d'epsilon
    verbose   : bool  = True,
                  # SharedState (optionnel, threading)
):
    """
    Q-Learning avec politique epsilon-greedy décroissante.

    Retourne
    --------
    Q          : Q-table finale (500 × 6)
    rewards    : récompenses par épisode
    steps_list : steps par épisode
    """
    n_states = getattr(env.observation_space, 'n', STATE_SIZE)
    Q   = np.zeros((n_states, ACTION_SIZE))
    rewards, steps_list = [], []
    eps = eps_start

    for ep in range(episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            # Politique epsilon-greedy
            if np.random.rand() < eps:
                action = env.action_space.sample()      # exploration
            else:
                action = int(np.argmax(Q[state]))       # exploitation

            ns, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Mise à jour de Bellman
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[ns]) - Q[state, action]
            )
            state        = ns
            total_reward += reward
            steps        += 1

        eps = max(eps_min, eps * eps_decay)
        rewards.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  Ep {ep+1:>5}/{episodes} | ε={eps:.3f} | Steps: {steps:>4} | Reward: {total_reward:>7.1f}")

    return Q, rewards, steps_list
