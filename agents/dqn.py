"""DQN tabulaire avec experience replay et target network."""

import numpy as np
import random
import gymnasium as gym
from collections import deque

STATE_SIZE  = 500
ACTION_SIZE = 6


def dqn_experience_replay(
    env          : gym.Env,
    episodes     : int,
    gamma        : float = 0.99,
    eps_start    : float = 1.0,
    eps_min      : float = 0.05,
    eps_decay    : float = 0.995,
    batch_size   : int   = 64,
    memory_size  : int   = 5000,
    lr           : float = 0.01,     # learning rate de la mise à jour tabulaire
    target_update: int   = 50,       # fréquence (en épisodes) de copie du target network
    verbose      : bool  = True,
    shared       = None,             # SharedState (optionnel, threading)
):
    """
    DQN tabulaire avec experience replay et target network.

    Paramètres
    ----------
    batch_size    : taille du mini-batch sampléé du replay buffer
    memory_size   : capacité max du replay buffer
    lr            : taux d'apprentissage pour la mise à jour tabulaire
    target_update : copie du target network tous les N épisodes

    Retourne
    --------
    Q          : Q-table principale finale (500 × 6)
    rewards    : récompenses par épisode
    steps_list : steps par épisode
    """
    Q        = np.zeros((STATE_SIZE, ACTION_SIZE))   # réseau principal
    Q_target = np.zeros((STATE_SIZE, ACTION_SIZE))   # réseau cible (frozen)
    memory   = deque(maxlen=memory_size)             # replay buffer
    rewards, steps_list = [], []
    eps = eps_start

    for ep in range(episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            # Politique epsilon-greedy sur Q principal
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            ns, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Stocker la transition dans le replay buffer
            memory.append((state, action, reward, ns, done))
            state        = ns
            total_reward += reward
            steps        += 1

            # Experience replay
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                for s, a, r, s2, d in batch:
                    # Target calculé avec Q_target (frozen)
                    target = r if d else r + gamma * np.max(Q_target[s2])
                    # Mise à jour incrémentale de Q principal
                    Q[s, a] += lr * (target - Q[s, a])

        eps = max(eps_min, eps * eps_decay)

        # Mise à jour du target network
        if (ep + 1) % target_update == 0:
            Q_target = Q.copy()

        rewards.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  Ep {ep+1:>5}/{episodes} | ε={eps:.3f} | Steps: {steps:>4} | Reward: {total_reward:>7.1f}")

    return Q, rewards, steps_list
