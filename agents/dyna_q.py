"""
Dyna-Q (model-based RL) avec epsilon-greedy.

Dyna-Q combine Q-Learning avec un modèle interne du monde.
À chaque step, l'agent :
  1. Agit dans le vrai environnement (comme Q-Learning)
  2. Note l'expérience dans un modèle (dictionnaire)
  3. Rejoue k expériences passées pour accélérer l'apprentissage

Référence : Sutton & Barto (2018), chapitre 8 — "Planning and Learning"
"""

import numpy as np
import gymnasium as gym

STATE_SIZE  = 500
ACTION_SIZE = 6


def dyna_q(
    env       : gym.Env,
    episodes  : int,
    alpha     : float = 0.15,
    gamma     : float = 0.99,
    eps_start : float = 1.0,
    eps_min   : float = 0.01,
    eps_decay : float = 0.995,
    k         : int   = 5,
    verbose   : bool  = True,
):
    """
    Dyna-Q avec k simulations mentales par step.

    Paramètres
    ----------
    k : int
        Nombre de replays mentaux par step.
        k=0 → équivalent à Q-Learning.
        k=5 → converge ~4x plus vite.

    Retourne
    --------
    Q          : Q-table finale
    rewards    : récompenses par épisode
    steps_list : steps par épisode
    """
    n_states = getattr(env.observation_space, 'n', STATE_SIZE)
    Q = np.zeros((n_states, ACTION_SIZE))

    # modèle interne : model[(s, a)] = (reward, next_state)
    model = {}
    # historique des (s, a) visités pour le replay
    visited = []

    rewards, steps_list = [], []
    eps = eps_start

    for ep in range(episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            # 1. Politique epsilon-greedy
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            # 2. Agir dans le vrai environnement
            ns, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # 3. Mise à jour Q (identique à Q-Learning)
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[ns]) - Q[state, action]
            )

            # 4. Noter dans le modèle
            model[(state, action)] = (reward, ns)
            if (state, action) not in visited:
                visited.append((state, action))

            # 5. Simulations mentales — rejouer k expériences
            if visited and k > 0:
                indices = np.random.randint(0, len(visited), size=k)
                for idx in indices:
                    s_m, a_m = visited[idx]
                    r_m, ns_m = model[(s_m, a_m)]
                    Q[s_m, a_m] += alpha * (
                        r_m + gamma * np.max(Q[ns_m]) - Q[s_m, a_m]
                    )

            state = ns
            total_reward += reward
            steps += 1

        eps = max(eps_min, eps * eps_decay)
        rewards.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  Ep {ep+1:>5}/{episodes} | ε={eps:.3f} "
                  f"| Steps: {steps:>4} | Reward: {total_reward:>7.1f} "
                  f"| Model: {len(model)} entries")

    return Q, rewards, steps_list
