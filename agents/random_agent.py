"""Agent aléatoire servant de baseline simple."""

import gymnasium as gym


def random_agent(env: gym.Env, episodes: int, verbose: bool = True):
    """
    Agent bruteforce : choisit une action aléatoire à chaque step.

    Paramètres
    ----------
    env      : environnement Gymnasium déjà instancié
    episodes : nombre d'épisodes à jouer
    verbose  : affiche la progression

    Retourne
    --------
    rewards    : liste des récompenses totales par épisode
    steps_list : liste du nombre de steps par épisode
    """
    rewards, steps_list = [], []

    for ep in range(episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            action = env.action_space.sample()          # action 100 % aléatoire
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

        if verbose and (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  Ep {ep+1:>5}/{episodes} | Steps: {steps:>4} | Reward: {total_reward:>7.1f}")

    return rewards, steps_list
