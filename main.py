#!/usr/bin/env python3
"""Taxi Driver RL — point d'entrée pour l'application Taxi-v3."""

import time
import os
import numpy as np
import gymnasium as gym

from agents        import random_agent, q_learning, monte_carlo, dqn_experience_replay
from core          import test_policy
from visualization.pygame_vis import visualize_policy
from utils         import (safe_input, print_separator,
                           plot_learning_curves, plot_bar_benchmark,
                           plot_boxplots, generate_report)

ALGO_LABELS = {
    "r"  : "Random",
    "q"  : "Q-Learning",
    "mc" : "Monte Carlo",
    "dqn": "DQN-ER",
}


# Mode utilisateur

def user_mode() -> None:
    print_separator("MODE UTILISATEUR — Paramètres personnalisés")

    train_ep  = safe_input("  Épisodes d'entraînement  (défaut=3000)  : ", 3000,  int)
    test_ep   = safe_input("  Épisodes de test         (défaut=200)   : ",  200,  int)
    alpha     = safe_input("  Learning rate alpha      (défaut=0.15)  : ", 0.15, float)
    gamma     = safe_input("  Discount gamma           (défaut=0.99)  : ", 0.99, float)
    eps_start = safe_input("  Epsilon initial          (défaut=1.0)   : ", 1.0,  float)
    eps_decay = safe_input("  Epsilon decay            (défaut=0.995) : ", 0.995,float)

    print()
    print("  Algorithmes disponibles : r  q  mc  dqn")
    algos_raw = safe_input("  Choisir (ex: r,q,mc,dqn / défaut=q) : ", "q", str)
    algos     = [a.strip() for a in algos_raw.split(",") if a.strip() in ALGO_LABELS]
    if not algos:
        algos = ["q"]

    all_results, test_results = {}, {}
    trained_tables = {}

    env = gym.make("Taxi-v3")

    for algo in algos:
        label = ALGO_LABELS[algo]
        print_separator(f"Entraînement : {label}")
        
        start_t = time.time()
        
        if algo == "r":
            rew, steps = random_agent(env, train_ep, verbose=True)
            Q = None
        elif algo == "q":
            Q, rew, steps = q_learning(
                env, train_ep, alpha=alpha, gamma=gamma,
                eps_start=eps_start, eps_decay=eps_decay, verbose=True
            )
        elif algo == "mc":
            Q, rew, steps = monte_carlo(
                env, train_ep, gamma=gamma,
                eps_start=eps_start, eps_decay=eps_decay, verbose=True
            )
        elif algo == "dqn":
            Q, rew, steps = dqn_experience_replay(
                env, train_ep, gamma=gamma,
                eps_start=eps_start, eps_decay=eps_decay, verbose=True
            )
            
        elapsed = time.time() - start_t
        print(f"  [OK] Entraînement terminé en {elapsed:.2f}s")
        
        all_results[label] = (rew, steps)
        trained_tables[label] = Q
        
        if Q is not None:
            print(f"\n  [TEST] Test greedy ({test_ep} épisodes)...")
            tr, ts = test_policy(Q, test_ep, verbose=True)
            test_results[label] = (tr, ts)
            print(f"      -> Récompense moy : {np.mean(tr):.2f} | Steps moy : {np.mean(ts):.2f}")
        else:
            # Pour random, on teste en utilisant le random_agent sur l'env test_ep fois (pour avoir les stats)
            tr, ts = random_agent(env, test_ep, verbose=False)
            test_results[label] = (tr, ts)

    env.close()

    # ── Résumé final ──────────────────────────────────────────
    print_separator("RÉSULTATS FINAUX")
    print(f"  {'Algorithme':<18} {'Train (100 dern.)':<22} {'Test reward':<16} {'Test steps'}")
    print_separator()
    for algo_lbl in all_results:
        r_tr = f"{np.mean(all_results[algo_lbl][0][-100:]):.2f}"
        r_te = f"{np.mean(test_results[algo_lbl][0]):.2f}"
        s_te = f"{np.mean(test_results[algo_lbl][1]):.2f}"
        print(f"  {algo_lbl:<18} {r_tr:<22} {r_te:<16} {s_te}")

    # ── Graphiques & rapport ──────────────────────────────────
    print_separator("Graphiques & Rapport")
    plot_learning_curves(all_results)
    plot_bar_benchmark(all_results)
    plot_boxplots(test_results)
    generate_report(
        all_results, test_results,
        mode   = "User",
        params = {"Episodes": train_ep, "Test episodes": test_ep,
                  "Alpha": alpha, "Gamma": gamma,
                  "Epsilon start": eps_start, "Epsilon decay": eps_decay,
                  "Algorithmes": algos_raw},
    )

    choose_and_visualize(trained_tables)


# Mode temps limité

def time_limited_mode() -> None:
    print_separator("MODE TEMPS LIMITÉ — Paramètres optimisés")

    time_limit = safe_input("  Durée d'entraînement (secondes) (défaut=10) : ", 10,  int)
    test_ep    = safe_input("  Épisodes de test                (défaut=200): ", 200, int)

    alpha, gamma, eps_start, eps_decay = 0.15, 0.99, 1.0, 0.99
    print(f"\n  Paramètres Q-Learning : α={alpha}  γ={gamma}  ε={eps_start}  decay={eps_decay}")
    print(f"  [WAIT] Entraînement en cours pour {time_limit}s...")

    env = gym.make("Taxi-v3")
    Q = np.zeros((500, 6))
    rewards, steps_list = [], []
    eps_cur = eps_start
    start = time.time()
    ep = 0

    while time.time() - start < time_limit:
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0
        
        while not done:
            action = env.action_space.sample() if np.random.rand() < eps_cur else int(np.argmax(Q[state]))
            ns, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            Q[state, action] += alpha * (reward + gamma * np.max(Q[ns]) - Q[state, action])
            state = ns
            total_reward += reward
            steps += 1
            
        eps_cur = max(0.01, eps_cur * eps_decay)
        rewards.append(total_reward)
        steps_list.append(steps)
        ep += 1
        
        if ep % 500 == 0:
            print(f"  Ep {ep:>6} | {time.time()-start:>5.1f}s/{time_limit}s | Récomp moy : {np.mean(rewards[-100:]):.2f}")

    env.close()
    elapsed = time.time() - start
    print(f"\n  OK Terminé : {ep} épisodes en {elapsed:.1f}s")

    print_separator("Test greedy")
    test_rew, test_steps = test_policy(Q, test_ep, verbose=True)
    print(f"\n  Récompense moy. : {np.mean(test_rew):.2f} | Steps moy. : {np.mean(test_steps):.2f}")

    algo_lbl = f"Q-Learning ({ep}ep, {elapsed:.0f}s)"
    all_results = {algo_lbl: (rewards, steps_list)}
    test_results = {algo_lbl: (test_rew, test_steps)}

    print_separator("Graphiques & Rapport")
    plot_learning_curves(all_results, output="time_limited_curves.png")
    plot_boxplots(test_results,       output="time_limited_boxplot.png")
    generate_report(
        all_results, test_results,
        mode   = "Time-Limited",
        params = {"Durée": f"{time_limit}s", "Episodes accomplis": ep,
                  "Alpha": alpha, "Gamma": gamma,
                  "Epsilon start": eps_start, "Epsilon decay": eps_decay},
        output_path="report_time_limited.txt"
    )

    choose_and_visualize({algo_lbl: Q})


# Choix de la visualisation post-entraînement

def choose_and_visualize(trained_tables: dict) -> None:
    if not trained_tables:
        print("  Aucune politique entraînée disponible pour la visualisation.")
        return

    print_separator("Visualisation post-entraînement")
    labels = list(trained_tables.keys())

    print("  Algorithmes disponibles :")
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}. {lbl}")
    print(f"    0. Ne pas visualiser")
    print()

    idx = safe_input(f"  Votre choix (1-{len(labels)}, défaut=1) : ", 1, int)
    if idx == 0 or idx > len(labels):
        print("  Visualisation ignorée.")
        return

    chosen_label = labels[idx - 1]
    Q = trained_tables[chosen_label]

    n_ep  = safe_input("  Nombre d'épisodes à visualiser (défaut=3) : ", 3, int)
    delay = safe_input("  Vitesse (délai entre steps en sec, défaut=0.4) : ", 0.4, float)

    visualize_policy(Q, n_episodes=n_ep, label=chosen_label, delay=delay)


# Main

def main() -> None:
    print_separator("TAXI DRIVER — Reinforcement Learning")
    print("""
  Algorithmes : Random | Q-Learning | Monte Carlo | DQN-ER

  1. Mode Utilisateur  — paramètres entièrement configurables
  2. Mode Temps Limité — paramètres optimisés + limite de temps
""")
    choice = safe_input("  Votre choix (1 ou 2, défaut=1) : ", 1, int)
    print()

    if choice == 2:
        time_limited_mode()
    else:
        user_mode()

    print_separator("Programme terminé")
    outputs = [
        "report.txt", "report_time_limited.txt", "learning_curves.png", "benchmark_bar.png",
        "boxplot_test.png", "time_limited_curves.png", "time_limited_boxplot.png",
    ]
    for f in outputs:
        if os.path.exists(f):
            print(f"    OK {f}")
    print()

if __name__ == "__main__":
    import logging
    # Suppress verbose sys tracebacks on window close sometimes
    logging.getLogger().setLevel(logging.ERROR)
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêté par l'utilisateur.")
