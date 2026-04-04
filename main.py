#!/usr/bin/env python3
"""Taxi Driver RL — point d'entree console avec interface Rich."""

import time
import os
import numpy as np
import gymnasium as gym

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.text import Text
from rich import box

from agents import random_agent, q_learning, sarsa, dyna_q, monte_carlo, dqn_experience_replay
from core import test_policy
from visualization.pygame_vis import visualize_policy
from utils import plot_learning_curves, plot_bar_benchmark, plot_boxplots, generate_report

console = Console()

LOGO = """
[bold #FBBF24]  ████████╗ █████╗ ██╗  ██╗██╗
  ╚══██╔══╝██╔══██╗╚██╗██╔╝██║
     ██║   ███████║ ╚███╔╝ ██║
     ██║   ██╔══██║ ██╔██╗ ██║
     ██║   ██║  ██║██╔╝ ██╗██║
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝[/]
[dim #f97316]       D R I V E R  ·  R L[/]
"""

ALGO_MAP = {
    "r":    ("Random",      random_agent),
    "q":    ("Q-Learning",  q_learning),
    "s":    ("SARSA",       sarsa),
    "d":    ("Dyna-Q",      dyna_q),
    "mc":   ("Monte Carlo", monte_carlo),
    "dqn":  ("DQN-ER",      dqn_experience_replay),
}


def show_banner():
    console.print(LOGO)
    stats = Table(show_header=False, box=box.ROUNDED, border_style="#FBBF24",
                  padding=(0, 2))
    stats.add_column(justify="center")
    stats.add_row("[bold #22d3ee]6[/] Algorithms  ·  [bold #22d3ee]3[/] Environments  ·  "
                  "[bold #22d3ee]14,400[/] States  ·  [bold #22d3ee]4[/] Reward Modes")
    console.print(stats, justify="center")
    console.print()


def show_algo_table():
    table = Table(title="[bold]Algorithmes disponibles[/]", box=box.SIMPLE_HEAVY,
                  border_style="dim", title_style="bold #FBBF24")
    table.add_column("Clé", style="bold #22d3ee", justify="center")
    table.add_column("Algorithme", style="bold white")
    table.add_column("Type", style="dim")
    table.add_column("Description", style="dim")

    table.add_row("r",   "Random",      "—",           "Baseline brute-force")
    table.add_row("q",   "Q-Learning",  "Off-policy",  "TD(0) avec max(Q[s'])")
    table.add_row("s",   "SARSA",       "On-policy",   "TD(0) prudent avec Q[s',a']")
    table.add_row("d",   "Dyna-Q",      "Model-based", "k replays mentaux par step")
    table.add_row("mc",  "Monte Carlo", "On-policy",   "First-visit, episodes complets")
    table.add_row("dqn", "DQN-ER",      "Off-policy",  "Experience replay + target network")
    console.print(table)
    console.print()


def run_training(algo_key, label, algo_fn, env, train_ep, alpha, gamma,
                 eps_start, eps_decay, dyna_k=5):
    """Run training for a single algorithm."""
    start_t = time.time()

    if algo_key == "r":
        rew, steps = algo_fn(env, train_ep, verbose=False)
        Q = None
    elif algo_key in ("q", "s"):
        Q, rew, steps = algo_fn(env, train_ep, alpha=alpha, gamma=gamma,
                                eps_start=eps_start, eps_decay=eps_decay, verbose=False)
    elif algo_key == "d":
        Q, rew, steps = algo_fn(env, train_ep, alpha=alpha, gamma=gamma,
                                eps_start=eps_start, eps_decay=eps_decay,
                                k=dyna_k, verbose=False)
    elif algo_key == "mc":
        Q, rew, steps = algo_fn(env, train_ep, gamma=gamma,
                                eps_start=eps_start, eps_decay=eps_decay, verbose=False)
    elif algo_key == "dqn":
        Q, rew, steps = algo_fn(env, train_ep, gamma=gamma,
                                eps_start=eps_start, eps_decay=eps_decay, verbose=False)
    else:
        Q, rew, steps = None, [], []

    elapsed = time.time() - start_t
    return Q, rew, steps, elapsed


def user_mode():
    console.print(Panel("[bold #FBBF24]MODE UTILISATEUR[/] — Parametres personnalises",
                        border_style="#FBBF24"))

    train_ep  = IntPrompt.ask("  Episodes d'entrainement", default=2000)
    test_ep   = IntPrompt.ask("  Episodes de test", default=200)
    alpha     = FloatPrompt.ask("  Alpha (learning rate)", default=0.15)
    gamma     = FloatPrompt.ask("  Gamma (discount)", default=0.99)
    eps_start = FloatPrompt.ask("  Epsilon start", default=1.0)
    eps_decay = FloatPrompt.ask("  Epsilon decay", default=0.995)
    dyna_k    = IntPrompt.ask("  Dyna-Q k (replays mentaux)", default=5)

    console.print()
    show_algo_table()

    algos_raw = Prompt.ask("  Algorithmes a lancer (ex: q,s,d)", default="q,s,d")
    algos = [a.strip() for a in algos_raw.split(",") if a.strip() in ALGO_MAP]
    if not algos:
        algos = ["q"]

    all_results, test_results, trained_tables = {}, {}, {}
    env = gym.make("Taxi-v3")

    console.print()
    with Progress(
        SpinnerColumn(style="#FBBF24"),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(bar_width=30, style="#334155", complete_style="#FBBF24"),
        TextColumn("[dim]{task.fields[status]}[/]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        for algo_key in algos:
            label, algo_fn = ALGO_MAP[algo_key]
            task = progress.add_task(f"  {label}", total=100, status="training...")

            Q, rew, steps, elapsed = run_training(
                algo_key, label, algo_fn, env, train_ep,
                alpha, gamma, eps_start, eps_decay, dyna_k
            )
            progress.update(task, completed=70, status=f"trained in {elapsed:.1f}s")

            all_results[label] = (rew, steps)
            trained_tables[label] = Q

            # test
            if Q is not None:
                tr, ts = test_policy(Q, test_ep, verbose=False)
            else:
                tr, ts = random_agent(env, test_ep, verbose=False)
            test_results[label] = (tr, ts)

            progress.update(task, completed=100,
                            status=f"[green]done[/] · {elapsed:.1f}s · reward: {np.mean(tr):.1f}")

    env.close()

    # results table
    console.print()
    results_table = Table(title="[bold #FBBF24]Resultats finaux[/]", box=box.DOUBLE_EDGE,
                          border_style="#FBBF24")
    results_table.add_column("Algorithme", style="bold white")
    results_table.add_column("Train (100 dern.)", justify="right", style="#22d3ee")
    results_table.add_column("Test reward", justify="right", style="green")
    results_table.add_column("Test steps", justify="right", style="#f97316")

    for label in all_results:
        r_tr = f"{np.mean(all_results[label][0][-100:]):.2f}"
        r_te = f"{np.mean(test_results[label][0]):.2f}"
        s_te = f"{np.mean(test_results[label][1]):.2f}"
        results_table.add_row(label, r_tr, r_te, s_te)

    console.print(results_table)

    # best algo
    best = min(test_results, key=lambda k: np.mean(test_results[k][1]))
    best_steps = np.mean(test_results[best][1])
    best_reward = np.mean(test_results[best][0])
    console.print()
    console.print(Panel(
        f"[bold green]{best}[/] est le meilleur algorithme\n"
        f"[bold]{best_steps:.1f}[/] steps · [bold]{best_reward:.1f}[/] reward · "
        f"[bold]{200 / max(1, best_steps):.0f}x[/] plus rapide qu'un agent aleatoire",
        title="[bold #FBBF24]Meilleur algorithme[/]",
        border_style="green"
    ))

    # plots & report
    console.print()
    with console.status("[bold #FBBF24]Generation des graphiques...[/]"):
        plot_learning_curves(all_results)
        plot_bar_benchmark(all_results)
        plot_boxplots(test_results)
        generate_report(
            all_results, test_results,
            mode="User",
            params={"Episodes": train_ep, "Test episodes": test_ep,
                    "Alpha": alpha, "Gamma": gamma,
                    "Epsilon start": eps_start, "Epsilon decay": eps_decay,
                    "Dyna-Q k": dyna_k, "Algorithmes": algos_raw},
        )
    console.print("[green]  ✓ Graphiques et rapport generes[/]")

    # list generated files
    outputs = ["learning_curves.png", "benchmark_bar.png", "boxplot_test.png", "report.txt"]
    for f in outputs:
        if os.path.exists(f):
            console.print(f"    [dim]→ {f}[/]")

    choose_and_visualize(trained_tables)


def time_limited_mode():
    console.print(Panel("[bold #FBBF24]MODE TEMPS LIMITE[/] — Parametres optimises",
                        border_style="#FBBF24"))

    time_limit = IntPrompt.ask("  Duree d'entrainement (secondes)", default=10)
    test_ep    = IntPrompt.ask("  Episodes de test", default=200)

    alpha, gamma, eps_start, eps_decay = 0.15, 0.99, 1.0, 0.99

    console.print()
    console.print(f"  [dim]Parametres Q-Learning : α={alpha}  γ={gamma}  ε={eps_start}  decay={eps_decay}[/]")

    env = gym.make("Taxi-v3")
    Q = np.zeros((500, 6))
    rewards, steps_list = [], []
    eps_cur = eps_start
    start = time.time()
    ep = 0

    with Progress(
        SpinnerColumn(style="#FBBF24"),
        TextColumn("[bold]Q-Learning[/]"),
        BarColumn(bar_width=30, style="#334155", complete_style="#FBBF24"),
        TextColumn("[dim]{task.fields[status]}[/]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("training", total=time_limit, status="training...")

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

            elapsed = time.time() - start
            progress.update(task, completed=min(elapsed, time_limit),
                            status=f"{ep} episodes · reward: {np.mean(rewards[-100:]):.1f}")

    env.close()
    elapsed = time.time() - start
    console.print(f"\n  [green]✓ {ep} episodes en {elapsed:.1f}s[/]")

    # test
    console.print()
    with console.status("[bold #FBBF24]Test greedy...[/]"):
        test_rew, test_steps = test_policy(Q, test_ep, verbose=False)

    console.print(f"  [green]✓ Test : reward = {np.mean(test_rew):.2f} · steps = {np.mean(test_steps):.2f}[/]")

    algo_lbl = f"Q-Learning ({ep}ep, {elapsed:.0f}s)"
    all_results = {algo_lbl: (rewards, steps_list)}
    test_results = {algo_lbl: (test_rew, test_steps)}

    with console.status("[bold #FBBF24]Generation des graphiques...[/]"):
        plot_learning_curves(all_results, output="time_limited_curves.png")
        plot_boxplots(test_results, output="time_limited_boxplot.png")
        generate_report(
            all_results, test_results,
            mode="Time-Limited",
            params={"Duree": f"{time_limit}s", "Episodes accomplis": ep,
                    "Alpha": alpha, "Gamma": gamma,
                    "Epsilon start": eps_start, "Epsilon decay": eps_decay},
            output_path="report_time_limited.txt"
        )
    console.print("[green]  ✓ Graphiques et rapport generes[/]")

    choose_and_visualize({algo_lbl: Q})


def choose_and_visualize(trained_tables: dict):
    tables = {k: v for k, v in trained_tables.items() if v is not None}
    if not tables:
        return

    console.print()
    console.print(Panel("[bold #FBBF24]Visualisation[/] — Regarder l'agent jouer",
                        border_style="#FBBF24"))

    labels = list(tables.keys())
    for i, lbl in enumerate(labels, 1):
        console.print(f"    [bold #22d3ee]{i}.[/] {lbl}")
    console.print(f"    [dim]0. Ne pas visualiser[/]")
    console.print()

    idx = IntPrompt.ask("  Votre choix", default=1)
    if idx == 0 or idx > len(labels):
        return

    chosen = labels[idx - 1]
    Q = tables[chosen]
    n_ep = IntPrompt.ask("  Episodes a visualiser", default=3)
    delay = FloatPrompt.ask("  Vitesse (delai en sec)", default=0.4)

    console.print(f"\n  [#FBBF24]▸ Lancement de la visualisation pour {chosen}...[/]\n")
    visualize_policy(Q, n_episodes=n_ep, label=chosen, delay=delay)


def main():
    console.clear()
    show_banner()

    console.print("  [bold white][1][/] Mode Utilisateur  [dim]— parametres configurables[/]")
    console.print("  [bold white][2][/] Mode Temps Limite  [dim]— parametres optimises + limite de temps[/]")
    console.print()

    choice = IntPrompt.ask("  Votre choix", default=1)
    console.print()

    if choice == 2:
        time_limited_mode()
    else:
        user_mode()

    console.print()
    console.print(Panel("[bold green]Programme termine[/]", border_style="green"))


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Arrete par l'utilisateur.[/]")
