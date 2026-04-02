"""Génération de graphiques matplotlib pour les résultats RL."""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas de fenêtre bloquante
import matplotlib.pyplot as plt

# Palette de couleurs par algorithme
_COLORS = {
    "Random"    : "#e74c3c",
    "Q-Learning": "#3498db",
    "Monte Carlo": "#2ecc71",
    "DQN-ER"    : "#f39c12",
}


def _get_color(label: str) -> str:
    for key, color in _COLORS.items():
        if key in label:
            return color
    return "#9b59b6"


def _moving_average(data: list, window: int = 50) -> np.ndarray:
    w = min(window, max(1, len(data) // 5))
    return np.convolve(data, np.ones(w) / w, mode="valid")


# ──────────────────────────────────────────────────────────────
# 1. Courbes d'apprentissage
# ──────────────────────────────────────────────────────────────

def plot_learning_curves(
    all_results: dict,
    output     : str = "learning_curves.png",
) -> None:
    """
    Courbes de récompenses et steps par épisode (moyenne glissante)
    pour tous les algorithmes.

    all_results : {label: (rewards_list, steps_list)}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage — Taxi-v3", fontsize=14, fontweight="bold")

    for label, (rew, steps) in all_results.items():
        c = _get_color(label)
        ax1.plot(_moving_average(rew),   label=label, color=c, linewidth=1.5)
        ax2.plot(_moving_average(steps), label=label, color=c, linewidth=1.5)

    ax1.set_title("Récompense (moyenne glissante)")
    ax1.set_xlabel("Épisodes")
    ax1.set_ylabel("Récompense")
    ax1.legend()

    ax2.set_title("Steps (moyenne glissante)")
    ax2.set_xlabel("Épisodes")
    ax2.set_ylabel("Steps")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close()
    print(f"  📊 {output}")


# ──────────────────────────────────────────────────────────────
# 2. Benchmark en barres (100 derniers épisodes)
# ──────────────────────────────────────────────────────────────

def plot_bar_benchmark(
    all_results: dict,
    output     : str = "benchmark_bar.png",
) -> None:
    """
    Barres comparant la performance finale (100 derniers épisodes)
    de chaque algorithme.
    """
    labels = list(all_results.keys())
    avg_r  = [np.mean(all_results[l][0][-100:]) for l in labels]
    avg_s  = [np.mean(all_results[l][1][-100:]) for l in labels]
    colors = [_get_color(l) for l in labels]
    x      = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Benchmark final (100 derniers épisodes)", fontsize=13, fontweight="bold")

    ax1.bar(x, avg_r, color=colors, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title("Récompense moyenne")
    ax1.set_ylabel("Récompense")

    ax2.bar(x, avg_s, color=colors, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_title("Steps moyens")
    ax2.set_ylabel("Steps")

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close()
    print(f"  📊 {output}")


# ──────────────────────────────────────────────────────────────
# 3. Boxplots de test
# ──────────────────────────────────────────────────────────────

def plot_boxplots(
    test_results: dict,
    output      : str = "boxplot_test.png",
) -> None:
    """
    Distribution des performances sur les épisodes de test
    (politique greedy). Montre la variance de chaque algorithme.

    test_results : {label: (rewards_list, steps_list)}
    """
    if not test_results:
        return

    labels     = list(test_results.keys())
    rew_data   = [test_results[l][0] for l in labels]
    steps_data = [test_results[l][1] for l in labels]
    colors     = [_get_color(l) for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution des performances — test greedy", fontsize=13, fontweight="bold")

    bp1 = ax1.boxplot(rew_data,   patch_artist=True, tick_labels=labels)
    bp2 = ax2.boxplot(steps_data, patch_artist=True, tick_labels=labels)

    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)

    ax1.set_title("Récompense")
    ax1.tick_params(axis="x", rotation=15)
    ax2.set_title("Steps")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close()
    print(f"  📊 {output}")
