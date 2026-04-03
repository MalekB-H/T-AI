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
# 1b. Q-Table Heatmap
# ──────────────────────────────────────────────────────────────

def plot_qtable_heatmap(
    trained_tables: dict,
    output: str = "qtable_heatmap.png",
) -> None:
    """
    Heatmap 5x5 showing max Q-value per grid cell for each algorithm.
    Green = agent knows what to do, Red = agent is unsure.

    trained_tables : {label: Q_table (ndarray)}
    """
    # filter out None (Random agent) and DQN (not a Q-table)
    tables = {k: v for k, v in trained_tables.items()
              if v is not None and hasattr(v, 'shape') and len(v.shape) == 2}
    if not tables:
        return

    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             facecolor="#080b14")
    fig.suptitle("Q-Table Heatmap — Best Q-value per cell",
                 fontsize=14, fontweight="bold", color="#e2e8f0")

    if n == 1:
        axes = [axes]

    loc_names = {(0, 0): "R", (0, 4): "G", (4, 0): "Y", (4, 3): "B"}

    for ax, (label, Q) in zip(axes, tables.items()):
        grid = np.zeros((5, 5))
        n_states = Q.shape[0]

        for row in range(5):
            for col in range(5):
                max_q = -np.inf
                # iterate over all passenger/dest combos for this cell
                if n_states == 500:
                    # Taxi-v3 encoding: (row*5+col)*5*4 + pass*4 + dest
                    for p in range(5):
                        for d in range(4):
                            s = ((row * 5 + col) * 5 + p) * 4 + d
                            val = np.max(Q[s])
                            if val > max_q:
                                max_q = val
                elif n_states == 14400:
                    # MultiPassenger encoding
                    for p1 in range(6):
                        for d1 in range(4):
                            for p2 in range(6):
                                for d2 in range(4):
                                    s = (row * 5 * 6 * 4 * 6 * 4 +
                                         col * 6 * 4 * 6 * 4 +
                                         p1 * 4 * 6 * 4 +
                                         d1 * 6 * 4 +
                                         p2 * 4 + d2)
                                    if s < n_states:
                                        val = np.max(Q[s])
                                        if val > max_q:
                                            max_q = val
                else:
                    # fallback: same as Taxi-v3
                    for p in range(5):
                        for d in range(4):
                            s = ((row * 5 + col) * 5 + p) * 4 + d
                            if s < n_states:
                                val = np.max(Q[s])
                                if val > max_q:
                                    max_q = val
                grid[row][col] = max_q

        im = ax.imshow(grid, cmap="RdYlGn", interpolation="nearest")
        ax.set_facecolor("#080b14")
        ax.set_title(label, fontsize=13, fontweight="bold", color="#FBBF24")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.tick_params(colors="#94a3b8")

        # annotate cells with values and location names
        for row in range(5):
            for col in range(5):
                val = grid[row][col]
                loc = loc_names.get((row, col), "")
                text = f"{val:.1f}"
                if loc:
                    text = f"{loc}\n{val:.1f}"
                color = "white" if val < (grid.max() + grid.min()) / 2 else "black"
                ax.text(col, row, text, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors="#94a3b8")

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
