"""Génération de rapport texte pour les benchmarks RL."""

import numpy as np


def generate_report(
    all_results : dict,
    test_results: dict = None,
    mode        : str  = "User",
    params      : dict = None,
    output_path : str  = "report.txt",
) -> None:
    """
    Écrit un rapport complet dans output_path.

    Paramètres
    ----------
    all_results  : {algo: (rewards_train, steps_train)}
    test_results : {algo: (rewards_test, steps_test)}
    mode         : "User" ou "Time-Limited"
    params       : dictionnaire des hyperparamètres utilisés
    output_path  : chemin du fichier de sortie
    """
    sep = "=" * 60

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"  TAXI DRIVER — RAPPORT DE BENCHMARK\n")
        f.write(f"  Mode : {mode}\n")
        f.write(f"{sep}\n\n")

        if params:
            f.write("Hyperparamètres :\n")
            for k, v in params.items():
                f.write(f"  {k:<24}: {v}\n")
            f.write("\n")

        f.write("── Résultats d'entraînement ──────────────────────────────\n\n")
        for algo, (rew, steps) in all_results.items():
            f.write(f"  {algo}\n")
            f.write(f"    Récompense moy. (tous)    : {np.mean(rew):.2f}\n")
            f.write(f"    Récompense moy. (100 dern.): {np.mean(rew[-100:]):.2f}\n")
            f.write(f"    Meilleure récompense       : {max(rew):.2f}\n")
            f.write(f"    Steps moy. (tous)          : {np.mean(steps):.2f}\n")
            f.write(f"    Steps moy. (100 dern.)     : {np.mean(steps[-100:]):.2f}\n\n")

        if test_results:
            f.write("── Résultats de test (politique greedy) ──────────────────\n\n")
            for algo, (rew, steps) in test_results.items():
                f.write(f"  {algo}\n")
                f.write(f"    Récompense moy. : {np.mean(rew):.2f}\n")
                f.write(f"    Steps moy.      : {np.mean(steps):.2f}\n\n")

        f.write(f"\nFichiers générés :\n")
        for fname in ["learning_curves.png", "benchmark_bar.png",
                      "boxplot_test.png", "time_limited_curves.png",
                      "time_limited_boxplot.png"]:
            f.write(f"  - {fname}\n")

    print(f"  📄 {output_path}")
