import os
import time
import threading
import gymnasium as gym
import streamlit as st

from agents import random_agent, q_learning, monte_carlo, dqn_experience_replay
from core.tester import test_policy
from utils.plots import plot_learning_curves, plot_bar_benchmark, plot_boxplots
from utils.report import generate_report
from visualization.pygame_vis import visualize_policy

STORAGE_DIR = "streamlit_outputs"
os.makedirs(STORAGE_DIR, exist_ok=True)

ALGO_LABELS = {
    "Random": random_agent,
    "Q-Learning": q_learning,
    "Monte Carlo": monte_carlo,
    "DQN-ER": dqn_experience_replay,
}


def run_algorithm(label, algo, train_episodes, alpha, gamma, eps_start, eps_decay, batch_size, memory_size, lr):
    env = gym.make("Taxi-v3")
    if label == "Random":
        rewards, steps = algo(env, train_episodes, verbose=False)
        Q = None
    elif label == "Q-Learning":
        Q, rewards, steps = algo(
            env, train_episodes,
            alpha=alpha, gamma=gamma,
            eps_start=eps_start, eps_decay=eps_decay,
            verbose=False,
        )
    elif label == "Monte Carlo":
        Q, rewards, steps = algo(
            env, train_episodes,
            gamma=gamma,
            eps_start=eps_start, eps_decay=eps_decay,
            verbose=False,
        )
    else:  # DQN-ER
        Q, rewards, steps = algo(
            env, train_episodes,
            gamma=gamma,
            eps_start=eps_start, eps_decay=eps_decay,
            batch_size=batch_size,
            memory_size=memory_size,
            lr=lr,
            verbose=False,
        )
    env.close()
    return Q, rewards, steps


def evaluate_policy(label, Q, test_episodes):
    if Q is None:
        env = gym.make("Taxi-v3")
        rewards, steps = random_agent(env, test_episodes, verbose=False)
        env.close()
    else:
        rewards, steps = test_policy(Q, test_episodes, verbose=False)
    return rewards, steps


def plot_all_results(all_results, test_results):
    learning_path = os.path.join(STORAGE_DIR, "learning_curves.png")
    benchmark_path = os.path.join(STORAGE_DIR, "benchmark_bar.png")
    boxplot_path = os.path.join(STORAGE_DIR, "boxplot_test.png")

    plot_learning_curves(all_results, output=learning_path)
    plot_bar_benchmark(all_results, output=benchmark_path)
    plot_boxplots(test_results, output=boxplot_path)

    return learning_path, benchmark_path, boxplot_path


def main():
    st.set_page_config(page_title="Taxi RL Dashboard", layout="wide")
    st.title("Taxi Driver RL — Interface interactive")
    st.markdown(
        "Ce tableau de bord permet de comparer plusieurs méthodes de Reinforcement Learning sur `Taxi-v3`."
    )

    with st.sidebar:
        st.header("Paramètres")
        selected_algos = st.multiselect(
            "Algorithmes",
            options=list(ALGO_LABELS.keys()),
            default=["Q-Learning", "Monte Carlo", "DQN-ER"],
        )
        train_episodes = st.number_input("Épisodes d'entraînement", min_value=100, max_value=10000, value=2000, step=100)
        test_episodes = st.number_input("Épisodes de test", min_value=20, max_value=1000, value=200, step=10)
        alpha = st.slider("Alpha", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
        gamma = st.slider("Gamma", min_value=0.80, max_value=0.999, value=0.99, step=0.01)
        eps_start = st.slider("Epsilon initial", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        eps_decay = st.slider("Epsilon decay", min_value=0.90, max_value=0.999, value=0.995, step=0.001)
        st.markdown("---")
        st.subheader("Paramètres DQN")
        batch_size = st.number_input("Batch size", min_value=16, max_value=256, value=64, step=16)
        memory_size = st.number_input("Memory size", min_value=1000, max_value=20000, value=5000, step=500)
        lr = st.slider("Learning rate DQN", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        st.markdown("---")
        st.write("Les graphiques et le rapport sont générés automatiquement après exécution.")

    if not selected_algos:
        st.warning("Sélectionne au moins un algorithme pour lancer la simulation.")
        return

    st.session_state.setdefault("trained_tables", {})
    st.session_state.setdefault("run_done", False)

    run_button = st.button("Lancer la simulation")

    if run_button:
        all_results = {}
        test_results = {}
        summary = []
        report_params = {
            "Episodes": train_episodes,
            "Test episodes": test_episodes,
            "Alpha": alpha,
            "Gamma": gamma,
            "Epsilon start": eps_start,
            "Epsilon decay": eps_decay,
            "Algorithmes": ", ".join(selected_algos),
        }

        progress_text = st.empty()
        progress_bar = st.progress(0)

        trained_tables = {}
        for idx, label in enumerate(selected_algos, start=1):
            progress_text.text(f"Entraînement de {label} ({idx}/{len(selected_algos)})...")
            algo = ALGO_LABELS[label]
            Q, rewards, steps = run_algorithm(
                label, algo,
                train_episodes,
                alpha, gamma, eps_start, eps_decay,
                batch_size, memory_size, lr,
            )
            trained_tables[label] = Q
            all_results[label] = (rewards, steps)

            progress_text.text(f"Évaluation de {label}...")
            test_rewards, test_steps = evaluate_policy(label, Q, test_episodes)
            test_results[label] = (test_rewards, test_steps)

            summary.append({
                "Algorithme": label,
                "Train reward mean": f"{float(sum(rewards) / len(rewards)):.2f}",
                "Train reward last100": f"{float(sum(rewards[-100:]) / min(100, len(rewards))):.2f}",
                "Test reward mean": f"{float(sum(test_rewards) / len(test_rewards)):.2f}",
                "Test steps mean": f"{float(sum(test_steps) / len(test_steps)):.2f}",
            })

            progress_bar.progress(int(idx / len(selected_algos) * 100))
            time.sleep(0.2)

        progress_text.text("Simulation terminée.")
        progress_bar.empty()

        st.session_state.trained_tables = trained_tables
        st.session_state.run_done = True

        st.success("Les modèles ont été entraînés et évalués.")

        report_path = os.path.join(STORAGE_DIR, "report_streamlit.txt")
        generate_report(all_results, test_results, mode="Streamlit", params=report_params, output_path=report_path)

        st.header("Résumé des résultats")
        st.table(summary)

        st.header("Graphiques")
        learning_path, benchmark_path, boxplot_path = plot_all_results(all_results, test_results)
        st.image(learning_path, caption="Courbes d'apprentissage", width=700)
        st.image(benchmark_path, caption="Benchmark des algorithmes", width=700)
        st.image(boxplot_path, caption="Boxplots des résultats de test", width=700)

        with open(report_path, "r", encoding="utf-8") as f:
            st.header("Rapport texte généré")
            st.text(f.read())

        st.markdown(f"Fichier de rapport sauvegardé : `{report_path}`")

    if st.session_state.run_done and st.session_state.trained_tables:
        st.header("Visualisation Pygame")
        vis_algo = st.selectbox(
            "Choisir l'algorithme à visualiser",
            options=list(st.session_state.trained_tables.keys()),
        )
        vis_episodes = st.number_input(
            "Épisodes à visualiser", min_value=1, max_value=10, value=3, step=1, key="vis_episodes"
        )
        vis_delay = st.slider(
            "Délai entre les steps (secondes)", min_value=0.05, max_value=1.5, value=0.4, step=0.05, key="vis_delay"
        )

        if st.button("Visualiser la politique Pygame"):
            Q = st.session_state.trained_tables[vis_algo]
            try:
                thread = threading.Thread(
                    target=visualize_policy,
                    args=(Q, vis_episodes, vis_algo, vis_delay),
                    daemon=True,
                )
                thread.start()
                st.info("La fenêtre Pygame devrait s'ouvrir. Ferme-la pour revenir à l'interface Streamlit.")
            except Exception as exc:
                st.error(f"Impossible d'ouvrir Pygame : {exc}")


if __name__ == "__main__":
    main()
