<p align="center">
  <h1 align="center">TAXI DRIVER</h1>
  <p align="center">
    <strong>Plateforme de Benchmark en Reinforcement Learning</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-Taxi--v3-00ADD8?logo=openaigym&logoColor=white" alt="Gymnasium"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  </p>
</p>

<p align="center"><img src="assets/demo.gif" alt="Demo" width="800"></p>

---

## Apercu

Taxi Driver est une plateforme de benchmark qui compare **6 algorithmes de Reinforcement Learning** sur l'environnement Gymnasium Taxi-v3. L'agent (un taxi) doit récupérer des passagers et les déposer à la bonne destination en un minimum de steps.

La plateforme inclut un **dashboard Streamlit** interactif pour l'entraînement, la comparaison et la visualisation — avec des courbes d'apprentissage en temps réel, une analyse auto-générée et un lecteur Pygame.

---

## Fonctionnalités

| Catégorie | Détails |
|-----------|---------|
| **6 Algorithmes** | Random, Q-Learning, SARSA, Dyna-Q, Monte Carlo, DQN avec Experience Replay |
| **3 Environnements** | Taxi-v3 standard, 2 Passagers (14 400 états), Obstacles (zones de danger) |
| **4 Modes de Reward** | Default, Distance, Milestone, Aggressive |
| **Visualisations** | Courbes d'apprentissage, barres de benchmark, boxplots, heatmap Q-Table, analyse IA |
| **Entraînement Live** | Graphique en temps réel pendant l'entraînement |
| **Lecteur Pygame** | Regarder l'agent jouer avec effets sonores |

---

## Architecture

```
taxi-driver/
├── agents/                     # Algorithmes de RL
│   ├── q_learning.py           # Q-Learning (off-policy TD)
│   ├── sarsa.py                # SARSA (on-policy TD)
│   ├── dyna_q.py               # Dyna-Q (model-based, k replays mentaux)
│   ├── monte_carlo.py          # Monte Carlo (first-visit)
│   ├── dqn.py                  # DQN avec experience replay + target network
│   └── random_agent.py         # Baseline aléatoire (brute-force)
│
├── environments/               # Environnements custom
│   ├── multi_passenger.py      # 2 passagers, 14 400 états
│   ├── obstacle_taxi.py        # Zones de danger avec pénalité -100
│   └── shaped_taxi.py          # Wrapper de reward shaping (4 modes)
│
├── visualization/              # Visualisation Pygame
│   └── pygame_vis.py           # Replay de l'agent avec effets sonores
│
├── utils/                      # Graphiques & rapports
│   ├── plots.py                # Graphiques Matplotlib + heatmap Q-Table
│   └── report.py               # Rapport de benchmark auto-généré
│
├── core/                       # Évaluation de politique
│   └── tester.py               # Test greedy de la politique
│
├── streamlit_app.py            # Dashboard interactif
├── main.py                     # Mode console
└── requirements.txt
```

---

## Algorithmes

```
                    ┌─────────────────────────────────────────────────┐
                    │            Reinforcement Learning               │
                    │                                                 │
                    │   Agent ──action──► Environnement               │
                    │     ▲                    │                      │
                    │     └──reward + état─────┘                      │
                    └─────────────────────────────────────────────────┘
```

| Algorithme | Type | Méthode | Idée clé |
|------------|------|---------|----------|
| **Q-Learning** | Off-policy | TD(0) | Apprend la politique optimale via `max(Q[s'])` |
| **SARSA** | On-policy | TD(0) | Apprend à partir des actions réelles `Q[s', a']` — plus prudent |
| **Dyna-Q** | Model-based | TD(0) + planning | k replays mentaux par step — convergence 4x plus rapide |
| **Monte Carlo** | On-policy | First-visit | Apprend à partir d'épisodes complets |
| **DQN-ER** | Off-policy | DQN tabulaire | Experience replay + target network |
| **Random** | — | Brute-force | Comparaison baseline (~200 steps) |

### Comparaisons clés

- **Q-Learning vs SARSA** — off-policy (optimiste) vs on-policy (prudent). Visible sur l'environnement Obstacles.
- **Q-Learning vs Dyna-Q** — model-free vs model-based. Dyna-Q converge ~4x plus vite avec k=5 replays mentaux.
- **Default vs Shaped Rewards** — le reward engineering réduit les steps de ~20 à ~13.

---

## Démarrage rapide

```bash
# Cloner
git clone https://github.com/MalekB-H/T-AI.git
cd T-AI

# Installer
pip install -r requirements.txt

# Dashboard
streamlit run streamlit_app.py

# Mode console
python main.py
```

---

## Utilisation

### Dashboard (recommandé)

1. Ouvrir http://localhost:8501
2. Sélectionner le **Game mode** (1 Passager, 2 Passagers, Obstacles)
3. Sélectionner les **Algorithmes** (Q-Learning, SARSA, Dyna-Q...)
4. Choisir un **Preset** ou ajuster les hyperparamètres manuellement
5. Cliquer **Run Simulation**
6. Explorer les résultats : métriques, graphiques, heatmap, analyse IA
7. Cliquer **Watch Agent Play** pour voir l'agent en action (Pygame)

### Mode console

```bash
python main.py
```

Entrer le nombre d'épisodes d'entraînement et de test. Les résultats sont sauvegardés en PNG et rapport texte.

---

## Résultats

Benchmark sur Taxi-v3 standard (2 000 épisodes d'entraînement) :

| Algorithme | Reward test | Steps test | Convergence |
|------------|-----------|-----------|-------------|
| **Dyna-Q (k=5)** | +8.2 | ~13 | ~400 ep |
| **Q-Learning** | +7.5 | ~15 | ~800 ep |
| **SARSA** | +6.8 | ~16 | ~900 ep |
| **DQN-ER** | +7.0 | ~15 | ~1000 ep |
| **Monte Carlo** | +5.0 | ~18 | ~1500 ep |
| **Random** | -200 | ~200 | jamais |

> Dyna-Q converge **4x plus vite** que Q-Learning grâce aux simulations de replay mental.

---

## Références

| Article | Auteurs | Année | Lien |
|---------|---------|-------|------|
| Reinforcement Learning: An Introduction (Ch. 6, 8) | Sutton & Barto | 2018 | [Livre (gratuit)](http://incompleteideas.net/book/the-book-2nd.html) |
| Learning from Delayed Rewards (Q-Learning) | Watkins | 1989 | [Thèse](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) |
| On-Line Q-Learning Using Connectionist Systems (SARSA) | Rummery & Niranjan | 1994 | [Article](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=5765a58ca19511aba0e3a3fcc1e46657530403a6) |
| Integrated Architectures for Learning, Planning, and Reacting (Dyna-Q) | Sutton | 1990 | [Article](https://doi.org/10.1016/B978-1-55860-141-3.50030-4) |
| Human-level Control Through Deep RL (DQN) | Mnih et al., DeepMind | 2015 | [Nature](https://www.nature.com/articles/nature14236) |
| Mastering the Game of Go Without Human Knowledge (AlphaGo Zero) | Silver et al., DeepMind | 2017 | [Nature](https://www.nature.com/articles/nature24270) |
| Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero) | Schrittwieser et al., DeepMind | 2020 | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| Gymnasium Taxi-v3 Environment | Farama Foundation | 2023 | [Documentation](https://gymnasium.farama.org/environments/toy_text/taxi/) |

> **Note** : Notre implémentation de Dyna-Q s'inscrit dans la lignée des approches **model-based RL** initiées par Sutton (1990). Cette famille d'algorithmes a évolué vers les architectures de DeepMind : **AlphaGo Zero** (2017) et **MuZero** (2020), qui utilisent le même principe fondamental — construire un modèle interne du monde pour planifier sans interaction réelle. Dyna-Q est l'ancêtre conceptuel de ces systèmes à la pointe de l'état de l'art.

---

<p align="center">
  <strong>TAXI DRIVER</strong> · Plateforme de Benchmark en Reinforcement Learning · v1.0 · 2026
</p>
