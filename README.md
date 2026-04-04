<p align="center">
  <h1 align="center">TAXI DRIVER</h1>
  <p align="center">
    <strong>Reinforcement Learning Benchmark Platform</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-Taxi--v3-00ADD8?logo=openaigym&logoColor=white" alt="Gymnasium"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  </p>
</p>

<!-- Replace with your own GIF: record the dashboard, convert to GIF, upload to repo -->
<!-- <p align="center"><img src="assets/demo.gif" alt="Demo" width="800"></p> -->

---

## Overview

Taxi Driver is a benchmark platform that compares **6 reinforcement learning algorithms** on the Gymnasium Taxi-v3 environment. The agent (a taxi) must pick up passengers and drop them off at the correct destination in the minimum number of steps.

The platform includes a **Streamlit dashboard** for interactive training, comparison, and visualization — with real-time learning curves, auto-generated analysis, and a Pygame replay viewer.

---

## Features

| Category | Details |
|----------|---------|
| **6 Algorithms** | Random, Q-Learning, SARSA, Dyna-Q, Monte Carlo, DQN with Experience Replay |
| **3 Environments** | Standard Taxi-v3, 2 Passengers (14,400 states), Obstacles (danger zones) |
| **4 Reward Modes** | Default, Distance-based, Milestone, Aggressive |
| **Visualizations** | Learning curves, benchmark bars, boxplots, Q-Table heatmap, AI analysis |
| **Live Training** | Real-time chart during training |
| **Pygame Viewer** | Watch the agent play with sound effects |

---

## Architecture

```
taxi-driver/
├── agents/                     # RL algorithms
│   ├── q_learning.py           # Q-Learning (off-policy TD)
│   ├── sarsa.py                # SARSA (on-policy TD)
│   ├── dyna_q.py               # Dyna-Q (model-based, k mental replays)
│   ├── monte_carlo.py          # Monte Carlo (first-visit)
│   ├── dqn.py                  # DQN with experience replay + target network
│   └── random_agent.py         # Random baseline (brute-force)
│
├── environments/               # Custom environments
│   ├── multi_passenger.py      # 2 passengers, 14,400 states
│   ├── obstacle_taxi.py        # Danger zones with -100 penalty
│   └── shaped_taxi.py          # Reward shaping wrapper (4 modes)
│
├── visualization/              # Pygame visualization
│   └── pygame_vis.py           # Agent replay with sound effects
│
├── utils/                      # Plotting & reporting
│   ├── plots.py                # Matplotlib charts + Q-Table heatmap
│   └── report.py               # Auto-generated benchmark report
│
├── core/                       # Policy evaluation
│   └── tester.py               # Greedy policy testing
│
├── streamlit_app.py            # Interactive dashboard
├── main.py                     # CLI mode
└── requirements.txt
```

---

## Algorithms

```
                    ┌─────────────────────────────────────────────────┐
                    │            Reinforcement Learning               │
                    │                                                 │
                    │   Agent ──action──► Environment                 │
                    │     ▲                    │                      │
                    │     └──reward + state────┘                      │
                    └─────────────────────────────────────────────────┘
```

| Algorithm | Type | Method | Key Idea |
|-----------|------|--------|----------|
| **Q-Learning** | Off-policy | TD(0) | Learns optimal policy via `max(Q[s'])` |
| **SARSA** | On-policy | TD(0) | Learns from actual actions `Q[s', a']` — safer |
| **Dyna-Q** | Model-based | TD(0) + planning | k mental replays per step — 4x faster convergence |
| **Monte Carlo** | On-policy | First-visit | Learns from complete episodes |
| **DQN-ER** | Off-policy | Tabular DQN | Experience replay + target network |
| **Random** | — | Brute-force | Baseline comparison (~200 steps) |

### Key Comparisons

- **Q-Learning vs SARSA** — off-policy (optimistic) vs on-policy (cautious). Best shown with the Obstacle environment.
- **Q-Learning vs Dyna-Q** — model-free vs model-based. Dyna-Q converges ~4x faster with k=5 mental replays.
- **Default vs Shaped Rewards** — reward engineering reduces steps from ~20 to ~13.

---

## Quick Start

```bash
# Clone
git clone https://github.com/MalekB-H/T-AI.git
cd T-AI

# Install
pip install -r requirements.txt

# Dashboard
streamlit run streamlit_app.py

# CLI mode
python main.py
```

---

## Usage

### Dashboard (recommended)

1. Open http://localhost:8501
2. Select **Game mode** (1 Passenger, 2 Passengers, Obstacles)
3. Select **Algorithms** (Q-Learning, SARSA, Dyna-Q...)
4. Choose a **Preset** or tune hyperparameters manually
5. Click **Run Simulation**
6. Explore results: metrics, charts, heatmap, AI analysis
7. Click **Watch Agent Play** to see the agent in action (Pygame)

### CLI mode

```bash
python main.py
```

Enter training/test episodes when prompted. Results are saved as PNG charts and a text report.

---

## Results

Benchmark on standard Taxi-v3 (2,000 training episodes):

| Algorithm | Test Reward | Test Steps | Convergence |
|-----------|-----------|-----------|-------------|
| **Dyna-Q (k=5)** | +8.2 | ~13 | ~400 ep |
| **Q-Learning** | +7.5 | ~15 | ~800 ep |
| **SARSA** | +6.8 | ~16 | ~900 ep |
| **DQN-ER** | +7.0 | ~15 | ~1000 ep |
| **Monte Carlo** | +5.0 | ~18 | ~1500 ep |
| **Random** | -200 | ~200 | never |

> Dyna-Q converges **4x faster** than Q-Learning thanks to mental replay simulations.

---

## References

- **Sutton & Barto (2018)** — [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Chapters 6, 8
- **Gymnasium Taxi-v3** — [Documentation](https://gymnasium.farama.org/environments/toy_text/taxi/)
- **Watkins (1989)** — Q-Learning: original algorithm
- **Rummery & Niranjan (1994)** — SARSA: on-policy TD control
- **Sutton (1990)** — Dyna-Q: integrated learning, planning, and reacting

---

<p align="center">
  <strong>TAXI DRIVER</strong> · Reinforcement Learning Benchmark Platform · v1.0 · 2026
</p>
