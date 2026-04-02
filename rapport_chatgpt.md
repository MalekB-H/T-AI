# Projet Taxi Driver RL — Données pour rédaction de rapport

## Contexte et objectif

- Projet : Taxi Driver RL
- Objectif : comparer plusieurs méthodes de Reinforcement Learning sur l’environnement `Taxi-v3` de Gymnasium.
- Méthodes implémentées : `Random`, `Q-Learning`, `Monte Carlo`, `DQN tabulaire` avec replay buffer et target network.
- Interface : mode console (`main.py`) et interface Streamlit (`streamlit_app.py`).
- Résultat attendu : comparaison chiffrée des méthodes avec graphiques et benchmark.

## Architecture du projet

- `agents/` : implémentations des algorithmes.
  - `random_agent.py`
  - `q_learning.py`
  - `monte_carlo.py`
  - `dqn.py`
- `core/` : évaluation de politique (`tester.py`).
- `utils/` : utilitaires, génération de graphiques et rapport.
- `visualization/` : visualisation Pygame de la politique.
- `streamlit_app.py` : interface web interactive.

## Méthodes implémentées

### Random
- Algorithme entièrement aléatoire.
- Utilisé comme baseline naïve (brute-force) de comparaison.
- Pas de phase d’apprentissage.

### Q-Learning
- Algorithme TD(0) off-policy avec politique epsilon-greedy.
- Mise à jour de la Q-table : `Q[s,a] += alpha * (reward + gamma * max(Q[s']) - Q[s,a])`.
- Bonne méthode pour Taxi-v3, rapide et efficace sur cet environnement discret.

### Monte Carlo
- Algorithme on-policy First-Visit.
- Génère un épisode complet puis met à jour les Q-values en arrière sur la première visite de chaque `(s,a)`.
- Plus lent à converger que Q-Learning, mais utile pour comparaison méthodologique.

### DQN tabulaire
- Implémentation simplifiée de DQN adaptée à Taxi-v3.
- Comporte :
  - replay buffer (experience replay)
  - target network
  - mini-batch de transitions
- Sur Taxi-v3, cette version est pédagogique, mais plus lente que Q-Learning.

## Hyperparamètres et rôle

### Paramètres généraux
- `train_episodes` : nombre d’épisodes d’entraînement. Plus élevé signifie plus de temps de calcul, mais meilleure convergence possible.
- `test_episodes` : nombre d’épisodes d’évaluation. Plus élevé donne des statistiques plus fiables.

### Paramètres de Q-Learning et Monte Carlo
- `alpha` : taux d’apprentissage.
  - augmente la vitesse de mise à jour.
  - trop élevé = instabilité.
  - trop faible = apprentissage lent.
- `gamma` : facteur de discount.
  - proche de 1 = les récompenses futures sont valorisées.
  - plus faible = l’agent privilégie le gain immédiat.
- `eps_start`, `eps_decay`, `eps_min` : paramètres de l’exploration epsilon-greedy.
  - `eps_start` = niveau d’exploration initial.
  - `eps_decay` = vitesse de diminution de l’exploration.
  - `eps_min` = exploration minimale conservée.

### Paramètres DQN
- `batch_size` : taille du mini-batch issu du replay buffer.
- `memory_size` : taille maximale du replay buffer.
- `lr` : taux d’apprentissage pour la mise à jour par mini-batch.
- `target_update` : fréquence de copie du target network.

## Valeurs conseillées utilisées pour l’expérience

- `train_episodes` : 2000
- `test_episodes` : 200

### Q-Learning
- `alpha = 0.15`
- `gamma = 0.99`
- `eps_start = 1.0`
- `eps_decay = 0.995`
- `eps_min = 0.05`

### Monte Carlo
- `gamma = 0.99`
- `eps_start = 1.0`
- `eps_decay = 0.995`
- `eps_min = 0.05`

### DQN tabulaire
- `gamma = 0.99`
- `eps_start = 1.0`
- `eps_decay = 0.995`
- `eps_min = 0.05`
- `batch_size = 64`
- `memory_size = 5000`
- `lr = 0.01`
- `target_update = 50`

## Protocole expérimental

- Environnement : `Taxi-v3` de Gymnasium.
- Entraînement : 2000 épisodes pour chaque méthode.
- Évaluation : 200 épisodes en mode greedy pour Q-Learning, Monte Carlo et DQN.
- Baseline aléatoire : 200 épisodes aléatoires pour la comparaison.

## Résultats chiffrés observés

| Méthode        | Reward moyen train | Reward moyen test | Steps moyen test | Temps entraînement (s) | Notes |
|---------------|-------------------:|------------------:|-----------------:|-----------------------:|-------|
| Random        | -768.78            | -773.81           | 196.14           | 10.25                  | Baseline naïve |
| Q-Learning    | -69.59             | -3.11             | 22.96            | 2.98                   | Convergence rapide |
| Monte Carlo   | -593.51            | -438.72           | 196.14           | 6.64                   | Très lent, stable tardivement |
| DQN tabulaire | -150.20            | 4.71              | 15.98            | 86.40                  | Plus lent, pédagogique |

### Observations directes
- `Random` : baseline naïve, performance très faible comme attendu.
- `Q-Learning` : meilleur compromis temps/performance sur Taxi-v3.
- `Monte Carlo` : apprentissage lent, nécessite plus d’épisodes pour bien converger.
- `DQN` : meilleur test reward que Q-Learning ici, mais coût d’entraînement très élevé et méthode plus lourde.

## Graphiques recommandés

1. `Reward moyen par épisode` pour chaque méthode.
2. `Steps moyen par épisode` ou `convergence des steps`.
3. `Comparaison des performances finales` (bar chart des rewards moyens et steps moyens).

### Ce qu’il faut analyser dans les graphes

- Comparer la courbe d’apprentissage de chaque méthode.
- Vérifier si la méthode converge vers une récompense stable.
- Expliquer pourquoi Q-Learning est plus rapide à converger que Monte Carlo.
- Montrer que DQN est plus coûteux en temps, même si ses résultats sont bons.

## Structure conseillée du rapport final

1. **Introduction**
   - Objectif du projet.
   - Environnement utilisé.
   - Méthodes comparées.

2. **Description du projet**
   - Architecture des dossiers.
   - Interface console et Streamlit.
   - Fichiers de génération de graphiques et de rapport.

3. **Méthodes implémentées**
   - Random / baseline naïve.
   - Q-Learning.
   - Monte Carlo.
   - DQN tabulaire.

4. **Hyperparamètres et leur rôle**
   - Explication de `alpha`, `gamma`, `epsilon`, etc.
   - Pourquoi ces paramètres sont utiles.

5. **Valeurs testées et stratégie d’optimisation**
   - Valeurs de base choisies.
   - Raisons des choix.
   - Impacts attendus.

6. **Résultats expérimentaux**
   - Tableau chiffré avec reward, steps, temps.
   - Analyse claire des performances.

7. **Visualisation des résultats**
   - Décrire 3 graphes pertinents.
   - Analyser les tendances.

8. **Analyse et observations**
   - Q-Learning : meilleur sur Taxi-v3.
   - Monte Carlo : apprentissage plus lent.
   - DQN : méthode pédagogique, coût élevé.
   - Random : baseline naïve.

9. **Conclusion**
   - Résumé des points clés.
   - Validation du respect du sujet.

10. **Perspectives et améliorations**
    - Tester plus d’hyperparamètres.
    - Ajouter un vrai DQN avec réseau de neurones.
    - Mesurer le temps d’entraînement dans le rapport.
    - Améliorer la visualisation.

## Instructions pour ChatGPT

- Rédige un rapport clair, structuré et complet en utilisant les sections ci-dessus.
- Utilise le tableau chiffré comme base de résultats.
- Précise que l’environnement est `Taxi-v3`.
- Mentionne que le projet contient une baseline naïve, trois méthodes RL et une interface Streamlit.
- Insiste sur le fait que les hyperparamètres sont modifiables et leur rôle.
- Ajoute une section dédiée aux graphiques et à leur analyse.
