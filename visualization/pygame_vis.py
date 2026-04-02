"""Visualisation d'une politique Taxi-v3 avec Pygame ou mode texte."""

import time
import os
import numpy as np
import gymnasium as gym

def _pump_events() -> bool:
    """Pompe la file d'événements Pygame. Retourne False si fermé."""
    try:
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
    except Exception:
        pass
    return True

def _sleep_pumping(delay: float) -> bool:
    """Attente fractionnée en pompant les événements (évite freeze Windows)."""
    CHUNK = 0.05
    remaining = delay
    while remaining > 0:
        time.sleep(min(CHUNK, remaining))
        remaining -= CHUNK
        if not _pump_events():
            return False
    return True

def visualize_policy(Q: np.ndarray, n_episodes: int = 3, label: str = "Agent", delay: float = 0.5) -> None:
    """
    Joue n_episodes de démonstration avec Pygame, en utilisant
    une politique complétement greedy (pas d'exploration).
    """
    try:
        import pygame
        env = gym.make("Taxi-v3", render_mode="human")
    except Exception:
        print("  [!]  Pygame non disponible \u2014 mode texte (ANSI).")
        env = gym.make("Taxi-v3", render_mode="ansi")

    print(f"\n  [TAXI]  Visualisation {label} ({n_episodes} épisodes) :")
    
    for i in range(n_episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0

        while not done:
            if not _pump_events():
                print("  [STOP]  Fenêtre fermée par l'utilisateur.")
                env.close()
                return

            # Politique Greedy pur
            action = int(np.argmax(Q[state])) if Q is not None else env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
            steps += 1

            # Fallback texte si Pygame plante
            try:
                frame = env.render()
                if isinstance(frame, str):
                    os.system("cls" if os.name == "nt" else "clear")
                    print(frame)
                    print(f"Steps: {steps}  |  Reward: {total_reward:.1f}")
            except Exception:
                pass

            if not _sleep_pumping(delay):
                env.close()
                return

        print(f"  \u00c9pisode {i+1}/{n_episodes} -> Steps: {steps} | Reward: {total_reward:.1f}")
        if not _sleep_pumping(1.0):
            env.close()
            return
            
    env.close()
    print("  [OK]  Visualisation terminée.")

