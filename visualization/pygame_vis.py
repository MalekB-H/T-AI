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


def _make_stereo(wave):
    """Convert mono int16 array to stereo for pygame.sndarray."""
    return np.column_stack([wave, wave])


def _init_sounds():
    """Generate simple sound effects using pygame.mixer and numpy."""
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        sr = 44100

        # Ding sound (pickup) — short high-pitched sine
        t = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
        wave = (np.sin(2 * np.pi * 880 * t) * 0.4 * 32767).astype(np.int16)
        fade = np.linspace(1, 0, len(wave), dtype=np.float32)
        wave = (wave * fade).astype(np.int16)
        ding = pygame.sndarray.make_sound(_make_stereo(wave))

        # Success sound (dropoff) — ascending three-tone melody
        notes = [523, 659, 784]  # C5, E5, G5 — major chord
        parts = []
        for j, freq in enumerate(notes):
            dur = 0.15
            t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
            w = (np.sin(2 * np.pi * freq * t) * 0.4 * 32767).astype(np.int16)
            fade = np.linspace(1, 0.3 if j < 2 else 0, len(w), dtype=np.float32)
            parts.append((w * fade).astype(np.int16))
        combined = np.concatenate(parts)
        success = pygame.sndarray.make_sound(_make_stereo(combined))

        # Error sound (wrong action) — low buzz
        t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
        wave = (np.sin(2 * np.pi * 220 * t) * 0.25 * 32767).astype(np.int16)
        fade = np.linspace(1, 0, len(wave), dtype=np.float32)
        wave = (wave * fade).astype(np.int16)
        error = pygame.sndarray.make_sound(_make_stereo(wave))

        return {"ding": ding, "success": success, "error": error}
    except Exception:
        return None


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

    sounds = _init_sounds()

    print(f"\n  [TAXI]  Visualisation {label} ({n_episodes} épisodes) :")

    for i in range(n_episodes):
        state, _ = env.reset()
        done, total_reward, steps = False, 0, 0
        prev_passenger_loc = (state // 4) % 5

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

            # Sound effects
            if sounds:
                curr_passenger_loc = (state // 4) % 5
                if action == 4 and reward >= 0 and prev_passenger_loc < 4 and curr_passenger_loc == 4:
                    sounds["ding"].play()
                elif action == 5 and reward >= 0 and done:
                    sounds["success"].play()
                elif reward == -10:
                    sounds["error"].play()
                prev_passenger_loc = curr_passenger_loc

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
