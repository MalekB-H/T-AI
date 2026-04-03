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


def _encode_taxi_state(taxi_row, taxi_col, pass_loc, dest):
    """Encode into Taxi-v3 state format."""
    return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest


def visualize_policy(Q: np.ndarray, n_episodes: int = 3, label: str = "Agent",
                     delay: float = 0.5, multi_passenger: bool = False) -> None:
    """
    Joue n_episodes de démonstration avec Pygame, en utilisant
    une politique complétement greedy (pas d'exploration).
    """
    try:
        import pygame
        render_env = gym.make("Taxi-v3", render_mode="human")
    except Exception:
        print("  [!]  Pygame non disponible \u2014 mode texte (ANSI).")
        render_env = gym.make("Taxi-v3", render_mode="ansi")

    sounds = _init_sounds()

    # logic env: either Taxi-v3 or MultiPassenger
    if multi_passenger:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from environments import MultiPassengerTaxiEnv
        logic_env = MultiPassengerTaxiEnv()
    else:
        logic_env = None  # use render_env directly

    LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

    print(f"\n  [TAXI]  Visualisation {label} ({n_episodes} épisodes) :")

    for i in range(n_episodes):
        if multi_passenger:
            state, _ = logic_env.reset()
            render_env.reset()
            # sync render env to show initial state
            tr, tc = logic_env.taxi_row, logic_env.taxi_col
            p1_loc, d1 = logic_env.pass1_loc, logic_env.dest1
            render_env.unwrapped.s = _encode_taxi_state(tr, tc, p1_loc, d1)
            render_env.render()
            # show start screen for passenger 1
            try:
                import pygame
                surface = pygame.display.get_surface()
                if surface:
                    overlay = pygame.Surface(surface.get_size())
                    overlay.fill((0, 0, 0))
                    overlay.set_alpha(180)
                    surface.blit(overlay, (0, 0))
                    big_font = pygame.font.SysFont("Arial", 32, bold=True)
                    small_font = pygame.font.SysFont("Arial", 22)
                    line1 = big_font.render("2 Passengers Mode", True, (251, 191, 36))
                    line2 = small_font.render("Starting: Passenger 1...", True, (255, 255, 255))
                    cx = surface.get_width() // 2
                    cy = surface.get_height() // 2
                    surface.blit(line1, (cx - line1.get_width() // 2, cy - 30))
                    surface.blit(line2, (cx - line2.get_width() // 2, cy + 20))
                    pygame.display.flip()
                    pygame.time.wait(2000)
            except Exception:
                pass
        else:
            state, _ = render_env.reset()

        done, total_reward, steps = False, 0, 0
        if not multi_passenger:
            prev_passenger_loc = (state // 4) % 5
        else:
            prev_pass1 = logic_env.pass1_loc
            prev_pass2 = logic_env.pass2_loc

        while not done:
            if not _pump_events():
                print("  [STOP]  Fenêtre fermée par l'utilisateur.")
                render_env.close()
                if logic_env:
                    logic_env.close()
                return

            action = int(np.argmax(Q[state])) if Q is not None else (logic_env or render_env).action_space.sample()

            if multi_passenger:
                state, reward, done, truncated, _ = logic_env.step(action)
                done = done or truncated
                total_reward += reward
                steps += 1

                tr, tc = logic_env.taxi_row, logic_env.taxi_col
                p1 = logic_env.pass1_loc
                p2 = logic_env.pass2_loc
                d1 = logic_env.dest1
                d2 = logic_env.dest2

                # determine which passenger to show on Taxi-v3 renderer
                if p1 < 5:  # passenger 1 not delivered yet
                    show_pass = p1 if p1 < 4 else 4  # 0-3=location, 4=in taxi
                    show_dest = d1
                    phase = "Passenger 1/2"
                else:  # passenger 1 delivered, showing passenger 2
                    show_pass = p2 if p2 < 4 else 4
                    show_dest = d2
                    phase = "Passenger 2/2"

                render_env.unwrapped.s = _encode_taxi_state(tr, tc, show_pass, show_dest)

                # detect passenger 1 just delivered → show transition screen
                if prev_pass1 != 5 and p1 == 5:
                    # --- TRANSITION SCREEN ---
                    try:
                        import pygame
                        surface = pygame.display.get_surface()
                        if surface:
                            # darken the screen
                            overlay = pygame.Surface(surface.get_size())
                            overlay.fill((0, 0, 0))
                            overlay.set_alpha(180)
                            surface.blit(overlay, (0, 0))
                            # big message
                            big_font = pygame.font.SysFont("Arial", 32, bold=True)
                            small_font = pygame.font.SysFont("Arial", 22)
                            line1 = big_font.render("Passenger 1 Delivered!", True, (34, 197, 94))
                            line2 = small_font.render("Next: Passenger 2...", True, (255, 255, 255))
                            cx = surface.get_width() // 2
                            cy = surface.get_height() // 2
                            surface.blit(line1, (cx - line1.get_width() // 2, cy - 30))
                            surface.blit(line2, (cx - line2.get_width() // 2, cy + 20))
                            pygame.display.flip()
                            pygame.time.wait(2000)  # pause 2 seconds
                    except Exception:
                        pass

                # sound effects
                if sounds:
                    if action == 4 and reward >= 0:
                        if (prev_pass1 < 4 and p1 == 4) or (prev_pass2 < 4 and p2 == 4):
                            sounds["ding"].play()
                    elif action == 5 and reward == 20:
                        sounds["success"].play()
                    elif reward == -10:
                        sounds["error"].play()
                    prev_pass1, prev_pass2 = p1, p2

                try:
                    frame = render_env.render()
                    if isinstance(frame, str):
                        os.system("cls" if os.name == "nt" else "clear")
                        print(frame)
                        print(f"[{phase}]  Steps: {steps}  |  Reward: {total_reward:.1f}")
                    else:
                        # overlay text on Pygame window
                        try:
                            import pygame
                            surface = pygame.display.get_surface()
                            if surface:
                                font = pygame.font.SysFont("Arial", 22, bold=True)
                                # phase badge
                                bg_color = (251, 191, 36) if "1" in phase else (34, 197, 94)
                                txt = font.render(phase, True, (0, 0, 0))
                                pad = 8
                                badge = pygame.Surface((txt.get_width() + pad * 2, txt.get_height() + pad * 2))
                                badge.fill(bg_color)
                                badge.blit(txt, (pad, pad))
                                surface.blit(badge, (10, 10))
                                # steps + reward
                                info_font = pygame.font.SysFont("Arial", 16)
                                info = info_font.render(f"Steps: {steps}  |  Reward: {total_reward:.0f}", True, (255, 255, 255))
                                info_bg = pygame.Surface((info.get_width() + 12, info.get_height() + 8))
                                info_bg.fill((0, 0, 0))
                                info_bg.set_alpha(180)
                                surface.blit(info_bg, (10, 52))
                                surface.blit(info, (16, 56))
                                pygame.display.flip()
                        except Exception:
                            pass
                except Exception:
                    pass

            else:
                state, reward, done, truncated, _ = render_env.step(action)
                done = done or truncated
                total_reward += reward
                steps += 1

                if sounds:
                    curr_passenger_loc = (state // 4) % 5
                    if action == 4 and reward >= 0 and prev_passenger_loc < 4 and curr_passenger_loc == 4:
                        sounds["ding"].play()
                    elif action == 5 and reward >= 0 and done:
                        sounds["success"].play()
                    elif reward == -10:
                        sounds["error"].play()
                    prev_passenger_loc = curr_passenger_loc

                try:
                    frame = render_env.render()
                    if isinstance(frame, str):
                        os.system("cls" if os.name == "nt" else "clear")
                        print(frame)
                        print(f"Steps: {steps}  |  Reward: {total_reward:.1f}")
                except Exception:
                    pass

            if not _sleep_pumping(delay):
                render_env.close()
                if logic_env:
                    logic_env.close()
                return

        print(f"  \u00c9pisode {i+1}/{n_episodes} -> Steps: {steps} | Reward: {total_reward:.1f}")
        if not _sleep_pumping(1.0):
            render_env.close()
            if logic_env:
                logic_env.close()
            return

    render_env.close()
    if logic_env:
        logic_env.close()
    print("  [OK]  Visualisation terminée.")
