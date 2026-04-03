import os
import sys
import subprocess
import tempfile
import pickle
import json
import time
import base64
import threading
import numpy as np
import gymnasium as gym
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agents import random_agent, q_learning, monte_carlo, dqn_experience_replay
from core.tester import test_policy
from utils.plots import plot_learning_curves, plot_bar_benchmark, plot_boxplots
from utils.report import generate_report
from visualization.pygame_vis import visualize_policy
from environments import MultiPassengerTaxiEnv

STORAGE_DIR = "streamlit_outputs"
os.makedirs(STORAGE_DIR, exist_ok=True)

ALGO_LABELS = {
    "Random":      random_agent,
    "Q-Learning":  q_learning,
    "Monte Carlo": monte_carlo,
    "DQN-ER":      dqn_experience_replay,
}
ALGO_COLORS = {
    "Random":      "#6b7280",
    "Q-Learning":  "#FBBF24",
    "Monte Carlo": "#3b82f6",
    "DQN-ER":      "#8b5cf6",
}
ALGO_ICONS = {
    "Random":      "🎲",
    "Q-Learning":  "⚡",
    "Monte Carlo": "🔵",
    "DQN-ER":      "🧠",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def moving_avg(data, window):
    out = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        out.append(sum(data[s:i + 1]) / (i - s + 1))
    return out


def run_algorithm(label, algo, train_episodes, alpha, gamma,
                  eps_start, eps_decay, batch_size, memory_size, lr,
                  multi_passenger=False):
    env = MultiPassengerTaxiEnv() if multi_passenger else gym.make("Taxi-v3")
    if label == "Random":
        rewards, steps = algo(env, train_episodes, verbose=False)
        Q = None
    elif label == "Q-Learning":
        Q, rewards, steps = algo(env, train_episodes, alpha=alpha, gamma=gamma,
                                 eps_start=eps_start, eps_decay=eps_decay, verbose=False)
    elif label == "Monte Carlo":
        Q, rewards, steps = algo(env, train_episodes, gamma=gamma,
                                 eps_start=eps_start, eps_decay=eps_decay, verbose=False)
    else:
        Q, rewards, steps = algo(env, train_episodes, gamma=gamma,
                                 eps_start=eps_start, eps_decay=eps_decay,
                                 batch_size=batch_size, memory_size=memory_size,
                                 lr=lr, verbose=False)
    env.close()
    return Q, rewards, steps


def run_episodes_for_viz(Q, label, n_episodes=3, max_steps=200):
    """Run episodes with rgb_array render and capture frames as base64 PNG."""
    from PIL import Image
    import io

    all_episodes = []
    for ep in range(n_episodes):
        env = gym.make("Taxi-v3", render_mode="rgb_array")
        state, _ = env.reset()
        total_r = 0.0
        frames = []

        rgb = env.render()
        img = Image.fromarray(rgb).resize((500, 500), Image.NEAREST)
        buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
        frames.append({"img": base64.b64encode(buf.getvalue()).decode(),
                        "reward": 0.0, "done": False, "action": -1})

        for _ in range(max_steps):
            action = int(np.argmax(Q[state])) if Q is not None else env.action_space.sample()
            ns, reward, terminated, truncated, _ = env.step(action)
            total_r += float(reward)
            done = bool(terminated or truncated)
            rgb = env.render()
            img = Image.fromarray(rgb).resize((500, 500), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
            frames.append({"img": base64.b64encode(buf.getvalue()).decode(),
                            "reward": round(total_r, 1), "done": done, "action": action})
            state = ns
            if done:
                break
        env.close()
        all_episodes.append(frames)
    return all_episodes


def build_grid_html(all_episodes, algo_label):
    episodes_meta = []
    all_images = []
    for ep in all_episodes:
        ep_meta = []
        for f in ep:
            all_images.append(f["img"])
            ep_meta.append({"idx": len(all_images) - 1, "reward": f["reward"],
                            "done": f["done"], "action": f["action"]})
        episodes_meta.append(ep_meta)

    meta_json = json.dumps(episodes_meta)
    imgs_json = json.dumps(all_images)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#080b14;font-family:'Inter',system-ui,sans-serif;
      color:#e2e8f0;display:flex;flex-direction:column;
      align-items:center;padding:20px 10px;gap:14px;}}
h3{{font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#FBBF24;}}
#stats-bar{{display:flex;gap:28px;}}
.stat{{display:flex;flex-direction:column;align-items:center;gap:2px;}}
.stat-label{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#475569;}}
.stat-value{{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#FBBF24;}}
#frame-img{{border:2px solid rgba(251,191,36,0.25);border-radius:10px;
            image-rendering:pixelated;width:500px;height:500px;}}
#action-lbl{{font-size:12px;color:#64748b;font-family:monospace;height:18px;}}
#controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center;}}
button{{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
        color:#FBBF24;padding:7px 14px;border-radius:6px;cursor:pointer;
        font-size:11px;font-weight:600;letter-spacing:1px;transition:background .2s;}}
button:hover{{background:rgba(251,191,36,0.18);}}
button.primary{{background:rgba(251,191,36,0.2);border-color:#FBBF24;}}
#speed-lbl{{font-size:11px;color:#64748b;font-family:monospace;min-width:36px;text-align:center;}}
</style></head>
<body>
<h3>🚕 Watch Agent Play — {algo_label}</h3>
<div id="stats-bar">
  <div class="stat"><div class="stat-label">Episode</div><div class="stat-value" id="ep-v">1</div></div>
  <div class="stat"><div class="stat-label">Step</div><div class="stat-value" id="step-v">0</div></div>
  <div class="stat"><div class="stat-label">Reward</div><div class="stat-value" id="rew-v">0</div></div>
  <div class="stat"><div class="stat-label">Status</div><div class="stat-value" id="stat-v" style="font-size:13px">—</div></div>
</div>
<img id="frame-img" src="" alt="Taxi-v3"/>
<div id="action-lbl">—</div>
<div id="controls">
  <button class="primary" onclick="togglePlay()" id="play-btn">▶ Play</button>
  <button onclick="prev()">‹ Prev</button>
  <button onclick="next()">Next ›</button>
  <button onclick="restart()">↺ Restart</button>
  <button onclick="chSpd(-1)">− Speed</button>
  <span id="speed-lbl">1×</span>
  <button onclick="chSpd(1)">+ Speed</button>
</div>
<script>
const EPISODES={meta_json};
const IMGS={imgs_json};
const ACTS=["South ↓","North ↑","East →","West ←","Pickup ▲","Dropoff ▼"];
const SPEEDS=[0.25,0.5,1,2,4];
let epIdx=0,fIdx=0,playing=false,spdIdx=2,timer=null;
const imgEl=document.getElementById('frame-img');

function show(){{
  const f=EPISODES[epIdx][fIdx];
  imgEl.src='data:image/png;base64,'+IMGS[f.idx];
  document.getElementById('ep-v').textContent=(epIdx+1)+'/'+EPISODES.length;
  document.getElementById('step-v').textContent=fIdx;
  document.getElementById('rew-v').textContent=f.reward;
  const sv=document.getElementById('stat-v');
  if(f.done){{sv.textContent='✓ Done';sv.style.color='#22c55e';}}
  else{{sv.textContent='Running';sv.style.color='#FBBF24';}}
  if(f.action>=0) document.getElementById('action-lbl').textContent='Last action: '+ACTS[f.action];
  else document.getElementById('action-lbl').textContent='—';
}}
show();

function advance(){{
  if(fIdx<EPISODES[epIdx].length-1)fIdx++;
  else if(epIdx<EPISODES.length-1){{epIdx++;fIdx=0;}}
  else{{stopPlay();return;}}
  show();
}}
function stopPlay(){{
  playing=false;clearInterval(timer);
  document.getElementById('play-btn').textContent='▶ Play';
}}
function togglePlay(){{
  playing=!playing;
  if(playing){{
    document.getElementById('play-btn').textContent='⏸ Pause';
    timer=setInterval(advance,700/SPEEDS[spdIdx]);
  }}else{{stopPlay();}}
}}
function next(){{if(!playing)advance();}}
function prev(){{
  if(!playing){{
    if(fIdx>0)fIdx--;
    else if(epIdx>0){{epIdx--;fIdx=EPISODES[epIdx].length-1;}}
    show();
  }}
}}
function restart(){{stopPlay();epIdx=0;fIdx=0;show();}}
function chSpd(d){{
  spdIdx=Math.max(0,Math.min(SPEEDS.length-1,spdIdx+d));
  document.getElementById('speed-lbl').textContent=SPEEDS[spdIdx]+'×';
  if(playing){{clearInterval(timer);timer=setInterval(advance,700/SPEEDS[spdIdx]);}}
}}
</script></body></html>"""
    return html


def evaluate_policy(label, Q, test_episodes, multi_passenger=False):
    if Q is None:
        env = MultiPassengerTaxiEnv() if multi_passenger else gym.make("Taxi-v3")
        rewards, steps = random_agent(env, test_episodes, verbose=False)
        env.close()
    elif multi_passenger:
        env = MultiPassengerTaxiEnv()
        rewards, steps = [], []
        for _ in range(test_episodes):
            state, _ = env.reset()
            total_r, s = 0, 0
            for _ in range(400):
                action = int(np.argmax(Q[state]))
                state, reward, done, trunc, _ = env.step(action)
                total_r += reward
                s += 1
                if done or trunc:
                    break
            rewards.append(total_r)
            steps.append(s)
        env.close()
    else:
        rewards, steps = test_policy(Q, test_episodes, verbose=False)
    return rewards, steps


def plot_all_results(all_results, test_results):
    lp = os.path.join(STORAGE_DIR, "learning_curves.png")
    bp = os.path.join(STORAGE_DIR, "benchmark_bar.png")
    xp = os.path.join(STORAGE_DIR, "boxplot_test.png")
    plot_learning_curves(all_results, output=lp)
    plot_bar_benchmark(all_results, output=bp)
    plot_boxplots(test_results, output=xp)
    return lp, bp, xp

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080b14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: #080b14 !important; }
[data-testid="block-container"] { padding-top: 0 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stHeader"] {
    background: rgba(8,11,20,0.97) !important;
    border-bottom: 1px solid rgba(251,191,36,0.1) !important;
}

/* sidebar toggle buttons */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    background: rgba(251,191,36,0.1) !important;
    border-radius: 6px !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill: #FBBF24 !important;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #0b0e18 !important;
    border-right: 1px solid rgba(251,191,36,0.12) !important;
}

/* download button */
[data-testid="stDownloadButton"] > button {
    background: rgba(251,191,36,0.08) !important;
    color: #FBBF24 !important;
    border: 1px solid rgba(251,191,36,0.35) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 10px 22px !important;
    transition: background .2s, border-color .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(251,191,36,0.15) !important;
    border-color: rgba(251,191,36,0.6) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #FBBF24; border-radius: 3px; }

/* sidebar */
[data-testid="stSidebar"] {
    background: #0b0e18 !important;
    border-right: 1px solid rgba(251,191,36,0.12) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #94a3b8 !important; font-size: 12px !important; }

.sidebar-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600; letter-spacing: 3px;
    color: rgba(251,191,36,0.6); text-transform: uppercase;
    margin-bottom: 20px; padding-bottom: 14px;
    border-bottom: 1px solid rgba(251,191,36,0.12);
}
.sidebar-section {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 2.5px !important; text-transform: uppercase !important;
    color: rgba(251,191,36,0.55) !important; margin: 18px 0 6px !important;
}
.sidebar-divider { height:1px; background:rgba(255,255,255,0.06); margin:14px 0; }

/* buttons */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#FBBF24 0%,#f97316 100%) !important;
    color: #080b14 !important; font-family:'Inter',sans-serif !important;
    font-weight: 700 !important; font-size: 12px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 8px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(251,191,36,0.28) !important;
    transition: opacity .2s, transform .15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: .88 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(251,191,36,0.42) !important;
}

/* progress */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg,#FBBF24,#f97316) !important;
}

/* section titles */
.section-title {
    font-size: 11px; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: #FBBF24;
    margin: 32px 0 18px; display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg,rgba(251,191,36,.3),transparent);
}

/* metric cards */
.metric-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 20px; position: relative; overflow: hidden;
    transition: border-color .3s, transform .2s;
}
.metric-card:hover { border-color:rgba(251,191,36,.3); transform:translateY(-2px); }
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: var(--accent,#FBBF24); opacity:.8;
}
.metric-label {
    font-size:11px; font-weight:500; letter-spacing:1.5px;
    text-transform:uppercase; color:#64748b; margin-bottom:10px;
}
.metric-value {
    font-size:2rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    color:var(--accent,#FBBF24); line-height:1; margin-bottom:4px;
}
.metric-sub { font-size:11px; color:#475569; }

/* algo cards */
.algo-card {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; padding: 18px; position: relative; overflow: hidden;
    transition: border-color .3s, transform .2s;
}
.algo-card:hover { border-color:rgba(255,255,255,.14); transform:translateY(-2px); }
.algo-card-accent {
    position:absolute; left:0; top:0; bottom:0; width:3px;
    background:var(--algo-color,#FBBF24); border-radius:10px 0 0 10px;
}
.algo-name {
    font-size:13px; font-weight:600; color:#e2e8f0;
    margin-bottom:14px; display:flex; align-items:center; gap:8px;
}
.algo-stat {
    display:flex; justify-content:space-between; align-items:center;
    padding:5px 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:12px;
}
.algo-stat:last-child { border-bottom:none; }
.algo-stat-label { color:#64748b; }
.algo-stat-value { font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--algo-color,#FBBF24); }

/* chart card */
.chart-card {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 24px 24px 8px; margin-bottom: 4px;
}
.chart-card-title {
    font-size:11px; font-weight:600; letter-spacing:2px;
    text-transform:uppercase; color:#94a3b8; margin-bottom:4px;
    display:flex; align-items:center; gap:8px;
}
.chart-card-title span { width:7px; height:7px; border-radius:50%; background:#FBBF24; display:inline-block; }

/* status */
.status-running {
    display:inline-flex; align-items:center; gap:6px;
    font-size:12px; font-weight:500; color:#FBBF24;
    background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.22);
    border-radius:20px; padding:4px 14px;
}
.status-dot {
    width:6px; height:6px; border-radius:50%; background:#FBBF24;
    animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:.4;transform:scale(1)} 50%{opacity:1;transform:scale(1.3)} }
.status-done {
    display:inline-flex; align-items:center; gap:6px;
    font-size:12px; font-weight:500; color:#10b981;
    background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.22);
    border-radius:20px; padding:4px 14px;
}

/* report */
.report-box {
    background:#0d1117; border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:20px 24px;
    font-family:'JetBrains Mono',monospace; font-size:12px;
    color:#94a3b8; white-space:pre-wrap; line-height:1.7;
    max-height:400px; overflow-y:auto;
}

/* table */
[data-testid="stTable"] table { background:transparent !important; width:100% !important; }
[data-testid="stTable"] th {
    background:rgba(251,191,36,.08) !important; color:#FBBF24 !important;
    font-size:11px !important; font-weight:600 !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
    padding:10px 14px !important; border-bottom:1px solid rgba(251,191,36,.18) !important;
}
[data-testid="stTable"] td {
    color:#e2e8f0 !important; font-family:'JetBrains Mono',monospace !important;
    font-size:12px !important; padding:10px 14px !important;
    border-bottom:1px solid rgba(255,255,255,.04) !important;
}
[data-testid="stTable"] tr:hover td { background:rgba(255,255,255,.03) !important; }
</style>
"""

def _video_b64():
    path = os.path.join(os.path.dirname(__file__), "assets", "bg.mp4")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

VIDEO_B64 = _video_b64()

HERO_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#080b14; overflow:hidden; width:100%; height:380px; font-family:'Inter',sans-serif; }

video {
    position:absolute; inset:0;
    width:100%; height:100%;
    object-fit:cover;
    z-index:0;
    opacity:0.45;
}
.video-overlay {
    position:absolute; inset:0;
    background: linear-gradient(
        to bottom,
        rgba(8,11,20,0.55) 0%,
        rgba(8,11,20,0.25) 50%,
        rgba(8,11,20,0.85) 100%
    );
    z-index:1;
}

.hero-content {
    position:absolute; inset:0; display:flex; flex-direction:column;
    align-items:center; justify-content:center; text-align:center; padding:0 20px;
    z-index:2;
}
.badge {
    display:inline-block;
    font-family:'Courier New',monospace; font-size:10px; font-weight:700;
    letter-spacing:3px; text-transform:uppercase; color:#FBBF24;
    border:1px solid rgba(251,191,36,.35); border-radius:20px;
    padding:5px 18px; margin-bottom:20px;
    background:rgba(251,191,36,.06);
    animation: fadeDown .6s ease both;
}
.title {
    font-size:clamp(2.6rem,7vw,4.2rem); font-weight:900; line-height:1.05;
    letter-spacing:-1px; margin-bottom:10px;
    background: linear-gradient(135deg,#ffffff 0%,#FBBF24 45%,#f97316 70%,#ffffff 100%);
    background-size:250% 250%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    animation: gradShift 5s ease infinite, fadeDown .7s ease both;
}
@keyframes gradShift {
    0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}
}
.tagline {
    font-family:'Courier New',monospace; font-size:12px; letter-spacing:3px;
    color:rgba(251,191,36,.65); margin-bottom:14px;
    animation: fadeDown .85s ease both;
}
.sub {
    font-size:14px; color:rgba(148,163,184,.8); max-width:480px; line-height:1.65;
    animation: fadeDown 1s ease both;
}
.taxi-badge {
    display:inline-block; background:rgba(251,191,36,.12);
    color:#FBBF24; padding:2px 8px; border-radius:4px;
    font-family:'Courier New',monospace; font-size:12px;
}
.dots { display:flex; gap:6px; justify-content:center; margin-top:20px; }
.dot {
    width:5px; height:5px; border-radius:50%;
    background:rgba(251,191,36,.35);
    animation: dotPulse 2s ease-in-out infinite;
}
.dot:nth-child(2){animation-delay:.3s;background:rgba(251,191,36,.6);}
.dot:nth-child(3){animation-delay:.6s;}
@keyframes dotPulse{0%,100%{transform:scale(1);opacity:.4}50%{transform:scale(1.5);opacity:1}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-14px)}to{opacity:1;transform:translateY(0)}}

.divider {
    position:absolute; bottom:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(251,191,36,.2), transparent);
}
</style>
</head>
<body>
<video autoplay loop muted playsinline id="bgvid">
    <source src="data:video/mp4;base64,VIDEO_B64_PLACEHOLDER" type="video/mp4">
</video>
<div class="video-overlay"></div>
<div class="hero-content">
    <div class="title">TAXI DRIVER</div>
    <div class="tagline">&lt; ANYTIME, ANYWHERE. /&gt;</div>
    <div class="sub">
        Dashboard de comparaison d'algorithmes de Reinforcement Learning
        sur <span class="taxi-badge">Taxi-v3</span>
    </div>
    <div class="dots">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
</div>
<div class="divider"></div>

<script>
void(0);
        ctx.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
        ctx.fill();

        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height)  p.vy *= -1;
    }

</script>
</body>
</html>
"""

# ── plotly charts ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,17,27,0.6)',
    font=dict(color='#94a3b8', family='Inter', size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=50, r=20, t=40, b=40),
    height=380,
)
AXIS_STYLE = dict(
    gridcolor='rgba(255,255,255,0.05)',
    zerolinecolor='rgba(255,255,255,0.08)',
    linecolor='rgba(255,255,255,0.06)',
)


def plotly_learning_curves(all_results):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reward — moving average", "Steps — moving average"),
    )
    for label, (rewards, steps) in all_results.items():
        color  = ALGO_COLORS.get(label, "#FBBF24")
        window = max(1, len(rewards) // 20)
        r_avg  = moving_avg(rewards, window)
        s_avg  = moving_avg(steps,   window)
        x      = list(range(len(r_avg)))
        fig.add_trace(go.Scatter(x=x, y=r_avg, name=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"<b>{label}</b><br>ep: %{{x}}<br>reward: %{{y:.1f}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=s_avg, name=label, showlegend=False,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"<b>{label}</b><br>ep: %{{x}}<br>steps: %{{y:.1f}}<extra></extra>"),
                      row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_annotations(font_color='#64748b', font_size=11)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def plotly_benchmark(all_results):
    labels = list(all_results.keys())
    colors = [ALGO_COLORS.get(l, "#FBBF24") for l in labels]
    r100   = [sum(v[0][-100:]) / min(100, len(v[0])) for v in all_results.values()]
    s100   = [sum(v[1][-100:]) / min(100, len(v[1])) for v in all_results.values()]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Mean Reward — last 100 ep.", "Mean Steps — last 100 ep."))
    fig.add_trace(go.Bar(x=labels, y=r100, marker_color=colors, name="Reward",
                         hovertemplate="<b>%{x}</b><br>reward: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=s100, marker_color=colors, name="Steps", showlegend=False,
                         hovertemplate="<b>%{x}</b><br>steps: %{y:.2f}<extra></extra>"),  row=1, col=2)
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_annotations(font_color='#64748b', font_size=11)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def plotly_boxplots(test_results):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Test Reward distribution", "Test Steps distribution"))
    for label, (rewards, steps) in test_results.items():
        color = ALGO_COLORS.get(label, "#FBBF24")
        fig.add_trace(go.Box(y=rewards, name=label, marker_color=color, line_color=color,
                             boxmean=True, hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Box(y=steps,   name=label, marker_color=color, line_color=color,
                             boxmean=True, showlegend=False), row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_annotations(font_color='#64748b', font_size=11)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── render helpers ────────────────────────────────────────────────────────────

def render_metric_cards(summary):
    best_steps  = min(summary, key=lambda x: float(x["Test steps mean"]))
    best_reward = max(summary, key=lambda x: float(x["Test reward mean"]))
    st.markdown('<div class="section-title">Performance Overview</div>', unsafe_allow_html=True)
    cards = [
        {"label": "Best Algorithm",  "value": best_steps["Algorithme"],
         "sub": f"Lowest steps: {best_steps['Test steps mean']}",    "accent": "#FBBF24"},
        {"label": "Best Reward",     "value": best_reward["Test reward mean"],
         "sub": f"Algorithm: {best_reward['Algorithme']}",           "accent": "#10b981"},
        {"label": "Min Steps",       "value": best_steps["Test steps mean"],
         "sub": "vs ~197 random baseline",                            "accent": "#3b82f6"},
        {"label": "Algorithms Run",  "value": str(len(summary)),
         "sub": "models trained & evaluated",                         "accent": "#8b5cf6"},
    ]
    cols = st.columns(4)
    for col, c in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{c['accent']}">
                <div class="metric-label">{c['label']}</div>
                <div class="metric-value">{c['value']}</div>
                <div class="metric-sub">{c['sub']}</div>
            </div>""", unsafe_allow_html=True)


def render_algo_cards(summary):
    st.markdown('<div class="section-title">Algorithm Breakdown</div>', unsafe_allow_html=True)
    cols = st.columns(len(summary))
    for col, row in zip(cols, summary):
        label = row["Algorithme"]
        color = ALGO_COLORS.get(label, "#FBBF24")
        icon  = ALGO_ICONS.get(label, "●")
        with col:
            st.markdown(f"""
            <div class="algo-card" style="--algo-color:{color}">
                <div class="algo-card-accent"></div>
                <div class="algo-name">{icon} {label}</div>
                <div class="algo-stat">
                    <span class="algo-stat-label">Train reward</span>
                    <span class="algo-stat-value">{row['Train reward mean']}</span>
                </div>
                <div class="algo-stat">
                    <span class="algo-stat-label">Last 100 ep.</span>
                    <span class="algo-stat-value">{row['Train reward last100']}</span>
                </div>
                <div class="algo-stat">
                    <span class="algo-stat-label">Test reward</span>
                    <span class="algo-stat-value">{row['Test reward mean']}</span>
                </div>
                <div class="algo-stat">
                    <span class="algo-stat-label">Test steps</span>
                    <span class="algo-stat-value">{row['Test steps mean']}</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Taxi Driver RL", page_icon="🚕",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # sidebar
    PRESETS_1P = {
        "Fast demo (500 ep)":      {"train": 500,  "test": 100, "alpha": 0.15, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.99},
        "Standard (2000 ep)":      {"train": 2000, "test": 200, "alpha": 0.15, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.995},
        "Deep training (5000 ep)": {"train": 5000, "test": 300, "alpha": 0.10, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.998},
    }
    PRESETS_2P = {
        "Fast demo (5000 ep)":      {"train": 5000,  "test": 100, "alpha": 0.15, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.9995},
        "Standard (15000 ep)":      {"train": 15000, "test": 200, "alpha": 0.15, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.9998},
        "Deep training (25000 ep)": {"train": 25000, "test": 300, "alpha": 0.10, "gamma": 0.99, "eps_s": 1.0, "eps_d": 0.9999},
    }

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🚕 &nbsp;Taxi Driver RL</div>', unsafe_allow_html=True)

        game_mode = st.radio("🎮 Game mode", ["1 Passenger", "2 Passengers (Bonus)"],
                             help="1 Passenger = standard Taxi-v3 | 2 Passengers = extended environment with route optimization")
        is_multi = game_mode.startswith("2")
        presets = PRESETS_2P if is_multi else PRESETS_1P

        preset = st.selectbox("⚡ Preset", options=["Custom"] + list(presets.keys()), index=0)
        p = presets.get(preset, None)

        st.markdown('<div class="sidebar-section">🧠 Algorithms</div>', unsafe_allow_html=True)
        selected_algos = st.multiselect("", options=list(ALGO_LABELS.keys()),
                                        default=["Q-Learning", "Monte Carlo", "DQN-ER"],
                                        label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">🏋️ Training</div>', unsafe_allow_html=True)
        default_train = p["train"] if p else (10000 if is_multi else 2000)
        default_test  = p["test"]  if p else 200
        train_episodes = st.number_input("Training episodes", min_value=100, max_value=50000,
                                         value=default_train, step=500 if is_multi else 100)
        test_episodes  = st.number_input("Test episodes", min_value=20, max_value=1000,
                                         value=default_test, step=10)

        st.markdown('<div class="sidebar-section">⚙️ Hyperparameters</div>', unsafe_allow_html=True)
        alpha     = st.slider("Alpha", 0.01, 1.0, p["alpha"] if p else 0.15, 0.01,
                              help="Learning rate — controls how much each update adjusts Q-values")
        gamma     = st.slider("Gamma", 0.80, 0.999, p["gamma"] if p else 0.99, 0.01,
                              help="Discount factor — how much the agent values future rewards vs immediate")
        eps_start = st.slider("Epsilon start", 0.0, 1.0, p["eps_s"] if p else 1.0, 0.05,
                              help="Initial exploration rate — 1.0 = fully random at start")
        eps_decay = st.slider("Epsilon decay", 0.90, 0.999, p["eps_d"] if p else 0.995, 0.001,
                              help="Decay multiplier per episode — lower = faster shift to exploitation")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">🔬 DQN Parameters</div>', unsafe_allow_html=True)
        batch_size  = st.number_input("Batch size", min_value=16, max_value=256, value=64, step=16,
                                      help="Number of transitions sampled from replay buffer per update")
        memory_size = st.number_input("Memory size", min_value=1000, max_value=20000, value=5000, step=500,
                                      help="Max capacity of the experience replay buffer")
        lr          = st.slider("DQN learning rate", 0.001, 0.1, 0.01, 0.001,
                                help="Step size for DQN tabular updates")

    # hero
    hero_html = HERO_HTML.replace("VIDEO_B64_PLACEHOLDER", VIDEO_B64)
    components.html(hero_html, height=380, scrolling=False)

    if not selected_algos:
        st.warning("Select at least one algorithm to run the simulation.")
        return

    st.session_state.setdefault("trained_tables", {})
    st.session_state.setdefault("run_done", False)
    st.session_state.setdefault("all_results", {})
    st.session_state.setdefault("test_results", {})
    st.session_state.setdefault("summary", [])

    if is_multi:
        st.markdown(
            '<div style="display:inline-block;background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.35);'
            'border-radius:20px;padding:5px 16px;font-size:12px;font-weight:600;color:#FBBF24;letter-spacing:1px;'
            'margin-bottom:16px;">🏆 BONUS — 2 Passengers Mode &nbsp;(14,400 states)</div>',
            unsafe_allow_html=True)

    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        run_button = st.button("▶  Run Simulation")

    if run_button:
        all_results  = {}
        test_results = {}
        summary      = []
        report_params = {
            "Episodes": train_episodes, "Test episodes": test_episodes,
            "Alpha": alpha, "Gamma": gamma,
            "Epsilon start": eps_start, "Epsilon decay": eps_decay,
            "Algorithmes": ", ".join(selected_algos),
        }

        status_slot   = col_status.empty()
        progress_bar  = st.progress(0)

        trained_tables = {}
        for idx, label in enumerate(selected_algos, start=1):
            status_slot.markdown(
                f'<div class="status-running"><div class="status-dot"></div>'
                f'Training {label} &nbsp;({idx}/{len(selected_algos)})</div>',
                unsafe_allow_html=True)
            progress_bar.progress(int((idx - 1) / len(selected_algos) * 100))

            algo = ALGO_LABELS[label]
            Q, rewards, steps = run_algorithm(label, algo, train_episodes,
                                              alpha, gamma, eps_start, eps_decay,
                                              batch_size, memory_size, lr,
                                              multi_passenger=is_multi)
            trained_tables[label] = Q
            all_results[label]    = (rewards, steps)

            status_slot.markdown(
                f'<div class="status-running"><div class="status-dot"></div>'
                f'Evaluating {label}…</div>', unsafe_allow_html=True)
            test_rewards, test_steps = evaluate_policy(label, Q, test_episodes, multi_passenger=is_multi)
            test_results[label]      = (test_rewards, test_steps)

            summary.append({
                "Algorithme":           label,
                "Train reward mean":    f"{float(sum(rewards)       / len(rewards)            ):.2f}",
                "Train reward last100": f"{float(sum(rewards[-100:]) / min(100, len(rewards)) ):.2f}",
                "Test reward mean":     f"{float(sum(test_rewards)  / len(test_rewards)       ):.2f}",
                "Test steps mean":      f"{float(sum(test_steps)    / len(test_steps)         ):.2f}",
            })
            time.sleep(0.1)

        progress_bar.progress(100)
        status_slot.markdown('<div class="status-done">✓ &nbsp;Simulation complete</div>',
                             unsafe_allow_html=True)

        report_path = os.path.join(STORAGE_DIR, "report_streamlit.txt")
        generate_report(all_results, test_results, mode="Streamlit",
                        params=report_params, output_path=report_path)
        plot_all_results(all_results, test_results)   # keep file exports

        st.session_state.trained_tables = trained_tables
        st.session_state.all_results    = all_results
        st.session_state.test_results   = test_results
        st.session_state.summary        = summary
        st.session_state.run_done       = True

    # results (persistent across reruns)
    if st.session_state.run_done and st.session_state.summary:
        summary      = st.session_state.summary
        all_results  = st.session_state.all_results
        test_results = st.session_state.test_results

        st.markdown("<br>", unsafe_allow_html=True)
        render_metric_cards(summary)
        render_algo_cards(summary)

        st.markdown('<div class="section-title">Detailed Results</div>', unsafe_allow_html=True)
        st.table(summary)

        # plotly charts
        st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card"><div class="chart-card-title"><span></span>Learning Curves</div></div>',
                    unsafe_allow_html=True)
        plotly_learning_curves(all_results)

        st.markdown('<div class="chart-card"><div class="chart-card-title"><span></span>Algorithm Benchmark</div></div>',
                    unsafe_allow_html=True)
        plotly_benchmark(all_results)

        st.markdown('<div class="chart-card"><div class="chart-card-title"><span></span>Test Distribution</div></div>',
                    unsafe_allow_html=True)
        plotly_boxplots(test_results)

        # report
        report_path = os.path.join(STORAGE_DIR, "report_streamlit.txt")
        if os.path.exists(report_path):
            st.markdown('<div class="section-title">Benchmark Report</div>', unsafe_allow_html=True)
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            col_dl1, col_dl2, col_info = st.columns([1, 1, 4])
            with col_dl1:
                st.download_button(
                    label="⬇ Export .txt",
                    data=report_content,
                    file_name="taxi_rl_report.txt",
                    mime="text/plain",
                )
            with col_dl2:
                st.download_button(
                    label="⬇ Export .md",
                    data=f"# Taxi Driver RL — Benchmark Report\n\n```\n{report_content}\n```",
                    file_name="taxi_rl_report.md",
                    mime="text/markdown",
                )
            with col_info:
                st.markdown(
                    '<p style="color:#475569;font-size:12px;margin-top:10px;">'
                    'Rapport généré automatiquement après simulation — contient hyperparamètres, '
                    'métriques d\'entraînement et résultats de test.</p>',
                    unsafe_allow_html=True,
                )

            with st.expander("Aperçu du rapport", expanded=False):
                st.markdown(f'<div class="report-box">{report_content}</div>', unsafe_allow_html=True)

    # policy visualization
    if st.session_state.run_done and st.session_state.trained_tables:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Policy Visualization</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#94a3b8;font-size:13px;margin-bottom:14px;">'
            'Lance une fenêtre de jeu Taxi-v3 — regarde l\'agent jouer en temps réel.</p>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            vis_algo = st.selectbox("Algorithm", options=list(st.session_state.trained_tables.keys()))
        with c2:
            vis_episodes = st.number_input("Episodes", min_value=1, max_value=10, value=3, step=1, key="vis_ep")
        with c3:
            vis_delay = st.slider("Step delay (s)", 0.05, 1.5, 0.4, 0.05, key="vis_dl")

        if st.button("▶  Watch Agent Play"):
            Q = st.session_state.trained_tables[vis_algo]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            pickle.dump(Q, tmp)
            tmp.close()
            script = f"""
import pickle, sys
sys.path.insert(0, '{os.path.dirname(__file__)}')
from visualization.pygame_vis import visualize_policy
with open('{tmp.name}', 'rb') as f:
    Q = pickle.load(f)
visualize_policy(Q, {int(vis_episodes)}, '{vis_algo}', {vis_delay}, multi_passenger={is_multi})
import os; os.unlink('{tmp.name}')
"""
            subprocess.Popen([sys.executable, "-c", script])
            st.info("Fenêtre Pygame ouverte. Ferme-la pour revenir au dashboard.")


if __name__ == "__main__":
    main()
