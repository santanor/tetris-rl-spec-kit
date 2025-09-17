from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from tetris_rl.agent.trainer import TrainingConfig
from tetris_rl.core.board import TetrisBoard
from tetris_rl.env.reward_config import RewardConfig

app = FastAPI(title="Tetris RL Dashboard")

# Simple in-memory state
latest_step: Dict[str, Any] = {}
current_board: Optional[TetrisBoard] = None
session_stats: Dict[str, Any] = {"episodes": 0, "total_reward": 0.0}
websocket_clients: List[WebSocket] = []
training_task: Optional[asyncio.Task] = None
reward_config = RewardConfig()  # shared minimal reward config
broadcast_every: int = 1  # dynamically adjustable cadence
training_mode: str = "idle"  # 'idle' | 'dashboard' | 'headless'
headless_stats: Dict[str, Any] = {
    "episodes": 0,
    "last_reward": 0.0,
    "last_lines": 0,
    "avg_reward": 0.0,
    "avg_lines": 0.0,
    "global_step": 0,
    # New aggregated reward component diagnostics (rolling window)
    "avg_components": {
        "line_reward": 0.0,
        "survival": 0.0,
        "placement": 0.0,
        "top_out": 0.0,
    },
    "last_components": {
        "line_reward": 0.0,
        "survival": 0.0,
        "placement": 0.0,
        "top_out": 0.0,
    },
}

# (All dynamic training/model override machinery removed for minimal demo)

@app.get("/api/session")
async def get_session():
    return session_stats

@app.get("/api/board")
async def get_board():
    if current_board:
        return current_board.snapshot()
    return {"board": [], "lines_cleared_total": 0}

@app.get("/")
async def root():
    static_index = Path(__file__).parent / "static" / "index.html"
    if static_index.exists():
        return HTMLResponse(static_index.read_text(encoding="utf-8"))
    # fallback redirect to /static/ if directory mounted
    return RedirectResponse(url="/static/")

async def broadcast(message: Dict[str, Any]):
    if not websocket_clients:
        return
    data = json.dumps(message)
    to_remove = []
    for ws in websocket_clients:
        try:
            await ws.send_text(data)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        websocket_clients.remove(ws)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    # on connect send current snapshot
    if current_board:
        await websocket.send_text(json.dumps({"type": "snapshot", "board": current_board.snapshot()}))
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

async def training_loop(output_dir: Path, episodes: int, device_str: str = "auto"):
    """Run minimal training using the simplified trainer; broadcast per-step + episode events."""
    global current_board, training_mode
    training_mode = "dashboard"

    # Configure training
    cfg = TrainingConfig(episodes=episodes, device=device_str)
    session_reward = 0.0

    # Step callback invoked from trainer.run_training
    async def async_broadcast_step(payload: Dict[str, Any]):
        await broadcast({"type": "step", **payload})

    # Wrap sync callback -> schedule coroutine
    def step_callback(step_info: Dict[str, Any]):
        # Insert board + network activations by performing a forward pass on current state (best‑effort)
        try:
            # step_info currently lacks board; we can ignore for now – Web UI will update only on episode boundaries
            pass
        except Exception:
            pass
        asyncio.get_event_loop().create_task(async_broadcast_step(step_info))

    # We re‑implement a very small inline loop to expose board & activations live instead of calling run_training directly.
    from tetris_rl.agent.trainer import select_action, optimize, _resolve_device
    from tetris_rl.agent.dqn import DQN
    from tetris_rl.agent.replay_buffer import ReplayBuffer
    from tetris_rl.env.tetris_env import TetrisEnv
    import torch, random as _rnd

    device = _resolve_device(cfg.device)
    rng = _rnd.Random(cfg.seed)
    # Probe obs dim
    probe_env = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
    probe_obs,_ = probe_env.reset()
    obs_dim = len(probe_obs)
    probe_env.close()
    policy_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64,64])
    target_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64,64])
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device); target_net.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    global_step = 0

    for ep in range(cfg.episodes):
        env = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
        state,_ = env.reset()
        done = False; truncated = False
        ep_reward = 0.0
        ep_lines = 0
        action_counts = {"random":0, "greedy":0}
        while not (done or truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            action, meta = select_action(policy_net, state_t, global_step, cfg, 5)
            next_state, reward, done, truncated, info = env.step(action)
            buffer.push(state, action, reward, next_state, done or truncated)
            current_board = env.board
            loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
            if global_step % cfg.target_sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            # Forward activations (best effort)
            try:
                q_vals, acts = policy_net.forward_with_activations(state_t.unsqueeze(0))  # type: ignore[attr-defined]
            except Exception:
                q_vals, acts = [], {"layers": []}
            # Only broadcast if cadence condition met or terminal event
            be_local = max(1, int(broadcast_every))
            if (global_step % be_local) == 0 or done or truncated:
                step_payload = {
                    "episode": ep,
                    "global_step": global_step,
                    "reward": reward,
                    "loss": loss_val,
                    "lines_cleared_total": info.get("lines_cleared_total"),
                    "lines_delta": info.get("lines_delta"),
                    "epsilon": meta.get("epsilon"),
                    "action_source": meta.get("action_source"),
                    "last_action": action,
                    "board": current_board.snapshot() if current_board else None,
                    "board_snapshot": current_board.snapshot() if current_board else None,
                    "reward_components": info.get("reward_components"),
                    "q_values": q_vals,
                    "net_meta": {"input_dim": obs_dim, "hidden_layers": cfg.hidden_layers, "num_actions": 5, "dueling": False},
                    "net_activations": acts,
                    "heights": current_board.heights() if current_board else None,
                    "hole_columns": current_board.hole_columns() if current_board else None,
                }
                await broadcast({"type": "step", **step_payload})
            state = next_state
            ep_reward += reward
            ep_lines = info.get("lines_cleared_total", ep_lines)
            src = meta.get("action_source")
            if src in action_counts:
                action_counts[src] += 1
            global_step += 1
            await asyncio.sleep(0)
        session_reward += ep_reward
        total_actions = max(1, sum(action_counts.values()))
        action_dist = {
            "random": action_counts["random"]/total_actions,
            "greedy": action_counts["greedy"]/total_actions,
            "boltzmann": 0.0,
        }
        await broadcast({
            "type": "episode_end",
            "episode": ep,
            "reward": ep_reward,
            "line_clears": ep_lines,
            "action_dist": action_dist,
            "last_epsilon": meta.get("epsilon"),
            "last_temperature": None,
        })
        env.close()

    await broadcast({"type": "session_end", "episodes": cfg.episodes, "avg_reward": session_reward / max(1,cfg.episodes)})
    training_mode = "idle"

@app.post("/api/train")
async def start_training(episodes: int = 5, device: str = "auto"):
    """Start a minimal training session (no multi-env, no resume)."""
    global training_task
    if training_task and not training_task.done():
        return {"status": "already-running"}
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device))
    return {"status": "started", "episodes": episodes, "device": device}


@app.get("/api/broadcast-config")
async def get_broadcast_config():
    return {"broadcast_every": broadcast_every}


@app.post("/api/broadcast-config")
async def update_broadcast_config(payload: Dict[str, Any]):
    global broadcast_every
    if not isinstance(payload, dict):
        return {"status": "error", "error": "invalid-payload"}
    be = payload.get("broadcast_every")
    try:
        if be is not None:
            v = int(be)
            if v < 1:
                raise ValueError
            broadcast_every = v
    except (TypeError, ValueError):
        return {"status": "error", "error": "broadcast_every must be >=1"}
    await broadcast({"type": "broadcast_config_update", "broadcast_every": broadcast_every})
    return {"status": "ok", "broadcast_every": broadcast_every}


# ---------------- Headless (fast) training mode ---------------- #
@app.post("/api/train-headless")
async def start_headless(
    episodes: int = 100,
    device: str = "auto",
    seed: int = 0,
    infinite: int = 0,
    print_every: int = 1,
    opt_every: int = 1,
):
    """Start headless training (no websocket broadcasting) at max speed.

    Parameters:
      episodes: number of episodes to run (ignored if infinite=1)
      infinite: if 1, run until manually stopped
      print_every: print progress every N episodes
    """
    global training_task, training_mode
    if training_task and not training_task.done():
        return {"status": "already-running", "mode": training_mode}
    training_task = asyncio.create_task(_headless_loop(episodes, device, seed, bool(infinite), max(1, print_every), max(1, opt_every)))
    return {"status": "started", "mode": "headless", "episodes": episodes, "infinite": bool(infinite), "opt_every": max(1, opt_every)}


@app.get("/api/headless-status")
async def get_headless_status():
    return {"mode": training_mode, "stats": headless_stats}


@app.post("/api/stop")
async def stop_training():
    global training_task, training_mode
    if training_task and not training_task.done():
        training_task.cancel()
        try:
            await training_task
        except Exception:
            pass
        training_mode = "idle"
        return {"status": "stopped"}
    return {"status": "idle"}


async def _headless_loop(episodes: int, device_str: str, seed: int, infinite: bool, print_every: int, opt_every: int):
    """Fast single-env loop with stdout logging only."""
    from tetris_rl.agent.trainer import TrainingConfig, select_action, optimize, _resolve_device
    from tetris_rl.agent.dqn import DQN
    from tetris_rl.agent.replay_buffer import ReplayBuffer
    from tetris_rl.env.tetris_env import TetrisEnv
    import torch, random as _rnd, time

    global headless_stats, training_mode
    training_mode = "headless"
    cfg = TrainingConfig(episodes=episodes if not infinite else 1, device=device_str)  # episodes reused per cycle if infinite
    cfg.seed = seed
    rng = _rnd.Random(cfg.seed)
    # Probe obs dim
    probe_env = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
    probe_obs,_ = probe_env.reset()
    obs_dim = len(probe_obs)
    probe_env.close()
    device = _resolve_device(cfg.device)
    policy_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64,64])
    target_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64,64])
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device); target_net.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    global_step = 0
    episode_counter = 0
    reward_window: list[float] = []
    lines_window: list[int] = []
    comp_window: list[Dict[str, float]] = []
    start_time = time.time()
    import torch
    # Reduce CPU thread contention (often helps for small models)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    print(f"[Headless] Device: {device} opt_every={opt_every}", flush=True)
    try:
        while True:
            # Re-evaluate episodes target if infinite (loop forever)
            target_episodes = episodes if not infinite else episodes or 1
            for _ in range(target_episodes):
                env = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
                state,_ = env.reset()
                done = False; truncated = False
                ep_reward = 0.0; ep_lines = 0
                ep_line = 0.0
                ep_surv = 0.0
                ep_place = 0.0
                ep_top = 0.0
                while not (done or truncated):
                    state_t = torch.tensor(state, dtype=torch.float32, device=device)
                    action, meta = select_action(policy_net, state_t, global_step, cfg, 5)
                    next_state, reward, done, truncated, info = env.step(action)
                    buffer.push(state, action, reward, next_state, done or truncated)
                    if (global_step % opt_every) == 0:
                        loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
                    else:
                        loss_val = 0.0
                    if global_step % cfg.target_sync == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                    state = next_state
                    ep_reward += reward
                    ep_lines = info.get("lines_cleared_total", ep_lines)
                    rc = info.get("reward_components", {})
                    ep_line += rc.get("line_reward", 0.0)
                    ep_surv += rc.get("survival", 0.0)
                    ep_place += rc.get("placement", 0.0)
                    # top_out component only applied once at termination; accumulate anyway
                    ep_top += rc.get("top_out", 0.0)
                    global_step += 1
                env.close()
                episode_counter += 1
                reward_window.append(ep_reward)
                lines_window.append(ep_lines)
                comp_window.append({
                    "line_reward": ep_line,
                    "survival": ep_surv,
                    "placement": ep_place,
                    "top_out": ep_top,
                })
                if len(reward_window) > 100:
                    reward_window.pop(0); lines_window.pop(0)
                if len(comp_window) > 100:
                    comp_window.pop(0)
                # Rolling averages for components
                if comp_window:
                    avg_line = sum(c["line_reward"] for c in comp_window)/len(comp_window)
                    avg_surv = sum(c["survival"] for c in comp_window)/len(comp_window)
                    avg_place = sum(c["placement"] for c in comp_window)/len(comp_window)
                    avg_top = sum(c["top_out"] for c in comp_window)/len(comp_window)
                else:
                    avg_line = avg_surv = avg_place = avg_top = 0.0
                headless_stats.update({
                    "episodes": episode_counter,
                    "last_reward": ep_reward,
                    "last_lines": ep_lines,
                    "avg_reward": sum(reward_window)/len(reward_window),
                    "avg_lines": sum(lines_window)/len(lines_window),
                    "global_step": global_step,
                    "device": str(device),
                    "last_components": {
                        "line_reward": ep_line,
                        "survival": ep_surv,
                        "placement": ep_place,
                        "top_out": ep_top,
                    },
                    "avg_components": {
                        "line_reward": avg_line,
                        "survival": avg_surv,
                        "placement": avg_place,
                        "top_out": avg_top,
                    },
                })
                if (episode_counter % print_every) == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = episode_counter / max(1e-6, elapsed)
                    print(f"[Headless] ep={episode_counter} reward={ep_reward:.2f} lines={ep_lines} avgR={headless_stats['avg_reward']:.2f} avgL={headless_stats['avg_lines']:.2f} steps={global_step} eps/s={eps_per_sec:.2f}", flush=True)
            if not infinite:
                break
    except asyncio.CancelledError:
        print("[Headless] Cancelled", flush=True)
        raise
    except Exception as e:  # log and finish
        print(f"[Headless] Error: {e}", flush=True)
    finally:
        training_mode = "idle"


## Removed /api/resume endpoint (checkpointing disabled in minimal version)

## Removed /api/training-config endpoints (live exploration tuning disabled)

@app.get("/api/reward-config")
async def get_reward_config():
    return reward_config.to_dict()

@app.post("/api/reward-config")
async def update_reward_config(payload: Dict[str, Any]):
    # Validate numeric fields; ignore unknown keys
    updated = 0
    errors = {}
    for k, v in payload.items():
        if not hasattr(reward_config, k):
            errors[k] = "unknown"
            continue
        try:
            setattr(reward_config, k, float(v))
            updated += 1
        except (TypeError, ValueError):
            errors[k] = "not-a-float"
    # Broadcast new config to all clients
    await broadcast({"type": "config_update", "config": reward_config.to_dict()})
    return {"updated": updated, "errors": errors, "config": reward_config.to_dict()}

## Removed /api/model-config endpoints (architecture fixed in minimal version)

# Mount static files directory (will be created separately)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

