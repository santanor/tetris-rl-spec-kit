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
    global current_board

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

@app.post("/api/train")
async def start_training(episodes: int = 5, device: str = "auto"):
    """Start a minimal training session (no multi-env, no resume)."""
    global training_task
    if training_task and not training_task.done():
        return {"status": "already-running"}
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device))
    return {"status": "started", "episodes": episodes, "device": device}

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

