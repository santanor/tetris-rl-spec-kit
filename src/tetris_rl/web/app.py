from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from tetris_rl.agent.trainer import run_training, TrainingConfig
from tetris_rl.core.board import TetrisBoard
from tetris_rl.env.reward_config import RewardConfig

app = FastAPI(title="Tetris RL Dashboard")

# Simple in-memory state
latest_step: Dict[str, Any] = {}
current_board: Optional[TetrisBoard] = None
session_stats: Dict[str, Any] = {"episodes": 0, "total_reward": 0.0}
websocket_clients: List[WebSocket] = []
training_task: Optional[asyncio.Task] = None
reward_config = RewardConfig()  # shared mutable config
training_overrides: Dict[str, Any] = {}  # holds mutable training hyperparameters (epsilon / temperature / exploration)
# Model config overrides (applied to new DQN instantiation each session; not hot-swapped mid-episode)
model_overrides: Dict[str, Any] = {}

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

async def training_loop(output_dir: Path, episodes: int, device_str: str = "auto", resume_from: Optional[str] = None):
    global current_board
    from tetris_rl.env.tetris_env import TetrisEnv
    from tetris_rl.models.episode import Episode

    cfg = TrainingConfig(episodes=episodes, device=device_str)
    # Apply model overrides early (architecture) - these are fixed for the session
    for k in ("hidden_layers","dueling","use_layer_norm","dropout"):
        if k in model_overrides:
            setattr(cfg, k, model_overrides[k])
    # Apply any mutable overrides (currently epsilon parameters)
    for key in ("epsilon_start", "epsilon_end", "epsilon_decay"):
        if key in training_overrides:
            setattr(cfg, key, training_overrides[key])
    # local mutable holders for stats
    session_reward = 0.0

    def on_step(step_info: Dict[str, Any]):
        # Called from training thread (sync). Offload to loop via create_task.
        asyncio.get_event_loop().create_task(broadcast({"type": "step", **step_info}))

    # We'll run a light-weight manual loop replicating run_training but exposing board each step.
    import math, random, torch
    from tetris_rl.agent.dqn import DQN
    from tetris_rl.agent.replay_buffer import ReplayBuffer
    from tetris_rl.agent.trainer import select_action, optimize, _resolve_device

    device = _resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    # Probe environment once for observation dimension (dynamic features)
    _tmp_env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
    _probe_obs, _ = _tmp_env.reset()
    obs_dim = int(len(_probe_obs))
    _tmp_env.close()
    _hidden = cfg.hidden_layers or [256, 256, 256]
    policy_net = DQN((obs_dim,), 5, hidden_layers=_hidden, dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
    target_net = DQN((obs_dim,), 5, hidden_layers=_hidden, dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    global_step = 0

    # ----- Resume logic -----
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.is_file():
            try:
                loaded = torch.load(ckpt_path, map_location="cpu")
                # If checkpoint has config, prefer its architecture to avoid shape mismatch
                loaded_cfg = loaded.get("config", {})
                if loaded_cfg:
                    arch_hidden = loaded_cfg.get("hidden_layers", cfg.hidden_layers)
                    arch_dueling = loaded_cfg.get("dueling", cfg.dueling)
                    arch_ln = loaded_cfg.get("use_layer_norm", cfg.use_layer_norm)
                    arch_do = loaded_cfg.get("dropout", cfg.dropout)
                    # Rebuild networks with loaded architecture (ensure list copy)
                    policy_net.__init__((4,), 5, hidden_layers=arch_hidden, dueling=arch_dueling, use_layer_norm=arch_ln, dropout=arch_do)  # type: ignore
                    target_net.__init__((4,), 5, hidden_layers=arch_hidden, dueling=arch_dueling, use_layer_norm=arch_ln, dropout=arch_do)  # type: ignore
                policy_net.load_state_dict(loaded.get("model", {}))
                target_net.load_state_dict(loaded.get("target_model", {}))
                try:
                    optimizer.load_state_dict(loaded.get("optimizer", {}))
                except Exception:
                    pass
                # Replay buffer restore
                rep = loaded.get("replay")
                if rep:
                    buffer.capacity = rep.get("capacity", buffer.capacity)
                    buffer.memory = rep.get("memory", buffer.memory)
                    buffer.position = rep.get("position", buffer.position)
                global_step = int(loaded.get("global_step", 0))
                sess_meta = loaded.get("session_meta", {})
                await broadcast({
                    "type": "resume_info",
                    "checkpoint": ckpt_path.name,
                    "prior_episodes": sess_meta.get("episodes_total"),
                    "prior_reward": sess_meta.get("total_reward"),
                    "prior_lines": sess_meta.get("total_lines_cleared"),
                    "global_step": global_step,
                })
            except Exception as e:
                await broadcast({"type": "resume_error", "error": str(e)})
        else:
            await broadcast({"type": "resume_error", "error": f"Checkpoint not found: {resume_from}"})

    policy_net.to(device)
    target_net.to(device)
    if not resume_from:
        target_net.load_state_dict(policy_net.state_dict())
    output_dir.mkdir(parents=True, exist_ok=True)

    # If resuming, continue episode numbering after prior episodes (best-effort if metadata broadcast earlier)
    base_episode = 0
    if resume_from:
        # Use prior episodes count if sent via resume_info; else 0
        # (We don't have persistent shared session object in this lightweight loop)
        pass
    # Directory to place lightweight dashboard checkpoints (served alongside static assets)
    static_checkpoint_dir = Path(__file__).parent / "static"
    static_checkpoint_dir.mkdir(exist_ok=True)

    def _save_dashboard_checkpoint(ep_index: int, final: bool = False):
        """Save (overwrite) a lightweight checkpoint in the static dir every N steps or final.

        We keep only a single file: dashboard_checkpoint_latest.pt
        """
        try:
            import torch as _torch
            ckpt = {
                "model": policy_net.state_dict(),
                "target_model": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "episode_index": ep_index,
                "final": final,
            }
            latest = static_checkpoint_dir / "dashboard_checkpoint_latest.pt"
            _torch.save(ckpt, latest)
            asyncio.create_task(broadcast({
                "type": "checkpoint_saved",
                "checkpoint": latest.name,
                "episode": ep_index,
                "global_step": global_step,
                "final": final
            }))
        except Exception as e:  # log but continue training
            asyncio.create_task(broadcast({"type": "checkpoint_error", "error": str(e)}))

    for ep_idx in range(base_episode, base_episode + cfg.episodes):
        # Pass shared reward_config so runtime changes propagate to new episodes
        env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
        state, _ = env.reset()
        episode = Episode(index=ep_idx)
        line_clears_this_ep = 0
        structural_sum = {"holes":0.0, "height":0.0, "bumpiness":0.0}
        structural_steps = 0
        action_counts = {"random":0, "greedy":0, "boltzmann":0}
        last_eps = None
        last_temp = None
        done = False
        truncated = False
        while not (done or truncated):
            # Access underlying board through env
            current_board = env.board
            import torch as _torch
            state_t = _torch.tensor(state, dtype=_torch.float32, device=device)
            # Re-apply mutable exploration overrides each step for live tuning
            for _k in ("epsilon_start","epsilon_end","epsilon_decay","epsilon_schedule","epsilon_mid","epsilon_mid_step","exploration_strategy","temperature_start","temperature_end","temperature_decay"):
                if _k in training_overrides:
                    setattr(cfg, _k, training_overrides[_k])
            action, meta = select_action(policy_net, state_t, global_step, cfg, 5)
            # Track exploration meta
            src = meta.get("action_source", "?")
            if src in action_counts:
                action_counts[src] += 1
            last_eps = meta.get("epsilon", last_eps)
            last_temp = meta.get("temperature", last_temp)
            next_state, reward, done, truncated, info = env.step(action)
            buffer.push(state, action, reward, next_state, done or truncated)
            episode.record_step(reward, info.get("lines_delta", 0), int(next_state[3]), int(next_state[2]))
            state = next_state
            loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
            if global_step % cfg.target_sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            global_step += 1
            # Network introspection (q-values + activations) for UI
            try:
                q_vals, acts = policy_net.forward_with_activations(state_t.unsqueeze(0))  # type: ignore[attr-defined]
            except Exception:
                q_vals, acts = [], {"layers": [], "advantages": None, "value": None}
            net_meta = {
                "input_dim": obs_dim,
                "hidden_layers": _hidden,
                "num_actions": 5,
                "dueling": cfg.dueling,
            }
            # broadcast
            step_payload = {
                "episode": ep_idx,
                "global_step": global_step,
                "reward": reward,
                "loss": loss_val,
                "lines_cleared_total": info.get("lines_cleared_total"),
                "lines_delta": info.get("lines_delta"),
                "board": current_board.snapshot() if current_board else None,
                "reward_components": info.get("reward_components"),
                "epsilon": last_eps,
                "temperature": last_temp,
                "action_source": src,
                "q_values": q_vals,
                "net_meta": net_meta,
                "net_activations": acts,
                "heights": current_board.heights() if current_board else None,
                "hole_columns": current_board.hole_columns() if current_board else None,
            }
            if info.get("lines_delta"):
                line_clears_this_ep += info.get("lines_delta",0)
            rc = info.get("reward_components") or {}
            sb = rc.get("structural_breakdown") or {}
            for k in structural_sum:
                structural_sum[k] += float(sb.get(k,0.0))
            structural_steps += 1
            await broadcast({"type": "step", **step_payload})
            # Save checkpoint every 20 global steps (and not on step 0)
            if global_step > 0 and (global_step % 20 == 0):
                _save_dashboard_checkpoint(ep_idx, final=False)
            await asyncio.sleep(0)  # yield control
        episode.finalize(terminated=bool(done), truncated=bool(truncated), interrupted=False, reason=None)
        session_reward += episode.total_reward
        env.close()
        avg_struct = {k: (v/structural_steps if structural_steps else 0.0) for k,v in structural_sum.items()}
        # Aggregate action source distribution
        total_actions = sum(action_counts.values()) or 1
        action_dist = {k: v/total_actions for k,v in action_counts.items()}
        await broadcast({
            "type": "episode_end",
            "episode": ep_idx,
            "reward": episode.total_reward,
            "line_clears": line_clears_this_ep,
            "avg_structural": avg_struct,
            "action_dist": action_dist,
            "last_epsilon": last_eps,
            "last_temperature": last_temp,
        })
    # (No per-episode checkpoint; handled by step cadence)

    # Final checkpoint
    _save_dashboard_checkpoint(base_episode + cfg.episodes - 1, final=True)
    await broadcast({"type": "session_end", "episodes": cfg.episodes, "avg_reward": session_reward / cfg.episodes})

@app.post("/api/train")
async def start_training(episodes: int = 5, device: str = "auto"):
    global training_task
    if training_task and not training_task.done():
        return {"status": "already-running"}
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device))
    return {"status": "started", "episodes": episodes, "device": device}

@app.post("/api/resume")
async def resume_training(checkpoint: str, episodes: int = 5, device: str = "auto"):
    """Resume training from a checkpoint path (relative or absolute).

    Parameters:
      checkpoint: path to a checkpoint_XXXXX.pt or checkpoint_latest.pt
      episodes: additional episodes to run
      device: target device (auto|cpu|cuda)
    """
    global training_task
    if training_task and not training_task.done():
        return {"status": "already-running"}
    # Provide alias 'latest' to static latest dashboard checkpoint
    static_checkpoint_dir = Path(__file__).parent / "static"
    if checkpoint == "latest":
        ckpt_path = static_checkpoint_dir / "dashboard_checkpoint_latest.pt"
    else:
        ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        # Try run dir
        alt = Path("training_runs/dashboard") / checkpoint
        if alt.is_file():
            ckpt_path = alt
        else:
            alt2 = static_checkpoint_dir / checkpoint
            if alt2.is_file():
                ckpt_path = alt2
            else:
                return {"status": "error", "error": f"Checkpoint not found: {checkpoint}"}
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device, resume_from=str(ckpt_path)))
    return {"status": "resuming", "from": str(ckpt_path), "episodes": episodes, "device": device}

@app.get("/api/training-config")
async def get_training_config():
    default_fields = [
        "epsilon_start","epsilon_end","epsilon_decay","epsilon_schedule","epsilon_mid","epsilon_mid_step",
        "exploration_strategy","temperature_start","temperature_end","temperature_decay"
    ]
    return {
        "overrides": training_overrides,
        "defaults": {k: getattr(TrainingConfig(), k) for k in default_fields},
    }

@app.post("/api/training-config")
async def update_training_config(payload: Dict[str, Any]):
    changed = {}
    numeric_fields = {
        "epsilon_start": float,
        "epsilon_end": float,
        "epsilon_decay": int,
        "epsilon_mid": float,
        "epsilon_mid_step": int,
        "temperature_start": float,
        "temperature_end": float,
        "temperature_decay": int,
    }
    str_fields = {"epsilon_schedule", "exploration_strategy"}
    for k, v in payload.items():
        if k in numeric_fields:
            try:
                training_overrides[k] = numeric_fields[k](v)
                changed[k] = training_overrides[k]
            except (TypeError, ValueError):
                continue
        elif k in str_fields:
            if isinstance(v, str):
                training_overrides[k] = v
                changed[k] = v
    await broadcast({"type": "training_config_update", "overrides": training_overrides})
    return {"updated": changed, "overrides": training_overrides}

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

@app.get("/api/model-config")
async def get_model_config():
    defaults = {
        "hidden_layers": TrainingConfig().hidden_layers,
        "dueling": TrainingConfig().dueling,
        "use_layer_norm": TrainingConfig().use_layer_norm,
        "dropout": TrainingConfig().dropout,
    }
    # Represent hidden layers as comma string for frontend convenience as well
    return {
        "overrides": model_overrides,
        "defaults": defaults,
        "hidden_layers_csv": ",".join(str(x) for x in (model_overrides.get("hidden_layers") or defaults["hidden_layers"]))
    }

@app.post("/api/model-config")
async def update_model_config(payload: Dict[str, Any]):
    changed = {}
    # hidden_layers can come as list or comma string
    if "hidden_layers" in payload:
        hl_val = payload["hidden_layers"]
        layers: List[int] = []
        if isinstance(hl_val, str):
            for part in hl_val.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    layers.append(int(part))
                except ValueError:
                    continue
        elif isinstance(hl_val, list):
            for item in hl_val:
                try:
                    layers.append(int(item))
                except (TypeError, ValueError):
                    continue
        if layers:
            model_overrides["hidden_layers"] = layers
            changed["hidden_layers"] = layers
    for flag_key in ("dueling","use_layer_norm"):
        if flag_key in payload:
            val = payload[flag_key]
            if isinstance(val, bool):
                model_overrides[flag_key] = val
                changed[flag_key] = val
            elif isinstance(val, str) and val.lower() in ("true","false"):
                b = val.lower() == "true"
                model_overrides[flag_key] = b
                changed[flag_key] = b
    if "dropout" in payload:
        try:
            dp = float(payload["dropout"])
            if 0.0 <= dp < 1.0:
                model_overrides["dropout"] = dp
                changed["dropout"] = dp
        except (TypeError, ValueError):
            pass
    await broadcast({"type": "model_config_update", "overrides": model_overrides})
    return {"updated": changed, "overrides": model_overrides}

# Mount static files directory (will be created separately)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

