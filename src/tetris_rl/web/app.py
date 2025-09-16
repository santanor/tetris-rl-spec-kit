from __future__ import annotations

import asyncio
import json
from pathlib import Path
import os
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
current_sweep_task: Optional[asyncio.Task] = None

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

HEADLESS = os.getenv("HEADLESS", "") == "1"

async def broadcast(message: Dict[str, Any]):
    """Send a JSON-serializable message to all connected websocket clients.

    In headless mode (env var HEADLESS=1), this becomes a no-op so that
    training can run in CI or servers without incurring websocket overhead
    or requiring any active UI clients.
    """
    if HEADLESS:
        return
    if not websocket_clients:
        return
    data = json.dumps(message)
    to_remove: list[WebSocket] = []
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

async def training_loop(
    output_dir: Path,
    episodes: int,
    device_str: str = "auto",
    resume_from: Optional[str] = None,
    num_envs: int = 1,
    broadcast_every: int = 1,
    profile_every: int = 0,
):
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
    import math, random, torch, time
    from tetris_rl.agent.dqn import DQN
    from tetris_rl.agent.replay_buffer import ReplayBuffer
    from tetris_rl.agent.trainer import select_action, optimize, _resolve_device, _compute_epsilon, _compute_temperature

    device = _resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    _hidden = cfg.hidden_layers or [256, 256, 256]
    policy_net = DQN((4,), 5, hidden_layers=_hidden, dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
    target_net = DQN((4,), 5, hidden_layers=_hidden, dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
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

    # Track best performing episode (by reward)
    best_episode_reward: float = float('-inf')
    best_episode_meta: Dict[str, Any] = {}

    def _save_best_checkpoint(meta: Dict[str, Any]):
        """Persist a 'best model' snapshot with richer metadata so it can be moved to another machine.

        We include model architecture config, reward config and training overrides so that
        resuming or inspecting elsewhere has context. Replay buffer is NOT included (can be large).
        """
        try:
            import torch as _torch
            ckpt = {
                "type": "best_model",
                "model": policy_net.state_dict(),
                "target_model": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),  # optional (may not be used later)
                "global_step": global_step,
                "episode_index": meta.get("episode"),
                "episode_reward": meta.get("reward"),
                "lines_cleared": meta.get("line_clears"),
                "env_id": meta.get("env_id"),
                "board_snapshot": meta.get("board_snapshot"),
                "config": {
                    "model_overrides": model_overrides.copy(),
                    "training_overrides": training_overrides.copy(),
                    "reward_config": reward_config.to_dict(),
                    "hidden_layers": cfg.hidden_layers,
                    "dueling": cfg.dueling,
                    "use_layer_norm": cfg.use_layer_norm,
                    "dropout": cfg.dropout,
                },
            }
            best_path = static_checkpoint_dir / "dashboard_checkpoint_best.pt"
            _torch.save(ckpt, best_path)
            asyncio.create_task(broadcast({
                "type": "best_checkpoint_saved",
                "checkpoint": best_path.name,
                "episode": meta.get("episode"),
                "reward": meta.get("reward"),
                "lines_cleared": meta.get("line_clears"),
                "env_id": meta.get("env_id"),
            }))
        except Exception as e:  # log but continue
            asyncio.create_task(broadcast({"type": "best_checkpoint_error", "error": str(e)}))

    # ---------------- Multi-env synchronous loop ---------------- #
    num_envs = max(1, int(num_envs))
    broadcast_every = max(1, int(broadcast_every))
    profile_every = max(0, int(profile_every))
    from tetris_rl.utils.profiler import Profiler, SectionTimer
    profiler = Profiler(interval_steps=profile_every) if profile_every > 0 else None
    from tetris_rl.env.tetris_env import TetrisEnv as _TEnv
    envs: list[_TEnv] = []
    states: list[Any] = []
    episodes_done = [0]*num_envs  # episodes completed per env
    episode_objs: list[Episode] = []
    episode_rewards_current = [0.0]*num_envs
    episode_lineclears_current = [0]*num_envs
    # No longer tracking detailed structural shaping metrics after reward simplification.
    action_counts_list = [{"random":0, "greedy":0, "boltzmann":0} for _ in range(num_envs)]
    last_eps_list: list[Optional[float]] = [None]*num_envs
    last_temp_list: list[Optional[float]] = [None]*num_envs
    active_mask = [True]*num_envs
    # store most recent env step info for each env to extract hole column stats for broadcast
    env_last_info: list[Optional[dict]] = [None]*num_envs

    for i in range(num_envs):
        e = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
        s, _ = e.reset()
        envs.append(e)  # type: ignore[arg-type]
        states.append(s)
        episode_objs.append(Episode(index=0))

    # Preallocate contiguous numpy buffer for fast stacking (avoids slow per-step python list -> tensor path)
    import numpy as _np
    state_dim = len(states[0]) if states else 0
    state_buffer = _np.zeros((num_envs, state_dim), dtype=_np.float32)

    def _select_actions(batch_states_tensor):
        # Vectorized forward once, then per-env exploration.
        with torch.no_grad():
            q_batch = policy_net(batch_states_tensor)
        acts = []
        metas = []
        for idx in range(len(envs)):
            if not active_mask[idx]:
                acts.append(0)
                metas.append({})
                continue
            # Re-apply mutable exploration overrides each step for live tuning
            for _k in ("epsilon_start","epsilon_end","epsilon_decay","epsilon_schedule","epsilon_mid","epsilon_mid_step","exploration_strategy","temperature_start","temperature_end","temperature_decay"):
                if _k in training_overrides:
                    setattr(cfg, _k, training_overrides[_k])
            eps = float(_compute_epsilon(global_step, cfg))
            action_source = "greedy"
            temperature = None
            q_row = q_batch[idx]
            if cfg.exploration_strategy == "boltzmann":
                temperature = _compute_temperature(global_step, cfg)
                logits = q_row / temperature
                probs = torch.softmax(logits, dim=0)
                a = int(torch.multinomial(probs, 1).item())
                action_source = "boltzmann"
            else:
                if random.random() < eps:
                    a = random.randrange(5)
                    action_source = "random"
                else:
                    a = int(q_row.argmax().item())
                    action_source = "greedy"
            meta = {"epsilon": eps, "action_source": action_source}
            if temperature is not None:
                meta["temperature"] = float(temperature)
            acts.append(a)
            metas.append(meta)
        return acts, metas

    # Total episodes target refers to per-env episodes (consistent with prior semantics)
    while any(episodes_done[i] < cfg.episodes for i in range(num_envs)):
        # Build batch state tensor for active envs
        import torch as _torch
        # Populate preallocated numpy buffer
        for _i in range(num_envs):
            if active_mask[_i]:
                state_buffer[_i] = states[_i]
        batch_states = _torch.from_numpy(state_buffer).to(device)
        actions, metas = _select_actions(batch_states)
        # Step each env
        t_env_start = None
        if profiler:
            t_env_start = time.perf_counter()
        for i, env in enumerate(envs):
            if not active_mask[i]:
                continue
            action = actions[i]
            next_state, reward, done, truncated, info = env.step(action)
            env_last_info[i] = info
            buffer.push(states[i], action, reward, next_state, (done or truncated))
            episode_objs[i].record_step(reward, info.get("lines_delta",0), int(next_state[3]), int(next_state[2]))
            episode_rewards_current[i] += reward
            if info.get("lines_delta"):
                episode_lineclears_current[i] += info.get("lines_delta",0)
            # Structural accumulation for averages
            # Nothing to accumulate structurally now; could collect imbalance metrics per step if desired.
            # Exploration meta
            m = metas[i]
            src = m.get("action_source")
            if src in action_counts_list[i]:
                action_counts_list[i][src] += 1
            last_eps_list[i] = m.get("epsilon", last_eps_list[i])
            last_temp_list[i] = m.get("temperature", last_temp_list[i])
            states[i] = next_state
            if done or truncated:
                episode_objs[i].finalize(terminated=bool(done), truncated=bool(truncated), interrupted=False, reason=None)
                session_reward += episode_objs[i].total_reward
                episodes_done[i] += 1
                # Broadcast episode end for this env
                avg_struct = {}
                total_actions = sum(action_counts_list[i].values()) or 1
                action_dist = {k: v/total_actions for k,v in action_counts_list[i].items()}
                await broadcast({
                    "type": "episode_end",
                    "env_id": i,
                    "episode": episode_objs[i].index,
                    "reward": episode_objs[i].total_reward,
                    "line_clears": episode_lineclears_current[i],
                    "avg_structural": avg_struct,
                    "action_dist": action_dist,
                    "last_epsilon": last_eps_list[i],
                    "last_temperature": last_temp_list[i],
                })
                # Best model logic: track highest episode reward so far
                if episode_objs[i].total_reward > best_episode_reward:
                    best_episode_reward = episode_objs[i].total_reward
                    board_snap = envs[i].board.snapshot() if envs[i].board else None
                    best_episode_meta = {
                        "env_id": i,
                        "episode": episode_objs[i].index,
                        "reward": episode_objs[i].total_reward,
                        "line_clears": episode_lineclears_current[i],
                        "board_snapshot": board_snap,
                    }
                    _save_best_checkpoint(best_episode_meta)
                # Prepare next episode if still needed
                if episodes_done[i] < cfg.episodes:
                    env.close()
                    envs[i] = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=reward_config)
                    s2,_ = envs[i].reset()
                    states[i] = s2
                    episode_objs[i] = Episode(index=episodes_done[i])
                    episode_rewards_current[i] = 0.0
                    episode_lineclears_current[i] = 0
                    # reset placeholders
                    action_counts_list[i] = {"random":0, "greedy":0, "boltzmann":0}
                else:
                    active_mask[i] = False
        # Record env step time
        if profiler and t_env_start is not None:
            profiler.record("env_step", time.perf_counter() - t_env_start)

        # Optimize once per multi-env step
        t_opt_start = None
        if profiler:
            t_opt_start = time.perf_counter()
        loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
        if profiler and t_opt_start is not None:
            profiler.record("optimize", time.perf_counter() - t_opt_start)
        if global_step % cfg.target_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())
        global_step += 1

        # Periodic checkpoint
        if global_step > 0 and (global_step % 20 == 0):
            # Use best performing env's current episode index for metadata
            t_ckpt_start = None
            if profiler:
                t_ckpt_start = asyncio.get_event_loop().time()
            _save_dashboard_checkpoint(max(episodes_done) - 1 if max(episodes_done)>0 else 0, final=False)
            if profiler and t_ckpt_start is not None:
                profiler.record("checkpoint", asyncio.get_event_loop().time() - t_ckpt_start)

        # Broadcast (throttled)
        if (global_step % broadcast_every) == 0:
            # Determine best performer (highest current episode reward among active, else last finished)
            best_idx = None
            best_val = float('-inf')
            for i in range(num_envs):
                val = episode_rewards_current[i]
                if active_mask[i] and val >= best_val:
                    best_val = val
                    best_idx = i
            if best_idx is None:
                # fallback to env with highest total reward
                best_idx = max(range(num_envs), key=lambda i: episode_rewards_current[i])
            current_board = envs[best_idx].board if active_mask[best_idx] else envs[best_idx].board
            envs_payload = []
            for i in range(num_envs):
                envs_payload.append({
                    "id": i,
                    "episode": episode_objs[i].index,
                    "episode_reward": episode_rewards_current[i],
                    "lines_cleared": envs[i].board.lines_cleared_total,
                    "epsilon": last_eps_list[i],
                    "temperature": last_temp_list[i],
                    "active": active_mask[i],
                })
            t_broadcast_start = None
            if profiler:
                t_broadcast_start = asyncio.get_event_loop().time()
            # Gather hole column stats from best env if available
            best_info = env_last_info[best_idx] or {}
            rc_best = best_info.get("reward_components") or {}
            await broadcast({
                "type": "step",
                "global_step": global_step,
                "loss": loss_val,
                "best_env": best_idx,
                "best_board": current_board.snapshot() if current_board else None,
                "envs": envs_payload,
                "heights": current_board.heights() if current_board else [],
                "hole_columns": best_info.get("hole_columns"),
                "hole_column_penalty": rc_best.get("hole_column_penalty"),
            })
            if profiler and t_broadcast_start is not None:
                profiler.record("broadcast", asyncio.get_event_loop().time() - t_broadcast_start)

        # Profile report if due
        if profiler:
            rep = profiler.maybe_report(global_step)
            if rep:
                # Print concise console line and broadcast full payload
                interval = rep.get("interval", {})
                env_pct = interval.get("env_step", {}).get("pct", 0)
                opt_pct = interval.get("optimize", {}).get("pct", 0)
                print(f"[PROFILE] step={global_step} env_step={env_pct:.1f}% optimize={opt_pct:.1f}%")
                await broadcast(rep)
        await asyncio.sleep(0)

    # Final checkpoint & session end
    _save_dashboard_checkpoint(max(episodes_done)-1 if max(episodes_done)>0 else 0, final=True)
    total_episodes_all = sum(episodes_done)
    await broadcast({"type": "session_end", "episodes": total_episodes_all, "avg_reward": session_reward / float(total_episodes_all or 1)})
    # At session end also broadcast best summary if exists
    if best_episode_meta:
        await broadcast({"type": "best_summary", **best_episode_meta})

@app.post("/api/train")
async def start_training(episodes: int = 5, device: str = "auto", num_envs: int = 1, broadcast_every: int = 1, profile_every: int = 0):
    global training_task
    if training_task and not training_task.done():
        return {"status": "already-running"}
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device, None, num_envs, broadcast_every, profile_every))
    return {"status": "started", "episodes": episodes, "device": device, "num_envs": num_envs, "broadcast_every": broadcast_every, "profile_every": profile_every}

@app.post("/api/resume")
async def resume_training(checkpoint: str, episodes: int = 5, device: str = "auto", num_envs: int = 1, broadcast_every: int = 1, profile_every: int = 0):
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
    elif checkpoint == "best":
        ckpt_path = static_checkpoint_dir / "dashboard_checkpoint_best.pt"
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
    training_task = asyncio.create_task(training_loop(Path("training_runs/dashboard"), episodes, device, str(ckpt_path), num_envs, broadcast_every, profile_every))
    return {"status": "resuming", "from": str(ckpt_path), "episodes": episodes, "device": device, "num_envs": num_envs, "broadcast_every": broadcast_every, "profile_every": profile_every}

@app.get("/api/best-checkpoint")
async def get_best_checkpoint():
    """Return metadata of best checkpoint if it exists (does not return raw weights)."""
    best_path = Path(__file__).parent / "static" / "dashboard_checkpoint_best.pt"
    if not best_path.is_file():
        return {"exists": False}
    try:
        import torch
        loaded = torch.load(best_path, map_location="cpu")
        meta = {
            "exists": True,
            "checkpoint": best_path.name,
            "episode": loaded.get("episode_index"),
            "reward": loaded.get("episode_reward"),
            "lines_cleared": loaded.get("lines_cleared"),
            "env_id": loaded.get("env_id"),
            "board_snapshot": loaded.get("board_snapshot"),
            "global_step": loaded.get("global_step"),
            "config": loaded.get("config"),
        }
        return meta
    except Exception as e:
        return {"exists": False, "error": str(e)}

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

@app.post("/api/sweep")
async def start_reward_sweep(trials: int = 30, episodes_stage1: int = 40, episodes_stage2: int = 120, seeds_stage1: int = 1, seeds_stage2: int = 2, topk: int = 5, device: str = "auto"):
    """Launch a background Bayesian reward sweep (Optuna) using same architecture overrides.

    Returns immediately with a status token; progress is broadcast via websocket messages:
      - sweep_progress
      - sweep_complete
    Artifacts saved under sweep_artifacts/ .
    """
    global current_sweep_task
    if current_sweep_task and not current_sweep_task.done():
        return {"status": "already-running"}

    async def _run_sweep():
        import optuna, random, statistics, json, math, torch
        from dataclasses import asdict
        from tetris_rl.env.reward_config import RewardConfig
        from tetris_rl.agent.trainer import optimize, _compute_epsilon, _compute_temperature, _resolve_device
        from tetris_rl.agent.dqn import DQN
        from tetris_rl.agent.replay_buffer import ReplayBuffer
        from tetris_rl.env.tetris_env import TetrisEnv
        from tetris_rl.agent.trainer import TrainingConfig
        out_dir = Path("sweep_artifacts")
        out_dir.mkdir(exist_ok=True)

        hidden = model_overrides.get("hidden_layers") or TrainingConfig().hidden_layers or [256,256,256]
        dueling = model_overrides.get("dueling", TrainingConfig().dueling)
        use_ln = model_overrides.get("use_layer_norm", TrainingConfig().use_layer_norm)
        dropout = model_overrides.get("dropout", TrainingConfig().dropout)

        def suggest(trial: optuna.Trial) -> RewardConfig:
            rc = RewardConfig()
            rc.line_reward_1 = trial.suggest_float("line_reward_1", 0.2, 2.0)
            rc.line_reward_2 = trial.suggest_float("line_reward_2", 0.5, 4.0)
            rc.line_reward_3 = trial.suggest_float("line_reward_3", 1.0, 6.0)
            rc.line_reward_4 = trial.suggest_float("line_reward_4", 2.0, 10.0)
            rc.step_penalty = trial.suggest_float("step_penalty", -0.02, 0.0)
            rc.imbalance_penalty_scale = trial.suggest_float("imbalance_penalty_scale", 0.0, 0.2)
            rc.imbalance_penalty_power = trial.suggest_float("imbalance_penalty_power", 1.0, 2.5)
            rc.top_out_penalty = trial.suggest_float("top_out_penalty", -15.0, -2.0)
            rc.imbalance_mode = trial.suggest_categorical("imbalance_mode", ["sum", "max"])
            return rc

        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=1234)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

        def eval_once(rc: RewardConfig, episodes: int, seed: int) -> float:
            cfg = TrainingConfig(episodes=episodes, device=device, hidden_layers=hidden, dueling=dueling, use_layer_norm=use_ln, dropout=dropout)
            dev = _resolve_device(cfg.device)
            rng = random.Random(seed)
            policy = DQN((4,),5, hidden_layers=hidden, dueling=dueling, use_layer_norm=use_ln, dropout=dropout).to(dev)
            target = DQN((4,),5, hidden_layers=hidden, dueling=dueling, use_layer_norm=use_ln, dropout=dropout).to(dev)
            target.load_state_dict(policy.state_dict())
            opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
            replay = ReplayBuffer(cfg.replay_capacity, seed)
            global_step = 0
            ep_rewards = []
            for ep in range(episodes):
                env = TetrisEnv(seed=rng.randint(0,1_000_000), max_steps=cfg.max_steps, reward_config=rc)
                state,_ = env.reset()
                done=False; truncated=False; total=0.0
                while not (done or truncated):
                    eps = float(_compute_epsilon(global_step, cfg))
                    if random.random() < eps:
                        a = random.randrange(5)
                    else:
                        with torch.no_grad():
                            q = policy(torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0))
                        a = int(q.argmax().item())
                    ns, r, done, truncated, info = env.step(a)
                    replay.push(state, a, r, ns, (done or truncated))
                    state = ns
                    total += r
                    _ = optimize(policy, target, replay, cfg, opt, dev)
                    global_step += 1
                    if global_step % cfg.target_sync == 0:
                        target.load_state_dict(policy.state_dict())
                ep_rewards.append(total)
                env.close()
            return sum(ep_rewards)/len(ep_rewards)

        def objective(trial: optuna.Trial):
            rc = suggest(trial)
            seeds = [9000 + i for i in range(seeds_stage1)]
            scores = []
            for idx,s in enumerate(seeds):
                val = eval_once(rc, episodes_stage1, s)
                scores.append(val)
                trial.report(sum(scores)/len(scores), step=idx+1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mean_score = sum(scores)/len(scores)
            asyncio.create_task(broadcast({"type":"sweep_progress","trial":trial.number,"score":mean_score}))
            return mean_score

        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        # Stage2 re-eval topK
        complete = [t for t in study.trials if t.value is not None and t.state.name=='COMPLETE']
        complete.sort(key=lambda t: t.value, reverse=True)
        finalists = complete[:topk]
        stage2 = []
        for t in finalists:
            rc = RewardConfig()
            for k,v in t.params.items():
                if hasattr(rc,k):
                    setattr(rc,k,v)
            seeds2 = [12000+i for i in range(seeds_stage2)]
            vals = [eval_once(rc, episodes_stage2, s) for s in seeds2]
            stage2.append({"trial": t.number, "stage1": t.value, "stage2_mean": sum(vals)/len(vals), "params": t.params})
        stage2.sort(key=lambda x: x['stage2_mean'], reverse=True)
        with (out_dir/"sweep_results.json").open('w') as f:
            json.dump({"stage1_best": [ {"trial":t.number, "value":t.value, "params":t.params} for t in finalists ], "stage2": stage2}, f, indent=2)
        asyncio.create_task(broadcast({"type":"sweep_complete","best": stage2[0] if stage2 else None}))

    current_sweep_task = asyncio.create_task(_run_sweep())
    return {"status": "started", "trials": trials}

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
        current = getattr(reward_config, k)
        if isinstance(current, float):
            try:
                setattr(reward_config, k, float(v))
                updated += 1
            except (TypeError, ValueError):
                errors[k] = "not-a-float"
        else:
            if isinstance(v, str):
                setattr(reward_config, k, v)
                updated += 1
            else:
                errors[k] = "invalid-type"
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

