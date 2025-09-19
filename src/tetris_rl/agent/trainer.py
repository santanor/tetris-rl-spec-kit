"""Minimal single‑environment DQN training loop (all legacy complexity removed).

This module intentionally excludes: multi‑env batching, checkpoint/resume, exploration
variants (Boltzmann, multi‑stage schedules), model architecture flags (dueling, layer
norm, dropout) and live hyperparameter mutation. It is a small, readable reference
for how the rest of the demo wires together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
# Use modern torch.amp APIs (cuda.amp is deprecated in recent PyTorch)

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.agent.dqn import DQN
from tetris_rl.agent.replay_buffer import ReplayBuffer
from tetris_rl.models.session import Session
from tetris_rl.models.episode import Episode


@dataclass
class TrainingConfig:
    # Core loop
    episodes: int = 20
    max_steps: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    # Epsilon‑greedy (simple exponential decay only)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 300  # Option E: faster decay to exploit structural signals sooner
    # Infrastructure
    target_sync: int = 200
    replay_capacity: int = 20_000
    min_replay: int = 5000
    seed: int = 0
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    # Model architecture (just hidden layer sizes for MLP body)
    hidden_layers: Optional[List[int]] = None

    # Early stopping (optional; disabled by default)
    early_stop_enable: bool = False
    early_stop_metric: str = "avg_reward"  # "avg_reward" | "avg_lines"
    early_stop_window: int = 10            # compute rolling average over last N episodes
    early_stop_min_delta: float = 0.01     # improvement threshold to reset patience
    early_stop_patience: int = 3           # stop after K non-improving windows
    # GPU/throughput tuning
    use_amp: bool = True                   # mixed precision on CUDA
    opt_steps_per_env_step: int = 4        # how many optimizer steps per environment step
    grad_clip_norm: Optional[float] = 10.0 # clip global grad norm (None disables)
    compile_model: bool = False            # torch.compile() for policy/target nets (PyTorch 2+)
    pin_memory: bool = True                # pin CPU memory for faster H2D copies

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 32]


def _compute_epsilon(step: int, cfg: TrainingConfig) -> float:
    """Simple exponential decay schedule."""
    import math
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / float(max(1, cfg.epsilon_decay)))


def select_action(policy_net: nn.Module, state: torch.Tensor, steps_done: int, cfg: TrainingConfig, num_actions: int) -> Tuple[int, Dict[str, Any]]:
    eps = _compute_epsilon(steps_done, cfg)
    if random.random() < eps:
        action = random.randrange(num_actions)
        src = "random"
    else:
        with torch.no_grad():
            q = policy_net(state.unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        src = "greedy"
    return action, {"epsilon": float(eps), "action_source": src}


def optimize(policy_net: nn.Module, target_net: nn.Module, buffer: ReplayBuffer, cfg: TrainingConfig, optimizer: optim.Optimizer, device: torch.device, scaler: Optional[Any] = None) -> float:
    # Require enough samples to draw a full batch safely
    if len(buffer) < max(int(cfg.min_replay), int(cfg.batch_size)):
        return 0.0
    states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)
    # Optional pinning for faster H2D when using CUDA
    if device.type == 'cuda' and cfg.pin_memory:
        try:
            states = states.pin_memory(); next_states = next_states.pin_memory()
            actions = actions.pin_memory(); rewards = rewards.pin_memory(); dones = dones.pin_memory()
        except Exception:
            pass
    states = states.to(device, non_blocking=True).float()
    next_states = next_states.to(device, non_blocking=True).float()
    actions = actions.to(device, non_blocking=True)
    rewards = rewards.to(device, non_blocking=True)
    dones = dones.to(device, non_blocking=True)

    use_amp = (device.type == 'cuda' and cfg.use_amp)
    optimizer.zero_grad(set_to_none=True)
    if use_amp and scaler is not None:
        # Modern autocast API (ignore type checker stub gaps)
        with torch.amp.autocast('cuda', dtype=torch.float16):  # type: ignore[attr-defined]
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + cfg.gamma * next_q * (1 - dones)
            loss = nn.functional.smooth_l1_loss(q_values, target)
        # Unscale before clipping
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0]
            target = rewards + cfg.gamma * next_q * (1 - dones)
        loss = nn.functional.smooth_l1_loss(q_values, target)
        loss.backward()
        if cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.grad_clip_norm)
        optimizer.step()
    return float(loss.detach().item())


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            print("[Training] Requested cuda but not available; falling back to cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def run_training(output_dir: Path, cfg: TrainingConfig | None = None, step_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Session:
    """Run a minimal single‑environment DQN training session.

    Artifacts:
      - policy_net.pt        (final weights)
      - episodes.jsonl       (one summary JSON per episode)
      - training_summary.txt (aggregate stats)
    """
    cfg = cfg or TrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    print("[Training] Device:", device)
    # Prefer faster matmul kernels if supported (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision('high')  # type: ignore[attr-defined]
    except Exception:
        pass

    rng = random.Random(cfg.seed)
    # Probe environment once to derive observation dimension dynamically
    probe_env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps)
    probe_obs, _ = probe_env.reset()
    obs_dim = len(probe_obs)
    probe_env.close()

    policy_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64, 64])
    target_net = DQN((obs_dim,), 5, hidden_layers=cfg.hidden_layers or [64, 64])
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device)
    target_net.to(device)
    # Execution modules (possibly compiled) for forward passes
    policy_exec: nn.Module = policy_net
    target_exec: nn.Module = target_net
    if cfg.compile_model:
        try:
            policy_exec = torch.compile(policy_net)  # type: ignore[attr-defined]
            target_exec = torch.compile(target_net)  # type: ignore[attr-defined]
        except Exception:
            policy_exec = policy_net
            target_exec = target_net
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    # Prefer the new GradScaler API
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg.use_amp))  # type: ignore[attr-defined]
    except Exception:
        scaler = None
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)

    session = Session(session_id="train", mode="train", seed=cfg.seed, config={})
    episodes_file = output_dir / "episodes.jsonl"

    global_step = 0
    # Early stopping trackers
    rewards_history: List[float] = []
    lines_history: List[int] = []
    best_window_avg: Optional[float] = None
    non_improve_windows = 0
    for ep in range(cfg.episodes):
        env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps)
        state, _ = env.reset()
        episode_obj = Episode(index=ep)
        done = False
        truncated = False
        ep_total_reward = 0.0
        while not (done or truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            action, meta = select_action(policy_exec, state_t, global_step, cfg, 5)
            next_state, reward, done, truncated, info = env.step(action)
            buffer.push(state, action, reward, next_state, done or truncated)
            # record structural metrics if exposed
            sf = info.get("state_features") or {}
            episode_obj.record_step(
                reward,
                info.get("lines_delta", 0),
                holes=int(sf.get("holes", 0)),
                height=int(sf.get("aggregate_height", 0)),
            )
            ep_total_reward += float(reward)
            state = next_state
            # Potentially perform multiple optimizer steps per env step to increase GPU utilization
            loss_val = 0.0
            for _ in range(max(1, int(cfg.opt_steps_per_env_step))):
                loss_val = optimize(policy_exec, target_exec, buffer, cfg, optimizer, device, scaler)
            if global_step % cfg.target_sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            global_step += 1
            if step_callback:
                try:
                    step_callback({
                        "episode": ep,
                        "global_step": global_step,
                        "reward": reward,
                        "loss": loss_val,
                        "lines_cleared_total": info.get("lines_cleared_total"),
                        "lines_delta": info.get("lines_delta"),
                        "epsilon": meta.get("epsilon"),
                        "action_source": meta.get("action_source"),
                        "done": done,
                        "truncated": truncated,
                    })
                except Exception as e:  # pragma: no cover - defensive
                    print("[Training] step_callback error:", e)
        episode_obj.finalize(terminated=bool(done), truncated=bool(truncated), interrupted=False, reason="top_out" if done else None)
        session.record_episode(episode_obj.total_reward, episode_obj.lines_cleared, episode_obj)
        # Early stopping bookkeeping (per-episode)
        rewards_history.append(ep_total_reward)
        lines_history.append(episode_obj.lines_cleared)
        if cfg.early_stop_enable:
            metric_series: List[float]
            if cfg.early_stop_metric == "avg_lines":
                metric_series = [float(x) for x in lines_history]
            else:
                metric_series = [float(x) for x in rewards_history]
            win = max(1, int(cfg.early_stop_window))
            if len(metric_series) >= win:
                cur_avg = sum(metric_series[-win:]) / float(win)
                if best_window_avg is None or (cur_avg - best_window_avg) > cfg.early_stop_min_delta:
                    best_window_avg = cur_avg
                    non_improve_windows = 0
                else:
                    non_improve_windows += 1
                # Stop if patience exceeded
                if non_improve_windows >= max(1, int(cfg.early_stop_patience)):
                    try:
                        (output_dir / "early_stop.txt").write_text(
                            f"Stopped early at episode {ep+1}\nBest {cfg.early_stop_metric} (window {win}): {best_window_avg:.4f}\n")
                    except Exception:
                        pass
                    break
        # Append summary line
        try:
            import json
            with episodes_file.open("a") as f:
                f.write(json.dumps(episode_obj.summary_dict()) + "\n")
        except Exception as e:  # pragma: no cover
            print("[Warn] could not write episode summary:", e)
        env.close()

    session.finish()
    (output_dir / "training_summary.txt").write_text(
        f"Episodes: {session.episodes_total}\nAvg reward: {session.avg_reward:.3f}\nAvg lines: {session.avg_lines_cleared:.3f}\n")
    torch.save(policy_net.state_dict(), output_dir / "policy_net.pt")
    return session

