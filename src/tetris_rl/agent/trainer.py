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
    min_replay: int = 500
    seed: int = 0
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    # Model architecture (just hidden layer sizes for MLP body)
    hidden_layers: Optional[List[int]] = None

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128, 64]


def _compute_epsilon(step: int, cfg: TrainingConfig) -> float:
    """Simple exponential decay schedule."""
    import math
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / float(max(1, cfg.epsilon_decay)))


def select_action(policy_net: DQN, state: torch.Tensor, steps_done: int, cfg: TrainingConfig, num_actions: int) -> Tuple[int, Dict[str, Any]]:
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


def optimize(policy_net: DQN, target_net: DQN, buffer: ReplayBuffer, cfg: TrainingConfig, optimizer: optim.Optimizer, device: torch.device) -> float:
    if len(buffer) < cfg.min_replay:
        return 0.0
    states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)
    states = states.to(device).float()
    next_states = next_states.to(device).float()
    actions = actions.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + cfg.gamma * next_q * (1 - dones)
    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


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
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)

    session = Session(session_id="train", mode="train", seed=cfg.seed, config={})
    episodes_file = output_dir / "episodes.jsonl"

    global_step = 0
    for ep in range(cfg.episodes):
        env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps)
        state, _ = env.reset()
        episode_obj = Episode(index=ep)
        done = False
        truncated = False
        while not (done or truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            action, meta = select_action(policy_net, state_t, global_step, cfg, 5)
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
            state = next_state
            loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
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

