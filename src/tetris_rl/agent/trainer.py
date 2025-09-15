"""Training loop orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Tuple, List
from pathlib import Path
import math
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
    episodes: int = 20  # Number of episodes to run (additive when resuming)
    max_steps: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    # --- Exploration parameters ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500  # generic scale parameter for schedule
    # Optional mid-stage parameters (used by two_stage schedule)
    epsilon_mid: float = 0.2
    epsilon_mid_step: int = 2000
    epsilon_schedule: str = "exp"  # exp | linear | two_stage | cosine
    exploration_strategy: str = "epsilon_greedy"  # epsilon_greedy | boltzmann
    # Boltzmann (softmax) temperature schedule
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay: int = 2000
    # --- Infrastructure ---
    target_sync: int = 200
    replay_capacity: int = 20_000
    min_replay: int = 500
    seed: int = 0
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    # --- Checkpointing / Resume ---
    checkpoint_every_episodes: int = 5  # 0 disables periodic checkpoints
    resume_from: Optional[str] = None  # Path to checkpoint (.pt) to resume from
    keep_last_n_checkpoints: int = 3  # simple rotation (0 => keep all)
    # --- Model Architecture (breaking change) ---
    hidden_layers: Optional[List[int]] = None  # will be set in __post_init__ to default
    dueling: bool = True
    use_layer_norm: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_layers is None:
            # Default deeper network now
            self.hidden_layers = [256, 256, 256]


def _compute_epsilon(step: int, cfg: TrainingConfig) -> float:
    """Compute epsilon according to configured schedule.

    Schedules:
      exp:     eps_end + (eps_start-eps_end)*exp(-step/decay)
      linear:  linear interpolation over epsilon_decay steps
      cosine:  cos annealing from start to end over epsilon_decay steps
      two_stage: decay from start->mid over epsilon_mid_step (exp form), then mid->end over remaining scale (epsilon_decay)
    """
    s = max(step, 0)
    es, ee = cfg.epsilon_start, cfg.epsilon_end
    if cfg.epsilon_schedule == "linear":
        if cfg.epsilon_decay <= 0:
            return ee
        frac = min(1.0, s / float(cfg.epsilon_decay))
        return ee + (es - ee) * (1.0 - frac)
    elif cfg.epsilon_schedule == "cosine":
        import math as _m
        if cfg.epsilon_decay <= 0:
            return ee
        frac = min(1.0, s / float(cfg.epsilon_decay))
        return ee + (es - ee) * 0.5 * (1.0 + _m.cos(_m.pi * frac))
    elif cfg.epsilon_schedule == "two_stage":
        import math as _m
        # Stage 1: start -> mid
        if s <= cfg.epsilon_mid_step:
            if cfg.epsilon_mid_step <= 0:
                return cfg.epsilon_mid
            return cfg.epsilon_mid + (es - cfg.epsilon_mid) * _m.exp(-s / float(max(1, cfg.epsilon_mid_step)))
        # Stage 2: mid -> end
        post = s - cfg.epsilon_mid_step
        return ee + (cfg.epsilon_mid - ee) * _m.exp(-post / float(max(1, cfg.epsilon_decay)))
    else:  # default exponential
        import math as _m
        return ee + (es - ee) * _m.exp(-s / float(max(1, cfg.epsilon_decay)))


def _compute_temperature(step: int, cfg: TrainingConfig) -> float:
    import math as _m
    ts, te = cfg.temperature_start, cfg.temperature_end
    # Use exponential style similar to epsilon
    temp = te + (ts - te) * _m.exp(-step / float(max(1, cfg.temperature_decay)))
    return max(1e-3, float(temp))


def select_action(policy_net: DQN, state: torch.Tensor, steps_done: int, cfg: TrainingConfig, num_actions: int) -> Tuple[int, Dict[str, Any]]:
    """Select an action according to exploration strategy.

    Returns (action, meta) where meta contains the computed epsilon/temperature and action source.
    This function remains backward-compatible conceptually but now returns richer metadata.
    """
    # Compute epsilon & temperature once
    eps_threshold = _compute_epsilon(steps_done, cfg)
    action_source = "greedy"
    temperature = None
    if cfg.exploration_strategy == "boltzmann":
        # Pure Boltzmann sampling (no epsilon randomness); epsilon still reported for consistency
        temperature = _compute_temperature(steps_done, cfg)
        with torch.no_grad():
            q = policy_net(state.unsqueeze(0)).squeeze(0)
            # Numerical stability: subtract max before exp
            logits = q / temperature
            probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1).item())
            action_source = "boltzmann"
    else:  # epsilon-greedy (default)
        if random.random() < eps_threshold:
            action = random.randrange(num_actions)
            action_source = "random"
        else:
            with torch.no_grad():
                q = policy_net(state.unsqueeze(0))
                action = int(q.argmax(dim=1).item())
            action_source = "greedy"
    meta = {"epsilon": float(eps_threshold), "action_source": action_source}
    if temperature is not None:
        meta["temperature"] = float(temperature)
    return action, meta


def optimize(policy_net: DQN, target_net: DQN, buffer: ReplayBuffer, cfg: TrainingConfig, optimizer: optim.Optimizer, device: torch.device):
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
    """Run training, supporting iterative resume & periodic checkpoints.

    Artifacts produced inside output_dir:
      - policy_net.pt                  (final policy weights only)
      - checkpoint_*.pt                (full training checkpoints)
      - checkpoint_latest.pt           (latest full checkpoint)
      - episodes.jsonl                 (one JSON per episode summary)
      - training_summary.txt           (rolled-up summary at end)
    When resuming, cfg.episodes means *additional* episodes to run.
    """
    cfg = cfg or TrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    print("Training using device:", device)

    # ------------------- Internal helper serialization ------------------- #
    def _replay_to_dict(rb: ReplayBuffer) -> Dict[str, Any]:
        return {
            "capacity": rb.capacity,
            "position": rb.position,
            "memory": rb.memory,  # list of tuples (numpy arrays will be pickled)
        }

    def _replay_from_dict(data: Dict[str, Any]) -> ReplayBuffer:
        rb = ReplayBuffer(capacity=data["capacity"])
        rb.memory = data["memory"]
        rb.position = data.get("position", 0)
        return rb

    def _save_checkpoint(name: str, extra_flags: Optional[Dict[str, Any]] = None):
        ckpt = {
            "model": policy_net.state_dict(),
            "target_model": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
            "session_meta": {
                "episodes_total": session.episodes_total,
                "total_reward": session.total_reward,
                "total_lines_cleared": session.total_lines_cleared,
            },
            "session_obj": session,  # full dataclass (episodes list)
            "replay": _replay_to_dict(buffer),
            "global_step": global_step,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "flags": extra_flags or {},
        }
        path = output_dir / name
        torch.save(ckpt, path)
        # Maintain latest pointer
        latest_path = output_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(path.name) if hasattr(latest_path, "symlink_to") else torch.save(ckpt, latest_path)

        # Simple rotation
        if cfg.keep_last_n_checkpoints > 0:
            ckpts = sorted(output_dir.glob("checkpoint_*.pt"))
            excess = len(ckpts) - cfg.keep_last_n_checkpoints
            for old in ckpts[:max(0, excess)]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def _load_checkpoint(path: Path):  # returns loaded components
        loaded = torch.load(path, map_location="cpu")
        return loaded

    # ------------------- Initialize or resume ------------------- #
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        loaded = _load_checkpoint(ckpt_path)

        # Use architecture from loaded config if present to avoid mismatch
        loaded_cfg_dict: Dict[str, Any] = loaded.get("config", {})
        hidden_layers = loaded_cfg_dict.get("hidden_layers", cfg.hidden_layers) or cfg.hidden_layers or [256, 256, 256]
        dueling = loaded_cfg_dict.get("dueling", cfg.dueling)
        use_layer_norm = loaded_cfg_dict.get("use_layer_norm", cfg.use_layer_norm)
        dropout = loaded_cfg_dict.get("dropout", cfg.dropout)

        policy_net = DQN((4,), 5, hidden_layers=hidden_layers, dueling=dueling, use_layer_norm=use_layer_norm, dropout=dropout)  # type: ignore[arg-type]
        target_net = DQN((4,), 5, hidden_layers=hidden_layers, dueling=dueling, use_layer_norm=use_layer_norm, dropout=dropout)  # type: ignore[arg-type]
        policy_net.load_state_dict(loaded["model"])  # move to device later
        target_net.load_state_dict(loaded["target_model"])
        optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
        try:
            optimizer.load_state_dict(loaded["optimizer"])
        except Exception:
            print("[Resume] Optimizer state mismatch; using fresh optimizer")
        buffer = _replay_from_dict(loaded["replay"]) if "replay" in loaded else ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
        session = loaded.get("session_obj") or Session(session_id="train-demo", mode="train", seed=cfg.seed, config={})
        global_step = int(loaded.get("global_step", 0))
        # Restore RNG states (best-effort)
        try:
            random.setstate(loaded["python_random_state"])  # type: ignore
        except Exception:
            pass
        try:
            torch.set_rng_state(loaded["torch_rng_state"])  # type: ignore
        except Exception:
            pass
        if torch.cuda.is_available() and loaded.get("torch_cuda_rng_state_all"):
            try:
                states: List[torch.ByteTensor] = loaded["torch_cuda_rng_state_all"]
                for i, st in enumerate(states):
                    torch.cuda.set_rng_state(st, device=i)
            except Exception:
                pass
        rng = random  # use module-level RNG after restore
        base_episode_index = session.episodes_total
        print(f"[Resume] Loaded checkpoint with {base_episode_index} episodes, global_step={global_step}")
    else:
        session = Session(session_id="train-demo", mode="train", seed=cfg.seed, config={})
        rng = random.Random(cfg.seed)
        policy_net = DQN((4,), 5, hidden_layers=cfg.hidden_layers or [256, 256, 256], dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
        target_net = DQN((4,), 5, hidden_layers=cfg.hidden_layers or [256, 256, 256], dueling=cfg.dueling, use_layer_norm=cfg.use_layer_norm, dropout=cfg.dropout)  # type: ignore[arg-type]
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
        buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
        global_step = 0
        base_episode_index = 0

    # Move models to requested device after (potential) loading
    policy_net.to(device)
    target_net.to(device)

    episodes_file = output_dir / "episodes.jsonl"

    additional_episodes = cfg.episodes
    for local_ep in range(additional_episodes):
        ep_idx = base_episode_index + local_ep
        env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=cfg.max_steps)
        state, _ = env.reset()
        episode = Episode(index=ep_idx)
        done = False
        truncated = False
        while not (done or truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            action, _meta = select_action(policy_net, state_t, global_step, cfg, 5)
            next_state, reward, done, truncated, info = env.step(action)
            buffer.push(state, action, reward, next_state, done or truncated)
            episode.record_step(reward, info.get("lines_delta", 0), int(next_state[3]), int(next_state[2]))
            state = next_state
            loss_val = optimize(policy_net, target_net, buffer, cfg, optimizer, device)
            if global_step % cfg.target_sync == 0:
                target_net.load_state_dict(policy_net.state_dict())
            global_step += 1
            if step_callback:
                try:
                    step_callback({
                        "episode": ep_idx,
                        "global_step": global_step,
                        "reward": reward,
                        "loss": loss_val,
                        "lines_cleared_total": info.get("lines_cleared_total"),
                        "lines_delta": info.get("lines_delta"),
                        "done": done,
                        "truncated": truncated,
                    })
                except Exception as e:  # defensive: streaming shouldn't crash training
                    print("step_callback error", e)
        episode.finalize(terminated=bool(done), truncated=bool(truncated), interrupted=False, reason=None)
        session.record_episode(episode.total_reward, episode.lines_cleared, episode)
        # Append summary line
        try:
            import json
            with episodes_file.open("a") as f:
                f.write(json.dumps(episode.summary_dict()) + "\n")
        except Exception as e:
            print("[Warn] Failed to write episode summary:", e)
        env.close()

        # Periodic checkpoint
        if cfg.checkpoint_every_episodes > 0 and (session.episodes_total % cfg.checkpoint_every_episodes == 0):
            _save_checkpoint(f"checkpoint_{session.episodes_total:05d}.pt", {"type": "periodic"})

    session.finish()
    # Persist summary & final artifacts
    (output_dir / "training_summary.txt").write_text(
        f"Episodes: {session.episodes_total}\nAvg reward: {session.avg_reward:.3f}\nAvg lines: {session.avg_lines_cleared:.3f}\n")
    torch.save(policy_net.state_dict(), output_dir / "policy_net.pt")
    _save_checkpoint(f"checkpoint_{session.episodes_total:05d}.pt", {"type": "final"})
    return session
