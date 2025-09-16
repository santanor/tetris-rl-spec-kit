# Tetris RL Demo

Reinforcement learning demo with a simplified Tetris environment, DQN baseline training loop, and a real-time web dashboard for visualization.

## Features

- Simplified Tetris core (`TetrisBoard`) with line clearing and feature extraction
- Gym-like environment wrapper (`TetrisEnv`)
- DQN training (epsilon-greedy, replay buffer, target network sync)
- Real-time web dashboard (FastAPI + WebSocket + Chart.js) for:
  - Live board state
  - Per-step reward & loss
  - Episode reward aggregation
  - Session summary

## Quick Start

Create and activate a virtual environment (if not already):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run a quick training demo (non-visual):

```bash
python scripts/train_demo.py
```

## Web Dashboard

Start the web server:

```bash
python -m uvicorn tetris_rl.web.app:app --reload
```

Then open: <http://127.0.0.1:8000/>

Click "Start Training" to begin streaming a session. Adjust episode count before launching.

If you see a raw message saying static files not found, ensure the directory `src/tetris_rl/web/static` exists (it is included in repo) and that you launched with the correct working directory (project root).

## Project Layout

```text
src/tetris_rl/core/board.py          # Tetris board logic
src/tetris_rl/env/tetris_env.py      # Gym-style env wrapper
src/tetris_rl/agent/dqn.py           # DQN network definition
src/tetris_rl/agent/trainer.py       # Training loop with callback support
src/tetris_rl/web/app.py             # FastAPI app + websocket streaming
src/tetris_rl/web/static/            # Frontend assets (index.html, main.js, styles.css)
scripts/train_demo.py                # CLI training demo
scripts/run_demo.py                  # Random policy session export demo
```

## Notes

- The environment observation is a small feature vector: `[step, lines_cleared_total, holes, aggregate_height]`
- Reward signal: lines cleared per step minus small step penalty (-0.01)
- Frontend rendering uses ASCII transformation of board state; active piece marked with `*`.
- Colored board: dashboard also renders a color grid using standard Tetris piece palette.

### Reward Shaping (Enhanced)

The environment now applies additional shaping to accelerate learning:

| Component | Description | Default Weight/Effect |
|-----------|-------------|------------------------|
| Line clear base | Non-linear: 1,3,5,8 for clearing 1–4 lines | Encourages multi-line clears |
| Step penalty | Constant per action | -0.01 |
| Survival bonus | Small reward each step survived | +0.002 |
| Holes delta | Penalizes creation, rewards reduction | -0.20 per new hole (implicit sign handling) |
| Height delta | Penalizes stack height growth | -0.005 per aggregate height increase |
| Bumpiness delta | Penalizes uneven surface increases | -0.01 per bumpiness increase |
| Top-out penalty | Applied when topping out | -2.0 |

Breakdowns are exposed in `info['reward_components']` for interpretability along with `holes_delta`, `height_delta`, and `bump_delta`.

These weights are intentionally conservative; adjust as needed for stability. Potential-based shaping could be added later for policy invariance.

### Dynamic Reward Configuration (Live Tuning)

You can now inspect and modify reward shaping weights at runtime:

API Endpoints:

```http
GET /api/reward-config        # returns current config as JSON
POST /api/reward-config       # accepts partial JSON with fields to update
```

Example update (increase line rewards, make holes penalty harsher):

```bash
curl -X POST http://127.0.0.1:8000/api/reward-config \
  -H 'Content-Type: application/json' \
  -d '{"line_reward_4":10, "holes_weight": -0.3}'
```

Successful updates trigger a WebSocket broadcast message:

```json
{ "type": "config_update", "config": { ... } }
```

Dashboard UI:

The web dashboard now includes a "Reward Config" panel with editable numeric fields. Press "Apply" to send a batch update. Values are immediately used by subsequent environment steps (the active episode picks up changes without restart since the environment reads the shared config each step).

`reward_components` now also includes a `structural_breakdown` with per-term contributions:

```json
"reward_components": {
  "base_line": 0,
  "step_penalty": -0.01,
  "structural": -0.034,
  "survival": 0.002,
  "top_out": 0,
  "structural_breakdown": {
     "holes": -0.02,
     "height": -0.004,
     "bumpiness": -0.01
  },
  "config_hash": 123456789
}
```

This enables quick experimentation without code edits or server restarts.

### GPU / Device Selection

Training now supports optional GPU execution for the neural network components.

By default the system uses `auto` device selection:

- If a CUDA-capable GPU is available: uses `cuda`.
- Otherwise: falls back to `cpu`.

Config field (in `TrainingConfig`):

```python
TrainingConfig(device="auto")  # values: 'auto' | 'cpu' | 'cuda'
```

Web API usage:

```bash
curl -X POST 'http://127.0.0.1:8000/api/train?episodes=20&device=cuda'
```

CLI example (modify your script or instantiate config):

```python
from tetris_rl.agent.trainer import run_training, TrainingConfig
cfg = TrainingConfig(episodes=50, device="cuda")
run_training(Path("training_runs/gpu_test"), cfg)
```

#### When GPU Helps

For this minimal DQN (tiny MLP, single environment) GPU speedups are modest; most wall time is Python env stepping and WebSocket I/O.

You will see more benefit if you:

- Increase batch size (e.g. 256+)
- Enlarge the network (deeper/wider or grid-based conv features)
- Run multiple environments in parallel and batch inference
- Perform multiple optimization steps per environment step


#### Verifying GPU Use

Watch `nvidia-smi` while training, or add a log:

```python
import torch; print("Device:", next(policy_net.parameters()).device)
```

If you request `device=cuda` but no GPU is present, the system logs a warning and falls back to CPU.

#### Caveats

- Small models may run slightly slower on GPU due to transfer overhead.
- Avoid creating new tensors on CPU every micro-step; current code already sends batches to the resolved device.
- For further speed, reduce WebSocket broadcast frequency (e.g., send every N steps) or run headless.

Feel free to open an issue / ask to add multi-env or convolutional observation support if you want to scale further.

## Future Enhancements (Ideas)

- Pause/resume/cancel training via API
- Additional charts (moving average reward, loss smoothing, action distribution)
- Persist and list historical sessions
- More advanced observation modes (grid, image)

## License

MIT

### Tuning for More Line Clears (New Defaults)

The reward weights were adjusted to encourage stacking for multi-line clears while still discouraging pathological board states:

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| line_reward_1 | 1.0 | 1.5 | Slightly more incentive even for singles |
| line_reward_2 | 3.0 | 4.0 | Maintain progression |
| line_reward_3 | 5.0 | 8.0 | Larger jump to reward setup skill |
| line_reward_4 | 8.0 | 14.0 | Strong payoff for tetrises |
| step_penalty | -0.01 | -0.005 | Less pressure to rush placement |
| survival_bonus | 0.002 | 0.0015 | Mild correction after lowering step penalty |
| holes_weight | -0.20 | -0.25 | Stronger discouragement of digging mistakes |
| height_weight | -0.005 | -0.003 | Softer to allow buildup before clears |
| bumpiness_weight | -0.01 | -0.008 | Reduced noise penalty to not dominate |

Added fields:

- `height_threshold` (default 100 aggregate) — below this, height deltas are ignored when `conditional_height=True`.
- `conditional_height` — delays height penalty until mid/late stack, allowing structured buildup for multi-line opportunities.

Instrumentation now broadcasts per-episode:

```json
{
  "type": "episode_end",
  "line_clears": <lines_this_episode>,
  "avg_structural": { "holes": ..., "height": ..., "bumpiness": ... }
}
```

Suggested next experiments:

1. Increase `line_reward_4` further (up to ~18) if tetrises remain rare.
2. Decrease `holes_weight` magnitude if agent becomes too conservative early.
3. Introduce a small positive bonus for reducing bumpiness when near threshold.
4. Replace step penalty + survival bonus with a single time-decayed survival shaping term.

If line clears are still sparse, consider: (a) increasing exploration (epsilon decay slower), (b) adding lookahead features (e.g., next piece), or (c) switching observation to a downsampled grid and enlarging the network capacity.

### Live Exploration (Epsilon) Tuning

The dashboard exposes epsilon-greedy parameters for on-the-fly adjustment during training startup (affects new episodes immediately since selection reads the shared overrides each step):

Endpoints:

```http
GET /api/training-config
POST /api/training-config  { "epsilon_start":1.0, "epsilon_end":0.05, "epsilon_decay":800 }
```

UI Panel: "Exploration (Epsilon)" — edit and Apply. A WebSocket `training_config_update` message syncs all clients.

Heuristics:
- Increase `epsilon_decay` to prolong exploration (slower exponential decay).
- Raise `epsilon_end` temporarily if policy prematurely exploits suboptimal stacking.
- Lower `epsilon_start` (e.g. 0.8) only after rewards stabilize; early reduction can hurt discovery of multi-line setups.
 

### Episode Line Clears Chart

Added a second chart plotting the number of lines cleared per episode (bar). Use it alongside reward to detect shaping misalignment (e.g. reward rising while line clears stagnate may indicate structural shaping dominance).

Potential additions (not yet implemented): moving average overlay, separate stacked bars for single/double/triple/quad distribution once those stats are tracked.

### Advanced Exploration Controls (Combo Feature)

The training system now supports multiple exploration strategies and schedules, all tunable live:

Supported strategies:

- `epsilon_greedy` (default): classic ε-greedy with configurable schedule
- `boltzmann`: softmax (a.k.a. Boltzmann) sampling over Q-values using a temperature τ

Epsilon schedules (`epsilon_schedule`):

- `exp`: ε = ε_end + (ε_start - ε_end) * exp(-t / ε_decay)
- `linear`: linear interpolation from start to end over `epsilon_decay` steps
- `cosine`: cosine anneal from start to end over `epsilon_decay` steps
- `two_stage`: exponential start→mid until `epsilon_mid_step`, then exponential mid→end using `epsilon_decay`

Additional parameters:

- `epsilon_mid`, `epsilon_mid_step` (only used by `two_stage`)
- `temperature_start`, `temperature_end`, `temperature_decay` (Boltzmann strategy)

Live overrides endpoint (POST `/api/training-config`):
 
```jsonc
{
  "epsilon_start": 1.0,
  "epsilon_end": 0.05,
  "epsilon_decay": 800,
  "epsilon_schedule": "exp",
  "epsilon_mid": 0.2,
  "epsilon_mid_step": 2000,
  "exploration_strategy": "epsilon_greedy",  // or "boltzmann"
  "temperature_start": 1.0,
  "temperature_end": 0.1,
  "temperature_decay": 2000
}
```

UI Additions:

- New select fields for schedule & strategy
- Mid ε + mid step inputs (two-stage)
- Temperature controls (when using Boltzmann)
- Live runtime panel showing: current ε, τ (if applicable), and per-episode action source percentages (random / greedy / boltz)

WebSocket step messages now include:

```jsonc
{
  "type": "step",
  "epsilon": 0.312,
  "temperature": 0.54,          // present only for boltzmann
  "action_source": "random"     // one of random|greedy|boltzmann
}
```

Episode end messages add aggregated distribution:

```jsonc
{
  "type": "episode_end",
  "action_dist": {"random":0.42, "greedy":0.53, "boltzmann":0.05},
  "last_epsilon": 0.18,
  "last_temperature": 0.32
}
```

Tuning Guidelines:

- If random% stays >60% for many episodes, consider lowering `epsilon_start` or increasing `epsilon_decay` (slower decay means high ε persists—ensure that matches your horizon).
- If greedy% climbs early but line clears stagnate, try switching temporarily to `boltzmann` with higher `temperature_start` to diversify high-value tie regions.
- Use `two_stage` when you want rapid early exploration (stage 1) then a gentler convergence (stage 2).
- Cosine can be useful for periodic mild boosts in exploitation near the end of training horizon.

Migration Notes:

- Existing scripts not specifying new fields continue to function (defaults preserve previous behavior: exponential ε-greedy).
- The core training API remains backward compatible; only `select_action` now returns metadata (ignored where not captured).

### Absolute Holes Penalty (Improved Line Clearing)

Added `holes_abs_weight` (default `-0.02`) which applies a per-step penalty proportional to the current number of holes (independent of delta). Motivation:

Problem: A purely delta-based holes penalty sometimes lets the agent tolerate a stable number of holes if it can offset penalties with incidental positive shaping elsewhere. This stagnates clearing behavior.

Solution: Combine:

- `holes_weight` (delta-based: punishes creation, rewards removal)
- `holes_abs_weight` (persistent pressure: every existing hole costs each step)

Tuning suggestions:

- Start with a small magnitude (e.g. -0.02 * holes). If agent still stacks above cavities without clearing, increase gradually (up to -0.05).
- If agent becomes overly conservative (avoids risk leading to low line clears), reduce absolute penalty and rely more on delta shaping plus stronger line rewards.
- Monitor new `abs_holes_penalty` component in the WebSocket `reward_components.structural_breakdown` to ensure it’s not dominating total reward (aim < ~25% of average per-step negative shaping early on).

Potential next refinements:

- Scale absolute holes penalty by normalized board height (higher stacks amplify urgency to resolve holes).
- Provide separate rewards for “covering” vs “clearing” holes (distinguish filling from clearing lines that eliminate holes).

### Depth-Weighted Hole Penalties

To emphasize the cost of deep, buried holes more than shallow ones, two additional shaping terms were added:

| Field | Meaning |
|-------|---------|
| `weighted_holes_weight` | Delta-based shaping using sum(depth^p) for holes vs previous state. Rewards reducing deep holes, penalizes creating them. |
| `weighted_holes_abs_weight` | Absolute per-step penalty proportional to current weighted hole score (persistent pressure). |
| `holes_depth_power` | Power `p` used when computing each hole's contribution `depth ** p` (default 1.5–2.0 recommended). |

Interpretation: A hole 8 rows below the top counts far more than one 2 rows below, so the agent is nudged to clear downwards rather than just building sideways.

Tuning guidelines:

- Start with a modest `holes_depth_power` (1.5). Too high (>=3) can overwhelm other objectives.
- Keep absolute weighted penalty small relative to standard absolute holes penalty (e.g. `weighted_holes_abs_weight` ~ 25–50% of `holes_abs_weight` magnitude) to avoid double-counting.
- If agent becomes paralyzed trying to dig, reduce absolute weighted penalty first, then delta weight.

Monitoring: Both `weighted_holes_delta` and `abs_weighted_holes_penalty` appear inside `reward_components.structural_breakdown` for transparency.

### Row Density Shaping (New)

To further encourage the agent to build compact, low-gap structures before clearing lines, a row density shaping signal was added.

Definitions:

- Row density (per row) = filled_cells / board_width.
- Average row density = mean density over all non-empty rows (rows containing at least one filled cell).

New `RewardConfig` fields:

| Field | Default | Effect |
|-------|---------|--------|
| `row_density_delta_weight` | 0.5 | Scales reward for increases (positive delta) or penalties for decreases (negative delta) in average row density between consecutive steps. |
| `row_density_abs_weight` | 0.0 | Optional absolute shaping: adds weight * current average density each step (persistent pressure). |

Reward components:

- `density_delta` appears in `reward_components.structural_breakdown` when `row_density_delta_weight != 0`.
- `density_abs` appears when `row_density_abs_weight != 0`.

Rationale:

Holes penalties focus on vertical cavities, but wide horizontal sparsity (scattered blocks) also harms future placement efficiency. By rewarding incremental improvements in compactness, the policy is nudged to fill sideways gaps sooner, reducing future hole creation risk and enabling multi-line clears.

Tuning Guidelines:

- Start with only the delta term (keep `row_density_abs_weight = 0`) to avoid constant reward inflation; absolute term can be introduced later (e.g. 0.05–0.1) if the agent oscillates between compact and sparse states.
- If density shaping dominates reward (agent fixates on micro-filling without clearing lines), reduce `row_density_delta_weight` or increase line clear rewards.
- Combine with moderate holes penalties; overlapping signals can otherwise double-penalize the same structural flaw (a gap often contributes to both lower density and a hole if covered above).

Observability:

WebSocket `step` info includes `density_delta` and `density_after`. The structural breakdown adds:

```jsonc
{
  "density_delta": <delta_contribution>,
  "density_abs": <absolute_contribution>
}
```

Potential Extensions:

- Weight density by row height (higher rows more costly when sparse).
- Track variance of row densities to encourage uniform coverage.
- Introduce a compaction bonus when all non-empty rows exceed a threshold density (e.g. >0.8).


### Iterative Training & Checkpointing (Resume Across Sessions / Devices)

The training loop now supports robust checkpointing so you can:

- Pause a long run and resume later.
- Transfer learning from a workstation GPU to a laptop CPU (or vice versa).
- Perform staged curriculum adjustments (modify reward config / exploration mid-run and continue training from the last checkpoint).

Artifacts written to a run directory (e.g. `runs/exp1`):

| File | Description |
|------|-------------|
| `policy_net.pt` | Final policy network weights only (for inference/deployment). |
| `checkpoint_latest.pt` | Symlink (or copy fallback) pointing to the newest full checkpoint. |
| `checkpoint_00005.pt`, `checkpoint_00010.pt`, ... | Periodic full checkpoints (model, optimizer, replay buffer, RNG state, session stats). |
| `episodes.jsonl` | One JSON per episode summary (append-only; safe for streaming analysis). |
| `training_summary.txt` | Final aggregate reward / lines summary. |

Checkpoint contents include:

- Policy + target network state dicts
- Optimizer state dict
- Replay buffer (transitions list + pointer)
- Full `Session` object (including completed `Episode` objects)
- Global step counter
- Python, Torch (CPU & CUDA) RNG states (best-effort restore)
- Original training config fields

#### Enabling Periodic Checkpoints

Configure via `TrainingConfig`:

```python
from pathlib import Path
from tetris_rl.agent.trainer import run_training, TrainingConfig

cfg = TrainingConfig(
  episodes=50,
  checkpoint_every_episodes=5,   # save every 5 episodes
  keep_last_n_checkpoints=3      # rotate to the last 3
)
run_training(Path("runs/exp1"), cfg)
```

#### Resuming Training

Provide `resume_from` pointing to any prior checkpoint. The `episodes` value in the new config is treated as *additional* episodes to run.

```python
from pathlib import Path
from tetris_rl.agent.trainer import run_training, TrainingConfig

# First stage
run_training(Path("runs/exp1"), TrainingConfig(episodes=40, checkpoint_every_episodes=10))

# Later (maybe on another machine)
run_training(
  Path("runs/exp1"),
  TrainingConfig(
    episodes=30,                 # adds 30 more (total will become 70)
    resume_from="runs/exp1/checkpoint_latest.pt",
    device="cpu"                # can change device when resuming
  )
)
```

You can also switch GPUs/CPUs; checkpoint tensors are loaded to CPU first then moved to the requested target device.

#### Inspecting Episode History

`episodes.jsonl` is append-only; each line is a JSON object like:

```json
{"index":12,"total_reward":4.75,"lines_cleared":2,"steps":498,"terminated":true,"truncated":false,"interrupted":false,"max_height":11,"holes_final":3,"notable_flags":[]}
```

Tail the file in real time:

```bash
tail -f runs/exp1/episodes.jsonl
```

#### Best Practices

- Keep `replay_capacity` constant across resumes to avoid distribution shift; resizing is not currently supported.
- If you change reward shaping drastically mid-run, consider starting a fresh directory unless you explicitly want mixed-policy replay.
- For evaluation-only usage, load `policy_net.pt`; you do not need full checkpoints.
- To archive a finished run, copy the directory; symlink for `checkpoint_latest.pt` will remain valid inside the copied folder (or resolve it manually if your filesystem lacks symlink support).

#### Failure Recovery

If a crash occurs mid-episode, only completed episodes are present in `episodes.jsonl`; on resume the partially completed one is discarded (this is usually acceptable). Increase checkpoint frequency if you want tighter RPO (recovery point objective).

---

With these additions you can orchestrate multi-phase or distributed-in-time training workflows without losing optimizer momentum or replay diversity.

### Scalable DQN Architecture (Breaking Change)

The previously fixed two-layer 128-unit MLP has been replaced by a configurable architecture supporting arbitrary hidden layer stacks, optional dueling heads, LayerNorm, and dropout. This enables scaling capacity as reward shaping and exploration mature.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hidden_layers` | list[int] | `[256,256,256]` | Sequence of hidden layer widths. Must be non-empty. |
| `dueling` | bool | `True` | Use value & advantage streams with mean-normalized advantages. |
| `use_layer_norm` | bool | `False` | Apply `nn.LayerNorm` after each hidden linear layer (before activation). |
| `dropout` | float | `0.0` | Dropout probability applied after each hidden layer (0 disables). |

Example usage:

```python
from tetris_rl.agent.dqn import DQN
net = DQN((4,), 5, hidden_layers=[512,512,256], dueling=True, use_layer_norm=True, dropout=0.1)
```

When resuming from a checkpoint, the stored architecture overrides any runtime overrides to guarantee tensor shape compatibility.

#### Dueling Rationale

Separating state-value and action-advantage estimation often reduces variance and speeds convergence when many actions have similar value (common early in Tetris when placements differ little).

#### LayerNorm vs BatchNorm

LayerNorm avoids dependency on batch statistics (important with small / variable batch sizes during replay sampling) and stabilizes deeper stacks without requiring large replay batches.

#### Dropout Guidance

Introduce mild dropout (0.05–0.15) only after scaling width/depth if you observe overfitting (e.g., oscillatory policy or divergence when exploration anneals). For small networks dropout may slow early learning slightly.

### Architecture Overrides via Dashboard

Endpoints:

```http
GET /api/model-config
POST /api/model-config
```

POST payload:

```jsonc
{
  "hidden_layers": "256,256,256",  // or [256,256,256]
  "dueling": true,
  "use_layer_norm": false,
  "dropout": 0.05
}
```

Rules:

- Takes effect on next training or resume session (running session is unchanged).
- `hidden_layers` string is comma-split; invalid entries ignored.
- Checkpoint resume always prioritizes checkpoint architecture.

The dashboard adds a "Model Architecture" panel. After applying overrides click Start / Resume to instantiate with the new configuration.

### Backward Compatibility

Old code that did `DQN((4,), 5)` must now specify architecture arguments. Start with the default triple-256 stack:

```python
from tetris_rl.agent.dqn import DQN
net = DQN((4,), 5, hidden_layers=[256,256,256], dueling=True, use_layer_norm=False, dropout=0.0)
```

Legacy checkpoints (pre-architecture change) are not loadable due to weight shape differences.

### Suggested Scaling Path

| Stage | Config | Notes |
|-------|--------|-------|
| Baseline | [256,256,256], dueling | Strong default; stable. |
| Wider | [512,512,256], dueling | Increases representational capacity. |
| Deeper + Norm | [512,512,512,256], dueling, LayerNorm | For more complex feature interactions. |
| Regularized | Add dropout=0.05 | Mitigate overfit if replay inflates. |

Monitor gradient norms & Q-value drift; consider double DQN or distributional extensions as future improvements.

