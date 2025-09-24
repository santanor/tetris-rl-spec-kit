# Tetris RL Demo

A clean, educational reinforcement learning environment for Tetris using DQN with comprehensive observation features and reward shaping.

## Overview

This project provides a complete Tetris environment with:
- **Rich Observation Space**: 76-dimensional feature vector covering board state, piece information, and strategic features
- **Sophisticated Reward System**: Multi-component rewards encouraging line clears and strategic play
- **Real-time Dashboard**: Web-based training visualization with live metrics
- **Simple Training Scripts**: Easy-to-use headless training and benchmarking

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project in development mode
pip install -e .
```

### Run Training Demo

```bash
# Quick 10-episode training demo
python scripts/train_demo.py

# Benchmark environment performance
python scripts/bench_headless.py
```

### Launch Web Dashboard

```bash
# Start the web dashboard
python -m uvicorn tetris_rl.web.app:app --reload

# Open http://127.0.0.1:8000 in your browser
# Click "Start Training" to begin
```

## Project Structure

```
src/tetris_rl/
├── core/
│   └── board.py              # Core Tetris game logic (10x20 board)
├── env/
│   ├── tetris_env.py         # Gymnasium environment wrapper
│   └── reward_config.py      # Reward configuration system
├── agent/
│   ├── dqn.py               # Deep Q-Network implementation
│   ├── replay_buffer.py     # Experience replay buffer
│   └── trainer.py           # Training loop and configuration
├── models/                  # Data models for episodes and sessions
├── utils/                   # Utilities (seeding, paths)
└── web/
    ├── app.py              # FastAPI dashboard backend
    └── static/             # Dashboard frontend (HTML/JS/CSS)

scripts/
├── train_demo.py           # Headless training example
└── bench_headless.py       # Environment benchmarking
```

## Observation Space (76 dimensions)

The environment provides rich observations designed for strategic Tetris play:

### Core Board Features (40 dims)
- **Column Heights** (10): Normalized height of each column (0-1)
- **Predicted Heights** (10): Height after dropping current piece at each column
- **Per-Column Holes** (10): Number of holes in each column
- **Predicted Holes Created** (10): New holes if dropping at each column

### Current Piece Information (15 dims)
- **Piece Type** (7): One-hot encoding for I, O, T, S, Z, J, L pieces
- **Rotation State** (4): One-hot encoding for 4 rotation states
- **Position** (2): Normalized x, y coordinates
- **Skyline Delta** (1): Height difference between tallest and shortest columns
- **Lines Possible** (1): Lines clearable from current placement

### Strategic Features (17 dims)
- **Aggregate Metrics** (5): Total height, bumpiness, max well depth, I-piece dependency, ready lines
- **Surface Analysis** (4): Row/column transitions, buried blocks, overhang cells
- **Placement Analysis** (3): Landing height, piece contact score, total blocks
- **Well Analysis** (2): Deep I-wells count, O-gaps count
- **Future Information** (3): Next piece placeholder (currently zeros)

### Next Piece Preview (7 dims)
- **Next Piece Type** (7): One-hot encoding (placeholder, currently zeros)

## Action Space

5 discrete actions:
- **0**: No-op / Natural drop (piece falls one row)
- **1**: Move left
- **2**: Move right  
- **3**: Rotate clockwise
- **4**: Hard drop (piece falls to bottom immediately)

## Reward System

The reward system uses multiple components to encourage strategic play:

### Line Clear Rewards (Primary)
- **1 line**: +10.0
- **2 lines**: +30.0  
- **3 lines**: +50.0
- **4 lines (Tetris)**: +200.0

### Continuous Shaping
- **Survival**: +0.02 per step (encourages staying alive)
- **Delta Stable**: +0.2 when lock doesn't increase height variance
- **Top Out**: -8.0 when game ends

### Placement Quality (applied on piece lock)
- **Holes**: -0.05 per new hole created
- **Bumpiness**: -0.005 × surface roughness  
- **Height**: -0.01 × maximum column height
- **Surface Transitions**: -0.001 × row/column transitions
- **Strategic Bonuses**: Landing height, ready lines, well usage
- **Penalties**: Buried blocks, overhangs, blocked wells

The reward components are exposed in `info['reward_components']` for analysis and interpretability.

## Configuration

### Training Configuration

```python
from tetris_rl.agent.trainer import TrainingConfig

config = TrainingConfig(
    episodes=100,           # Number of episodes to train
    max_steps=500,         # Max steps per episode
    batch_size=64,         # Training batch size
    gamma=0.99,            # Discount factor
    lr=1e-3,               # Learning rate
    epsilon_start=1.0,     # Initial exploration
    epsilon_end=0.05,      # Final exploration
    epsilon_decay=300,     # Exploration decay steps
    hidden_layers=[64,64], # Network architecture
    device="auto"          # Device selection (auto/cpu/cuda)
)
```

### Reward Configuration

```python
from tetris_rl.env.reward_config import RewardConfig

reward_config = RewardConfig(
    line_reward_1=10.0,      # Single line reward
    line_reward_4=200.0,     # Tetris reward
    survival_reward=0.02,    # Per-step survival
    hole_penalty_per=-0.05,  # Hole creation penalty
    # ... many other configurable rewards
)
```

## Network Architecture

- **Input**: 76-dimensional observation vector
- **Hidden Layers**: Configurable MLP (default: [64, 64])
- **Output**: 5 Q-values (one per action)
- **Activation**: ReLU
- **Optimization**: Adam optimizer with Huber loss

## Training Features

### Core DQN Components
- **Experience Replay**: 20K capacity buffer with random sampling
- **Target Network**: Synchronized every 200 steps
- **Epsilon-Greedy**: Exponential decay from 1.0 to 0.05
- **Device Support**: Automatic GPU detection and usage

### Monitoring & Visualization
- **Real-time Dashboard**: Live training metrics and board visualization
- **Episode Tracking**: Complete episode history with rewards and features
- **Feature Analysis**: Board state visualization and Q-value inspection

## Performance

- **Environment Speed**: ~3,100 steps/sec (headless, CPU)
- **GPU Training**: Automatic CUDA detection and usage
- **Memory Efficient**: Configurable replay buffer and batch processing

## Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ scripts/
ruff check src/ scripts/

# Type checking
mypy src/
```

### Extending the Environment
- **Observation Features**: Add new features in `TetrisEnv._build_obs()`
- **Reward Components**: Modify `RewardConfig` and reward calculation
- **Network Architecture**: Update `DQN` class or training config
- **Action Space**: Extend actions in `TetrisBoard.step()`

## Examples

### Headless Training

```python
from pathlib import Path
from tetris_rl.agent.trainer import run_training, TrainingConfig

# Configure training
config = TrainingConfig(
    episodes=50,
    hidden_layers=[128, 128],
    device="cuda"
)

# Run training
output_dir = Path("training_runs/experiment_1")
session = run_training(output_dir, config)

print(f"Average reward: {session.avg_reward:.2f}")
print(f"Average lines: {session.avg_lines_cleared:.1f}")
```

### Custom Environment

```python
from tetris_rl.env import TetrisEnv
from tetris_rl.env.reward_config import RewardConfig

# Create environment with custom rewards
custom_rewards = RewardConfig(
    line_reward_4=500.0,      # Higher Tetris reward
    hole_penalty_per=-0.1,    # Stricter hole penalty
)

env = TetrisEnv(
    max_steps=1000,
    reward_config=custom_rewards,
    seed=42
)

obs, info = env.reset()
# ... training loop
```

## Research Applications

This environment is designed for:
- **Interpretable RL**: Rich observations and reward decomposition
- **Reward Shaping Research**: Configurable multi-component rewards  
- **Deep RL Baselines**: Standard DQN with modern best practices
- **Educational Use**: Clear, readable codebase for learning RL concepts

## License

MIT

---

**Note**: This is an educational/research implementation focused on clarity and interpretability rather than competitive Tetris performance. For production Tetris AI, consider more sophisticated features like T-spin detection, advanced rotation systems, and multi-step planning.

## What You Get

* Core Tetris logic (`TetrisBoard`)
* Minimal environment wrapper (`TetrisEnv`) exposing a 12‑dim observation
* Basic DQN (configurable hidden sizes only)
* Single‑env training loop (epsilon‑greedy, replay buffer, target network)
* FastAPI + WebSocket dashboard with:
  * Live board & colored grid
  * Recent episode rewards & lines
  * Loss & reward charts (sliding window)
  * Minimal reward component bar (line / survival / top‑out)
  * Simple network activation & Q‑value visualization

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

## Run the Dashboard

Launch the server:

```bash
python -m uvicorn tetris_rl.web.app:app --reload
```

Then open: <http://127.0.0.1:8000/>

Open http://127.0.0.1:8000 and click Start. Adjust the episode count first if desired. The page streams per‑step telemetry until all episodes finish.

## Layout

```
src/tetris_rl/core/board.py      # Board + piece mechanics
src/tetris_rl/env/tetris_env.py  # Minimal env (12-dim obs, simple reward)
src/tetris_rl/agent/dqn.py       # Plain MLP Q-network
src/tetris_rl/agent/trainer.py   # Small single-env training loop
src/tetris_rl/web/app.py         # FastAPI + websocket streaming
src/tetris_rl/web/static/        # Frontend (index.html, main.js, styles.css)
scripts/train_demo.py            # Headless training example
```

## Observation (12 features)

```
[ h0..h9 (normalized column heights), lines_fraction, step_fraction ]
```
* Column heights are each /20.
* `lines_fraction` = total_lines_cleared / 200 (rough normalization).
* `step_fraction` = step_index / max_steps.

## Reward (Minimal Scheme)

| Component | Description |
|-----------|-------------|
| line_reward_N | Reward for clearing N lines simultaneously (monotonic table) |
| survival | Small positive reward every non-terminal step |
| top_out_penalty | Large negative penalty applied once when the board tops out |

Defaults (see `RewardConfig`): 1:1.0, 2:3.0, 3:5.0, 4:8.0, survival:0.02, top_out:-10.0.

No hole / height / bumpiness shaping. The agent must discover structure purely from line rewards and survival.

## Network

Simple feedforward MLP: input -> (Linear+ReLU)*N -> Linear(num_actions). Hidden sizes come from `TrainingConfig.hidden_layers` (default `[64, 64]`).

## Training Loop Basics

1. Epsilon-greedy with exponential decay: `eps = eps_end + (eps_start-eps_end)*exp(-t/decay)`
2. Replay buffer warm-up gate (`min_replay`)
3. Target network sync every `target_sync` steps
4. Smooth L1 (Huber) loss on Q-learning target `r + gamma * max_a' Q_target(s', a')`

## Customize Quickly

Edit `RewardConfig` for reward tweaks or `TrainingConfig` for learning params. Because the code path is short, changes are easy to reason about.

## Why So Minimal?

The project previously experimented with: multi-env batching, deep reward shaping (holes, bumpiness, depth weights, density), dueling heads, layer norm, dropout, advanced exploration schedules, live hyperparameter editing, checkpoint/resume. All were intentionally removed to present a clear baseline you can extend selectively.

## Extending (Suggestions)

* Add next-piece feature to the observation.
* Replace hand-crafted features with a downsampled grid + conv net.
* Reintroduce (selectively) shaping if learning is too slow.
* Implement prioritized replay or Double DQN.

## License

MIT

---
Happy tinkering! Keep it small; add only what you can justify with learning curves.

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

