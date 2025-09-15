# Data Model (Phase 1)

## Overview

Core entities supporting Tetris RL interpretability demo. Focus: traceability from Session → Episode → Frame while enabling extensible metrics & observation modes.

## Entities

### Session

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| session_id | UUID (string) | Unique identifier per training/eval run | Required |
| created_at | ISO datetime | Session creation timestamp | Required |
| mode | enum(`train`,`eval`) | Operating mode | Required |
| seed | int | Global seed used | Required |
| config | JSON object | Full configuration snapshot (reward weights, obs mode, early stop params) | Immutable post-create |
| status | enum(`running`,`completed`,`error`,`interrupted`) | Lifecycle state | Required |
| episodes_total | int | Number of episodes executed (excludes partial interrupted unless completed) | >=0 |
| metrics_summary | JSON object | Aggregated final metrics (avg reward, lines cleared, warnings triggered) | Computed end |
| export_paths | JSON object | References to CSV/JSON export artifacts | Optional |

### Episode

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| episode_id | string (session_id + index) | Composite identity | Required |
| session_id | string (FK) | Parent session link | Required |
| index | int | 0-based sequence number | >=0 |
| total_reward | float | Sum of per-step shaped rewards | Required |
| lines_cleared | int | Total lines cleared in episode | >=0 |
| steps | int | Steps executed | >=1 unless interrupted early |
| terminated | bool | True if env signaled done | Required |
| truncated | bool | True if truncated (stagnation/early stop) | Required |
| interrupted | bool | True if manual interrupt occurred mid-episode | Default false |
| max_height | int | Max column height reached | >=0 |
| holes_final | int | Hole count end of episode | >=0 |
| warnings | string[] | Warning codes triggered during episode | Optional |
| notable_flags | string[] | e.g., `top_reward`,`top_lines`,`worst_reward` | Derived post-run |

### Frame

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| frame_id | string (episode_id + step) | Unique per frame | Required |
| episode_id | string (FK) | Parent episode | Required |
| step_index | int | 0-based step number | >=0 |
| board_repr | ndarray or list[int] | Flattened grid (H*W) binary or integer occupancy | Size = H*W |
| features | list[float] | Engineered feature vector (heights, holes, bumpiness, etc.) | Length fixed per mode |
| rgb | optional ndarray(H,W,3) | Only if RGB mode active (evaluation) | Optional |
| action | int | Discrete action index taken | Required |
| action_label | string | Human-readable action (e.g., 'L2_R1_Drop') | Required |
| reward | float | Shaped reward emitted this step | Required |
| lines_cleared_delta | int | Lines cleared at this step (0..4) | >=0 |
| holes | int | Current hole count | >=0 |
| max_height | int | Current max height | >=0 |
| timestamp | float | Wall-clock time (s since session start) | Required |
| keyframe | bool | True if stored for replay | Required |

### MetricStream (Logical / Derived)

Not persisted as separate rows; materialized through frame & episode aggregation.

| Metric | Source | Aggregation |
|--------|--------|-------------|
| reward_per_step | Frame.reward | sequence |
| episode_return | Episode.total_reward | list per episode |
| lines_cleared_cumulative | sum(Frame.lines_cleared_delta) | running sum |
| holes_trajectory | Frame.holes | sequence |
| max_height_trajectory | Frame.max_height | sequence |

### Configuration (Embedded in Session.config)

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| observation_mode | enum(`grid`,`features`,`rgb`) | Active observation representation | `features` |
| reward.line_clear_weight | float | Multiplier for squared lines cleared term | 1.0 |
| reward.hole_weight | float | Penalty weight per new hole | 0.35 |
| reward.height_weight | float | Penalty weight proportional to normalized height | 0.5 |
| reward.step_penalty | float | Per-step negative reward | 0.01 |
| early_stop.max_episodes | int | Hard training cap | 300 |
| early_stop.stagnation.no_line_steps | int | Steps with no lines before episode truncation | 500 |
| early_stop.training_plateau.window | int | Episodes window for plateau detection | 50 |
| early_stop.training_plateau.delta | float | Improvement threshold fraction | 0.01 |
| variance.window | int | Episodes window for variance warning | 20 |
| export.enable_csv | bool | Enable CSV export | true |
| export.enable_json | bool | Enable JSON export | true |
| eval.episodes | int | Episodes for evaluation-only mode | 10 |
| ui.update_interval_steps | int | Steps between UI refresh batches | 5 |
| replay.keyframe_stride | int | Store every Nth frame | 4 |

## Relationships

- Session 1—N Episode
- Episode 1—N Frame
- Session aggregates Episode metrics; Episode aggregates Frame metrics.

## Invariants & Validation Rules

- `episode.total_reward == sum(frame.reward)` across frames of episode (floating tolerance 1e-6)
- `lines_cleared == sum(frame.lines_cleared_delta)`
- If `interrupted=true` then `terminated=false` OR `truncated=false` (cannot be both terminated & interrupted simultaneously)
- `features` length stable per `observation_mode='features'`
- `rgb` only populated if observation_mode includes rgb OR evaluation replay requires it

## Extensibility Points

| Area | Mechanism | Example |
|------|-----------|---------|
| New observation mode | Wrapper producing alternate `board_repr` / `features` | Stacked recent frames temporal features |
| Additional reward term | Add weight + compute in reward wrapper; append to config | Penalty for overhangs |
| Extra board metric | Derive from board each step; append to Frame & features vector | Aggregate well depth |
| External logger | Hook on episode end to serialize Episode + frames | W&B integration |

## Open Considerations

- Potential memory optimization: compress `board_repr` using bit packing (not MVP)
- Option to externalize large replays to separate artifact store (future)

## Status

Complete for Phase 1 planning; no unresolved fields blocking contracts.
