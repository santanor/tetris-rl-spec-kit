# Phase 0 Research: Tetris RL Interpretability Demo

## Overview

This document resolves outstanding clarifications from the feature specification and establishes concrete, testable defaults for planning and design. Decisions prioritize: (1) Interpretability, (2) Reproducibility, (3) Extensibility, (4) Performance (fast iteration), in that order.

## Clarification Resolutions

| # | Topic | Decision | Rationale | Alternatives Considered |
|---|-------|----------|-----------|--------------------------|
| 1 | Partial episode handling on manual interrupt | Include partial episode metrics; tag as `interrupted=true`; compute per-step averages with actual steps only | Preserves learning signal; avoids data loss | Dropping episode (data loss); Auto-completing (fabricates data) |
| 2 | Environment reset failure | Retry once; if fails again, mark session `error` and abort training gracefully with collected summaries | Prevents infinite loops; surfaces env issues early | Unlimited retries (hang risk); Immediate abort (flaky transient fail) |
| 3 | Stagnation threshold (no lines cleared) | 500 consecutive steps with zero new lines → early terminate episode | Generous for baseline; improvement usually <200 steps | 200 (risk premature); 1000 (slower feedback) |
| 4 | Flat reward variance warning | Rolling window=20 episodes; warn if std(reward) < 5% of mean(abs(reward)+1e-6) after ≥40 episodes | Detects plateau without early noise | Fixed absolute threshold; KL divergence |
| 5 | Headless (non-visual) support | YES. Auto-detect; store replays JSON + optional MP4 | Enables CI/regression; reproducibility | Manual flag only |
| 6 | Initial observation modes | Grid (flattened), Engineered features, RGB frame (eval only) | Covers speed + insight + demo | Only grid (less insight); Only pixels (slow) |
| 7 | Minimum replay interactivity | Play/Pause, Step, Speed (1x/2x/4x), Slider | Adequate value vs complexity | Reverse step; bookmarking |
| 8 | Default stagnation/early stop | Max episodes=300 OR moving avg last 50 < baseline_mean + 1% after 150 | Prevents wasted compute | Only max episodes; complex schedules |
| 9 | Ineffective learning heuristics | Holes median +15% (30 ep), flat variance (#4), lines<1 (50 ep after 100) | Multi-signal detection | Single metric only |
|10 | Preferred export formats | JSON + CSV under `runs/{session_id}` | Flexible analysis | Single format |
|11 | Cross-platform reproducibility | Best-effort (Linux/macOS) pinned deps + seeds | Full determinism costly | Strict deterministic GPU |
|12 | Partial results inclusion rules | Include partial, tag; exclude from aggregates by default (toggle) | Data preserved, charts stable | Always include; always discard |
|13 | Saved policy eval-only mode | Included: load `policy.pt`, run N=10 eval episodes | Enhances demo | Defer feature |

## Additional Derived Decisions

- Reward shaping:
  - Base reward = environment lines cleared reward
  - Shaping components:
    - `-0.01` per step (encourage progress)
    - `+(lines_cleared^2) * line_clear_weight` (default `1.0`)
    - `- hole_weight * new_holes` (default `0.35`)
    - `- height_weight * (max_height / board_height)` (default `0.5`)
- Action space: Use env's discrete composite mapping (placement actions).
- Logging cadence: Per-step ring buffer for UI; flush end-of-episode.
- Frame storage: Keep keyframes (every 4th) for replay to bound memory.
- Seed strategy: Apply to Python, NumPy, Torch, env.
- Dependencies (pinned later): gymnasium, torch, numpy, tqdm, matplotlib, ipywidgets, imageio[ffmpeg] (optional).

## Rationale Highlights

Interpretability drives inclusion of engineered features and board heuristics. Performance concerns justify flattened grid for baseline training and sampled frame capture. Extensibility preserved via modular wrappers for reward shaping and observation transformation.

## Open Risks (Monitored)

| Risk | Mitigation |
|------|------------|
| UI refresh overhead slows training | Batch UI update every N steps (default 5) |
| Memory growth from stored frames | Keyframe sampling + per-session directory rotation |
| Non-deterministic GPU ops | Allow CPU mode for reproducibility checks |

## Final Status

All prior NEEDS CLARIFICATION items resolved; Ready for Phase 1 design.
