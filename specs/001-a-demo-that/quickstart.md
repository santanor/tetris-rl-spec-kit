# Quickstart: Tetris RL Interpretability Demo

## 1. Environment Setup

Install Python 3.11+.

Create virtual environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies (placeholder; pinning added during implementation):

```bash
pip install gymnasium torch numpy tqdm matplotlib ipywidgets imageio[ffmpeg]
```
Enable Jupyter widget extension if required by environment (e.g., JupyterLab 3+ usually fine).

## 2. Launch Notebook

```bash
jupyter lab
```

Open `notebooks/tetris_interpretability.ipynb` (to be created) and run cells top to bottom.

## 3. Configuration Overview

| Parameter | Purpose | Default |
|-----------|---------|---------|
| observation_mode | State representation for agent (`features`,`grid`,`rgb`) | features |
| seed | Reproducibility seed | 42 |
| max_episodes | Hard cap on training episodes | 300 |
| line_clear_weight | Reward weight for squared lines cleared | 1.0 |
| hole_weight | Penalty per new hole | 0.35 |
| height_weight | Penalty proportional to normalized max height | 0.5 |
| step_penalty | Small per-step penalty | 0.01 |
| stagnation_no_line_steps | Steps with no lines before truncation | 500 |
| plateau_window | Episodes window for plateau detection | 50 |
| plateau_delta | Minimum improvement fraction | 0.01 |
| eval_episodes | Episodes when in evaluation-only mode | 10 |
| ui_update_interval_steps | Steps between UI refresh batches | 5 |
| replay_keyframe_stride | Store every Nth frame | 4 |

## 4. Running Training

1. Run Initialize cell → prints config & seeds.
2. Run Train cell → starts loop; live panels update every `ui_update_interval_steps`.
3. Interrupt (Kernel -> Interrupt) gracefully: partial episode stored & flagged.

## 5. Live Visualization Panels

- Left: Video (rgb_array) with overlays (action label, step reward, holes, height)
- Right/Top KPIs: Episode reward (running), Lines cleared, Exploration rate (if epsilon-greedy), Warning badges
- Lower Plots: Reward curve (rolling mean), Lines cleared per episode, Holes median trajectory

## 6. Replay

After training: choose episode (top reward / top lines / worst reward) and use controls (Play/Pause, Step, Speed, Slider).

## 7. Export Artifacts

Artifacts written under `runs/{session_id}/`:

- `metrics.json` (session summary, per-episode metrics, warnings)
- `episodes.csv` (episode-level table)
- `frames_sample.csv` (sampled keyframes)
- `config.json` (exact configuration used)
- Optional: `replay_{episode_id}.mp4` (if ffmpeg available)

## 8. Evaluation-Only Mode

If `EVAL_ONLY=True` (or CLI flag / notebook toggle), attempts to load `policy.pt` from latest session directory and runs interpretability pipeline for `eval_episodes` episodes without updating weights.

## 9. Extending

Add new reward term: edit reward wrapper class (no change to env). Add new observation mode: implement observation wrapper returning consistent shape and register in mode map. Add metric overlay: implement function producing text overlay data.

## 10. Troubleshooting

| Issue | Cause | Resolution |
|-------|-------|-----------|
| Slow UI | Too frequent updates | Increase `ui_update_interval_steps` |
| High memory | Many frames stored | Increase `replay_keyframe_stride` |
| No video in headless | Renderer disabled | Inspect `runs/{session_id}/` JSON replays |
| Non-deterministic results | GPU kernels nondet | Force CPU / set torch deterministic flags |

## 11. Success Criteria Validation

- All plots render without manual modification
- At least one warning triggers on intentionally poor config (e.g., zero line_clear_weight)
- Replay playable (controls responsive <250ms latency)
- Re-running training adds a new session directory without overwriting previous

## 12. Next Steps

Proceed to tasks generation (`tasks.md`) after design approval to implement environment wrappers, agent baseline (DQN), UI components, logging, and exports following TDD.
