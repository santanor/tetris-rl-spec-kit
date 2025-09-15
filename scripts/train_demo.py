#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from tetris_rl.agent.trainer import run_training, TrainingConfig


def main():
    output = Path("training_runs/demo")
    cfg = TrainingConfig(episodes=5, max_steps=200, min_replay=50, batch_size=32, epsilon_decay=300)
    session = run_training(output, cfg)
    print("Training complete")
    print("Episodes:", session.episodes_total)
    print("Avg reward:", session.avg_reward)
    print("Artifacts in:", output.resolve())

if __name__ == "__main__":
    main()
