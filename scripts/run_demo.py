#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from tetris_rl.env import DummyTetrisEnv
from tetris_rl.metrics.recorder import run_session


def main():
    output = Path("demo_runs/latest")
    session = run_session(lambda: DummyTetrisEnv(max_steps=50), episodes=3, output_dir=output)
    print("Demo session complete")
    print("Episodes:", session.episodes_total)
    print("Avg reward:", session.avg_reward)
    print("Output written to:", output.resolve())

if __name__ == "__main__":
    main()
