#!/usr/bin/env python3
"""Simple headless training demo script."""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tetris_rl.agent.trainer import run_training, TrainingConfig


def main():
    """Run a simple training demo."""
    # Create output directory
    output_dir = Path("training_runs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure training for a quick demo
    config = TrainingConfig(
        episodes=10,
        max_steps=200,
        hidden_layers=[64, 64],
        device="auto"
    )
    
    print(f"Starting training demo: {config.episodes} episodes")
    print(f"Output directory: {output_dir}")
    
    # Run training
    session = run_training(output_dir, config)
    
    print(f"Training completed!")
    print(f"Total episodes: {len(session.episodes)}")
    print(f"Average reward: {session.avg_reward:.3f}")
    print(f"Average lines cleared: {session.avg_lines_cleared:.1f}")


if __name__ == "__main__":
    main()