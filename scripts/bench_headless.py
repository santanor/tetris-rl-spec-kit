from __future__ import annotations

import time
from tetris_rl.env.tetris_env import TetrisEnv


def run_bench(steps: int = 200000, seed: int = 0):
    env = TetrisEnv(seed=seed)
    state, _ = env.reset()
    t0 = time.time()
    done = False
    truncated = False
    i = 0
    reward_acc = 0.0
    while i < steps:
        # take a simple action pattern to exercise step/obs
        act = i % 5
        state, reward, done, truncated, info = env.step(act)
        reward_acc += reward
        if done or truncated:
            state, _ = env.reset()
        i += 1
    dt = time.time() - t0
    env.close()
    sps = steps / max(1e-9, dt)
    print(f"Bench: {steps} env-steps in {dt:.3f}s -> {sps:.1f} steps/s, reward_acc={reward_acc:.2f}")


if __name__ == "__main__":
    run_bench()
