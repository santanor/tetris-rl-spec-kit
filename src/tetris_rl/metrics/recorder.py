"""Simple synchronous runner utilities for manual demo."""
from __future__ import annotations

from typing import Callable, Any
from pathlib import Path
import uuid

from tetris_rl.models.session import Session
from tetris_rl.models.episode import Episode
from tetris_rl.metrics.exporters import export_session_metrics, export_session_episodes


def play_episode(env, episode_index: int) -> Episode:
    ep = Episode(index=episode_index)
    obs, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # For dummy env we estimate lines delta if reward>0
        lines_delta = 1 if reward > 0 else 0
        holes = int(obs[2]) if len(obs) > 2 else 0
        height = int(obs[3]) if len(obs) > 3 else 0
        ep.record_step(reward, lines_delta, holes, height)
        done = terminated
    ep.finalize(terminated=done, truncated=truncated, interrupted=False, reason=None)
    return ep


def run_session(env_factory: Callable[[], Any], episodes: int, output_dir: Path) -> Session:
    session = Session(session_id=str(uuid.uuid4()), mode="demo", seed=0, config={})
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(episodes):
        env = env_factory()
        episode = play_episode(env, i)
        session.record_episode(episode.total_reward, episode.lines_cleared, episode)
        env.close()
    session.finish()
    export_session_metrics(session, output_dir / "metrics.json")
    export_session_episodes(session, output_dir / "episodes.csv")
    return session
