#!/usr/bin/env python
"""Bayesian (Optuna) reward shaping search harness.

This script performs a multi-fidelity Bayesian optimization over reward configuration
parameters for Tetris RL. It reuses the existing DQN components, running short training
stints to evaluate a sampled reward config and reporting an objective back to Optuna.

Two stages supported:
 - Stage 1: shorter episodes count (episodes_stage1) with pruning.
 - Stage 2 (optional re-run): Use --final-eval to re-evaluate best N with longer runs & more seeds.

Usage (basic):
  python scripts/reward_sweep.py --trials 60 --episodes-stage1 40 --episodes-stage2 120 --seeds 2 --device auto

Produces an artifacts directory with JSON summary of top configurations.

Environment Var Hints:
  HEADLESS=1 to suppress dashboard broadcasts if integrating into the FastAPI app context.

NOTE: This harness intentionally omits replay buffer persistence and does *not* attempt
to perfectly replicate the web training loop multi-env batching. Its goal is relative ranking
under a consistent protocol, not absolute best final reward.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

import torch
import optuna

from tetris_rl.agent.trainer import TrainingConfig, _resolve_device, optimize, _compute_epsilon, _compute_temperature
from tetris_rl.agent.dqn import DQN
from tetris_rl.agent.replay_buffer import ReplayBuffer
from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.env.reward_config import RewardConfig

# ---------------------- Configuration Structures ---------------------- #
@dataclass
class EvalResult:
    mean_reward: float
    final_reward: float
    auc_reward: float
    mean_lines: float
    final_lines: float
    mean_structural: Dict[str, float]
    config: Dict[str, Any]
    seed: int
    episodes: int

# ---------------------- Sampling Space Helpers ----------------------- #

def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def suggest_reward_config(trial: optuna.Trial) -> RewardConfig:
    rc = RewardConfig()
    # Core line rewards
    rc.line_reward_1 = trial.suggest_float("line_reward_1", 0.05, 0.4)
    rc.line_reward_2 = trial.suggest_float("line_reward_2", 0.1, 0.8)
    rc.line_reward_3 = trial.suggest_float("line_reward_3", 0.2, 1.4)
    rc.line_reward_4 = trial.suggest_float("line_reward_4", 0.5, 3.0)
    # Survival / temporal shaping
    rc.survival_bonus = trial.suggest_float("survival_bonus", 0.0, 0.02)
    rc.step_penalty = trial.suggest_float("step_penalty", -0.015, 0.0)
    rc.top_out_penalty = trial.suggest_float("top_out_penalty", -5.0, -0.5)
    # Structural weights (log scale)
    rc.holes_weight = trial.suggest_float("holes_weight", 1e-4, 5e-2, log=True)
    rc.holes_abs_weight = trial.suggest_float("holes_abs_weight", 1e-4, 5e-2, log=True)
    rc.weighted_holes_weight = trial.suggest_float("weighted_holes_weight", 1e-4, 5e-2, log=True)
    rc.weighted_holes_abs_weight = trial.suggest_float("weighted_holes_abs_weight", 1e-4, 5e-2, log=True)
    rc.height_weight = trial.suggest_float("height_weight", 1e-4, 1e-2, log=True)
    rc.bumpiness_weight = trial.suggest_float("bumpiness_weight", 1e-4, 1e-2, log=True)
    rc.row_density_delta_weight = trial.suggest_float("row_density_delta_weight", 1e-4, 5e-2, log=True)
    rc.row_density_abs_weight = trial.suggest_float("row_density_abs_weight", 1e-4, 5e-2, log=True)
    rc.row_density_line_clear_scale = trial.suggest_float("row_density_line_clear_scale", 0.0, 3.0)
    rc.holes_depth_power = trial.suggest_float("holes_depth_power", 0.8, 3.2)
    return rc

# ---------------------- Evaluation Harness --------------------------- #

def evaluate_config(
    rc: RewardConfig,
    episodes: int,
    seed: int,
    device_str: str,
    hidden_layers: List[int],
    dueling: bool,
    use_layer_norm: bool,
    dropout: float,
    max_steps: int,
    batch_size: int,
    replay_capacity: int,
    target_sync: int,
    exploration_strategy: str,
) -> EvalResult:
    cfg = TrainingConfig(
        episodes=episodes,
        device=device_str,
        hidden_layers=hidden_layers,
        dueling=dueling,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        max_steps=max_steps,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_sync=target_sync,
        exploration_strategy=exploration_strategy,
    )
    device = _resolve_device(cfg.device)
    rng = random.Random(seed)
    policy = DQN((4,), 5, hidden_layers=hidden_layers, dueling=dueling, use_layer_norm=use_layer_norm, dropout=dropout).to(device)
    target = DQN((4,), 5, hidden_layers=hidden_layers, dueling=dueling, use_layer_norm=use_layer_norm, dropout=dropout).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(capacity=replay_capacity, seed=seed)
    global_step = 0

    rewards: List[float] = []
    lines_list: List[int] = []
    structural_acc = {"holes": 0.0, "height": 0.0, "bumpiness": 0.0}
    structural_steps = 0

    for ep in range(episodes):
        env = TetrisEnv(seed=rng.randint(0, 1_000_000), max_steps=max_steps, reward_config=rc)
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            # Epsilon handling
            eps = float(_compute_epsilon(global_step, cfg))
            if exploration_strategy == "epsilon_greedy":
                if random.random() < eps:
                    action = random.randrange(5)
                else:
                    with torch.no_grad():
                        q = policy(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(q.argmax().item())
            else:  # boltzmann
                with torch.no_grad():
                    q = policy(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
                temp = _compute_temperature(global_step, cfg)
                logits = q / temp
                probs = torch.softmax(logits, dim=0)
                action = int(torch.multinomial(probs, 1).item())

            next_state, reward, done, truncated, info = env.step(action)
            replay.push(state, action, reward, next_state, (done or truncated))
            state = next_state
            ep_reward += reward

            loss = optimize(policy, target, replay, cfg, optimizer, device)
            global_step += 1
            if global_step % target_sync == 0:
                target.load_state_dict(policy.state_dict())
            # Structural tracking
            rc_comp = info.get("reward_components") or {}
            sb = rc_comp.get("structural_breakdown") or {}
            for k in structural_acc:
                structural_acc[k] += float(sb.get(k, 0.0))
            structural_steps += 1
        rewards.append(ep_reward)
        lines_list.append(env.board.lines_cleared_total)
        env.close()

    mean_reward = sum(rewards) / len(rewards)
    mean_lines = sum(lines_list) / len(lines_list)
    # AUC (simple average surrogate) — could replace with trapezoidal integration if desired
    auc = sum(rewards) / len(rewards)
    mean_struct = {k: (v / structural_steps if structural_steps else 0.0) for k, v in structural_acc.items()}
    return EvalResult(
        mean_reward=mean_reward,
        final_reward=rewards[-1],
        auc_reward=auc,
        mean_lines=mean_lines,
        final_lines=lines_list[-1],
        mean_structural=mean_struct,
        config=rc.to_dict(),
        seed=seed,
        episodes=episodes,
    )

# ---------------------- Optuna Objective ----------------------------- #

def make_objective(args, arch, stage: int):
    def objective(trial: optuna.Trial):
        rc = suggest_reward_config(trial)
        seeds = [args.base_seed + s for s in range(args.seeds_stage1 if stage == 1 else args.seeds_stage2)]
        episodes = args.episodes_stage1 if stage == 1 else args.episodes_stage2
        results: List[EvalResult] = []
        for s in seeds:
            er = evaluate_config(
                rc=rc,
                episodes=episodes,
                seed=s,
                device_str=args.device,
                hidden_layers=arch['hidden_layers'],
                dueling=arch['dueling'],
                use_layer_norm=arch['use_layer_norm'],
                dropout=arch['dropout'],
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_sync=args.target_sync,
                exploration_strategy=args.exploration_strategy,
            )
            results.append(er)
            # Intermediate report using running mean of mean_reward
            current_mean = statistics.mean(r.mean_reward for r in results)
            trial.report(current_mean, step=len(results))
            if trial.should_prune():
                raise optuna.TrialPruned()
        # Aggregate final objective: weighted combo
        mean_mean_reward = statistics.mean(r.mean_reward for r in results)
        mean_lines = statistics.mean(r.mean_lines for r in results)
        # Combine (tune weights if you prefer) — prioritize reward but include lines
        objective_value = mean_mean_reward + 0.1 * mean_lines
        trial.set_user_attr("detail", {
            "stage": stage,
            "results": [asdict(r) for r in results],
            "aggregate": {
                "mean_mean_reward": mean_mean_reward,
                "mean_lines": mean_lines,
            }
        })
        return objective_value
    return objective

# ---------------------- Persistence Helpers -------------------------- #

def dump_study(study: optuna.Study, path: Path):
    data = []
    for t in study.trials:
        if t.state.name.lower() == 'pruned':
            continue
        entry = {
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'user_attrs': t.user_attrs,
            'state': t.state.name,
        }
        data.append(entry)
    with path.open('w') as f:
        json.dump(data, f, indent=2)

# ---------------------- Main ----------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Bayesian reward shaping sweep (Optuna)")
    p.add_argument('--trials', type=int, default=40)
    p.add_argument('--episodes-stage1', type=int, default=40)
    p.add_argument('--episodes-stage2', type=int, default=120)
    p.add_argument('--seeds-stage1', type=int, default=1)
    p.add_argument('--seeds-stage2', type=int, default=2)
    p.add_argument('--device', default='auto')
    p.add_argument('--exploration-strategy', default='epsilon_greedy', choices=['epsilon_greedy','boltzmann'])
    p.add_argument('--hidden-layers', default='256,256,256')
    p.add_argument('--dueling', action='store_true')
    p.add_argument('--layer-norm', action='store_true')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--replay-capacity', type=int, default=20_000)
    p.add_argument('--target-sync', type=int, default=500)
    p.add_argument('--storage', default=None, help='Optuna storage URL (e.g. sqlite:///sweep.db)')
    p.add_argument('--study-name', default='reward_sweep')
    p.add_argument('--pruner-startup-trials', type=int, default=8)
    p.add_argument('--pruner-warmup-steps', type=int, default=1)
    p.add_argument('--out-dir', default='sweep_artifacts')
    p.add_argument('--base-seed', type=int, default=1234)
    p.add_argument('--final-topk', type=int, default=5, help='How many best configs to re-eval at stage2')
    p.add_argument('--final-eval', action='store_true', help='Only perform final evaluation on existing study best trials')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',') if x.strip()]
    arch = {
        'hidden_layers': hidden_layers,
        'dueling': bool(args.dueling),
        'use_layer_norm': bool(args.layer_norm),
        'dropout': float(args.dropout),
    }

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
    )
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.base_seed)

    storage = args.storage
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        pruner=pruner,
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(storage)
    )

    if not args.final_eval:
        objective_stage1 = make_objective(args, arch, stage=1)
        study.optimize(objective_stage1, n_trials=args.trials, show_progress_bar=True)
        dump_study(study, out_dir / 'study_stage1.json')

    # Gather top-K after stage1
    trials_sorted = [t for t in study.trials if t.value is not None and t.state.name=='COMPLETE']
    trials_sorted.sort(key=lambda t: t.value, reverse=True)
    topk = trials_sorted[: args.final_topk]

    # Stage 2 re-evaluation (longer episodes & more seeds) – we don't re-sample; we re-score existing params.
    stage2_results = []
    for t in topk:
        params = t.params
        # Rebuild RewardConfig from params (using suggest ranges names)
        rc = RewardConfig()
        for k, v in params.items():
            if hasattr(rc, k):
                setattr(rc, k, v)
        seeds = [args.base_seed + 10_000 + s for s in range(args.seeds_stage2)]
        per_seed = []
        for s in seeds:
            er = evaluate_config(
                rc=rc,
                episodes=args.episodes_stage2,
                seed=s,
                device_str=args.device,
                hidden_layers=arch['hidden_layers'],
                dueling=arch['dueling'],
                use_layer_norm=arch['use_layer_norm'],
                dropout=arch['dropout'],
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_sync=args.target_sync,
                exploration_strategy=args.exploration_strategy,
            )
            per_seed.append(asdict(er))
        agg_mean_reward = statistics.mean(p['mean_reward'] for p in per_seed)
        agg_mean_lines = statistics.mean(p['mean_lines'] for p in per_seed)
        stage2_results.append({
            'trial_number': t.number,
            'value_stage1': t.value,
            'aggregate_mean_reward': agg_mean_reward,
            'aggregate_mean_lines': agg_mean_lines,
            'params': params,
            'per_seed': per_seed,
        })
    stage2_results.sort(key=lambda x: x['aggregate_mean_reward'], reverse=True)

    with (out_dir / 'stage2_results.json').open('w') as f:
        json.dump(stage2_results, f, indent=2)

    # Write a concise BEST summary
    if stage2_results:
        best = stage2_results[0]
        with (out_dir / 'best_reward_config.json').open('w') as f:
            json.dump({
                'params': best['params'],
                'aggregate_mean_reward': best['aggregate_mean_reward'],
                'aggregate_mean_lines': best['aggregate_mean_lines']
            }, f, indent=2)
        print("Best config written to", out_dir / 'best_reward_config.json')
    else:
        print("No stage2 results produced.")

if __name__ == '__main__':
    main()
