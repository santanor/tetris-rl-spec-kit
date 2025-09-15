# Tasks: Tetris RL Interpretability Demo

Input: Design documents in `specs/001-a-demo-that/` (plan.md, research.md, data-model.md, quickstart.md, contracts/*.json)
Prerequisites: Planning phases complete (research + design). This file defines Phase 2 output.

Legend:

- [P] = Can be executed in parallel (different files / no ordering dependency)
- All tests MUST be authored and observed failing before implementing corresponding code (TDD gate)
- Numbering is strict execution ordering except groups of [P] which may run together once predecessors complete

## High-Level Categories
 
1. Setup & Scaffolding
2. Tests First (Contract / Integration / Unit)
3. Core Domain Models (dataclasses / simple structures)
4. Core Functionality (wrappers, agent, metrics, replay, exporters)
5. UI & Notebook Integration
6. Integration Features (warnings, early stop, evaluation-only)
7. Exports & Validation
8. Polish & Performance & Docs

## Phase 3.1: Setup & Scaffolding
 
- [ ] T001 Create Python package structure `src/tetris_rl/` with subpackages: `agent`, `env`, `metrics`, `replay`, `ui`, `utils` and add `__init__.py` files.
- [ ] T002 Add project metadata: `pyproject.toml` (set name `tetris-rl-demo`, version 0.1.0, deps: gymnasium, torch, numpy, tqdm, matplotlib, ipywidgets, imageio[ffmpeg], jsonschema, pytest, ruff, black, mypy (optional)).
- [ ] T003 Configure lint & formatting: `ruff.toml`, Black line length, add `pyproject.toml` tool sections. Add minimal `mypy.ini` (allow untyped defs initially) [P].
- [ ] T004 Add `tests/` directory tree: `tests/contract`, `tests/integration`, `tests/unit`, add `conftest.py` with pytest fixtures placeholder.
- [ ] T005 Implement utility module stubs: `src/tetris_rl/utils/paths.py` (run directory resolution), `seeding.py` (set global seeds function) [P].
- [ ] T006 Add Git ignore patterns for `runs/`, `.ipynb_checkpoints/`, `__pycache__/`, `*.pt`, `*.mp4`.
- [ ] T007 CI placeholder (if CI used later) create `.github/workflows/placeholder.yml` (deferred optional) — mark as MAYBE (skip if repo policy forbids).

## Phase 3.2: Tests First (Author & Watch Them Fail)
 
### Contract Tests (from contracts/*.json)
 
- [ ] T008 [P] Contract test for metrics export schema validation in `tests/contract/test_metrics_export_schema.py` (load example invalid & minimal valid skeleton, expect failure until exporter implemented).
- [ ] T009 [P] Contract test for replay export schema validation in `tests/contract/test_replay_export_schema.py`.

### Integration Tests (from user stories & quickstart scenarios)
User Stories mapped: (US1 init, US2 live streaming, US3 episode summary, US4 training end + replay, US5 config isolation, Edge: interruption, evaluation-only mode, stagnation early stop, warnings heuristics, headless run fallback)
 
- [ ] T010 Scenario test notebook initialization (simulate calling init function) in `tests/integration/test_notebook_init.py` (assert config echo & seeds reproducibility placeholder fails).
- [ ] T011 Scenario test minimal training loop 5 episodes generating episode summaries in `tests/integration/test_training_loop_basic.py` (assert creation of runs directory & metrics.json placeholder → failing).
- [ ] T012 Scenario test replay selection entries (top reward / top lines / worst reward labels) in `tests/integration/test_replay_selection.py`.
- [ ] T013 Scenario test configuration isolation between two runs (no overwrite + distinct session ids) in `tests/integration/test_config_run_isolated.py`.
- [ ] T014 Scenario test graceful interruption produces tagged partial episode summary in `tests/integration/test_interruption_partial_episode.py`.
- [ ] T015 Scenario test evaluation-only mode loads saved policy & runs N eval episodes without training in `tests/integration/test_evaluation_mode.py`.
- [ ] T016 Scenario test stagnation early stop triggers before max episodes when no lines cleared for threshold steps in `tests/integration/test_stagnation_early_stop.py`.
- [ ] T017 Scenario test warning heuristics emit at least one warning file entry for synthetic poor config in `tests/integration/test_warning_heuristics.py`.
- [ ] T018 Scenario test headless mode (disable UI env var) still produces exports (metrics.json, episodes.csv) in `tests/integration/test_headless_mode.py`.

### Unit Tests (focused, fast)
 
- [ ] T019 [P] Unit test reward shaping wrapper (weights produce expected composed reward) in `tests/unit/test_reward_wrapper.py`.
- [ ] T020 [P] Unit test observation modes (features vs grid shape & dtype) in `tests/unit/test_observation_modes.py`.
- [ ] T021 [P] Unit test stagnation detector logic in `tests/unit/test_stagnation_detection.py`.
- [ ] T022 [P] Unit test warning heuristic functions (flat reward variance, holes growth) in `tests/unit/test_warning_heuristics.py`.
- [ ] T023 [P] Unit test replay keyframe stride logic in `tests/unit/test_replay_keyframe_stride.py`.
- [ ] T024 [P] Unit test DQN network forward pass shape & epsilon schedule in `tests/unit/test_dqn_forward.py`.

## Phase 3.3: Core Domain Models (Implement AFTER T008–T024 exist & fail)
 
- [ ] T025 [P] Implement dataclass `Session` in `src/tetris_rl/models/session.py` (id, timestamp, config, episodes summary aggregator, warnings list).
- [ ] T026 [P] Implement dataclass `Episode` in `src/tetris_rl/models/episode.py` (index, steps, total_reward, lines_cleared, holes_traj, height_traj, flags, termination_reason).
- [ ] T027 [P] Implement dataclass `Frame` in `src/tetris_rl/models/frame.py` (step, action, reward, board_metrics dict, optional rgb frame reference path or ndarray).
- [ ] T028 [P] Implement structure/class `MetricStream` in `src/tetris_rl/models/metric_stream.py` (append, window stats for rolling mean).
- [ ] T029 [P] Implement dataclass `Configuration` in `src/tetris_rl/models/config.py` (parameters from quickstart table + validation, generate session id helper).
- [ ] T030 [P] Implement dataclass `ReplayArtifact` in `src/tetris_rl/models/replay.py` (episode_ref, frames indices, highlight_reason).

## Phase 3.4: Core Functionality
 
- [ ] T031 Implement seeding utility in `src/tetris_rl/utils/seeding.py` (python, numpy, torch seeds + torch.backends deterministic flags) & update tests referencing it.
- [ ] T032 Implement run directory/path helpers in `src/tetris_rl/utils/paths.py` (generate runs/{session_id}/ structure).
- [ ] T033 Implement observation wrappers in `src/tetris_rl/env/observation_wrappers.py` (FeatureObservationWrapper, GridObservationWrapper, RGBEvalWrapper stub) with mode registry.
- [ ] T034 Implement reward shaping wrapper in `src/tetris_rl/env/reward_wrapper.py` applying weights & logging components.
- [ ] T035 Implement stagnation termination wrapper in `src/tetris_rl/env/stagnation_wrapper.py` (track steps since last line clear).
- [ ] T036 Implement metrics recorder in `src/tetris_rl/metrics/recorder.py` (per-step record, per-episode finalize, rolling averages via MetricStream) to satisfy contract tests later.
- [ ] T037 Implement warning heuristics in `src/tetris_rl/metrics/warnings.py` (flat reward variance, holes growth) returning structured warning dicts.
- [ ] T038 Implement exporters in `src/tetris_rl/metrics/exporters.py` to write `metrics.json`, `episodes.csv`, `frames_sample.csv` (schema compliance) — make contract tests pass.
- [ ] T039 Implement replay store & keyframe sampling in `src/tetris_rl/replay/store.py` (stride sampling, serialization references for frames).
- [ ] T040 Implement replay player utilities in `src/tetris_rl/replay/player.py` (iterator/generator for frames & metadata for UI).
- [ ] T041 Implement DQN network, replay buffer, optimizer step logic in `src/tetris_rl/agent/dqn.py` (minimal: network, select_action epsilon schedule, optimize step frequency, save/load policy).
- [ ] T042 Implement training orchestrator in `src/tetris_rl/agent/trainer.py` (loop over episodes, integrate wrappers, metrics recorder, exporters flush points, early stop detection hooks).

## Phase 3.5: UI & Notebook Integration
 
- [ ] T043 Implement notebook panel helpers in `src/tetris_rl/ui/notebook_panels.py` (widgets: live frame display, KPI panels, plots stubs hooking into recorder callbacks).
- [ ] T044 Create notebook skeleton `notebooks/tetris_interpretability.ipynb` with sections: Config, Initialize, Train, Replay, Evaluation Only (placeholders referencing API functions) — ensure integration tests can programmatically import utilities.
- [ ] T045 Implement headless mode detection (env var `HEADLESS=1`) to disable widget creation & still collect metrics in `ui/notebook_panels.py`.

## Phase 3.6: Integration Features
 
- [ ] T046 Integrate stagnation detection with trainer (episode termination reason propagation) & satisfy T016.
- [ ] T047 Integrate warning heuristic evaluation cadence (per-episode evaluation updating session warnings) & satisfy T017.
- [ ] T048 Implement graceful interruption handling (signal or KeyboardInterrupt catch in trainer) to finalize partial episode with `partial=True` flag & satisfy T014.
- [ ] T049 Implement evaluation-only mode entry point in `trainer.py` (load `policy.pt`, run episodes without optimization, export metrics) & satisfy T015.
- [ ] T050 Implement configuration isolation logic (unique session ids + no overwrite policy) & satisfy T013.

## Phase 3.7: Exports & Validation
 
- [ ] T051 Finalize metrics exporter to fully satisfy schema fields (fill session summary aggregates) & make T008 pass.
- [ ] T052 Finalize replay export writing function producing JSON structure conforming to replay schema or per-episode frames sample to make T009 pass.
- [ ] T053 Implement CSV writers for episodes & frames sample ensuring deterministic column order.
- [ ] T054 Add validation helpers in `metrics/exporters.py` (jsonschema.validate before write) used in integration tests.

## Phase 3.8: Polish, Performance, Docs
 
- [ ] T055 [P] Add unit tests for DQN replay buffer edge cases in `tests/unit/test_replay_buffer.py`.
- [ ] T056 [P] Add performance micro-benchmark script `scripts/benchmark_env_loop.py` ensure >200 steps/sec headless for 1000 steps.
- [ ] T057 [P] Documentation: Update `quickstart.md` with any new config params & add troubleshooting entries for stagnation & headless mode.
- [ ] T058 [P] Add README section linking to notebook & describing feature modules.
- [ ] T059 [P] Static type pass: tighten mypy configuration (disallow untyped defs in src/tetris_rl) & fix issues.
- [ ] T060 Refactor pass: remove duplication, ensure function sizes reasonable (<60 lines where practical).
- [ ] T061 Manual scenario validation script `scripts/validate_scenarios.py` (invokes minimal training run & assertions) and record results in `docs/validation.md`.
- [ ] T062 Final sweep: ensure all tests green, run benchmark, capture session example artifacts under `examples/`.

## Dependencies Overview

- T001–T007 must precede all tests.
- Contract tests (T008–T009) independent of integration/unit tests -> may run in parallel with each other.
- Integration tests T010–T018 depend only on setup (T001–T007) not on model impl; they should fail initially.
- Unit tests T019–T024 depend only on setup; fail until implementation.
- Core model tasks T025–T030 wait for test suite presence (T008–T024) but can run in parallel among themselves.
- Functionality tasks T031–T042 depend on models they use (e.g., exporter depends on Session/Episode/Frame).
- UI tasks T043–T045 depend on basic trainer & recorder (T036, T041, T042).
- Integration features T046–T050 depend on earlier wrappers & trainer.
- Export validation tasks T051–T054 refine exporters to make contract tests pass (some overlap; rerun tests after).
- Polish tasks T055–T062 follow all earlier tasks; some parallel [P].

## Parallel Execution Example

Group after setup (post T007) you can launch these in parallel:
```text
Task: T008 Contract test metrics schema
Task: T009 Contract test replay schema
Task: T010 Integration test notebook initialization
Task: T019 Unit test reward shaping wrapper
Task: T020 Unit test observation modes
Task: T021 Unit test stagnation detection
Task: T022 Unit test warning heuristics
Task: T023 Unit test replay keyframe stride
Task: T024 Unit test DQN forward pass
```

## Validation Checklist (Use before marking complete)

- [ ] All contract & integration tests authored before any implementation.
- [ ] All entity dataclasses implemented with docstrings & type hints.
- [ ] Exporters pass jsonschema validation (T008, T009 green).
- [ ] Training loop integration test passes producing artifacts.
- [ ] Replay selection identifies episodes by metrics.
- [ ] Warning heuristics produce at least one warning for crafted config.
- [ ] Headless mode tests produce metrics without UI errors.
- [ ] Performance benchmark meets >200 steps/sec (features mode) goal.
- [ ] Quickstart instructions remain accurate & reproducible.

## Notes

- Keep functions cohesive; prefer simple procedural trainer over premature abstraction.
- Avoid over-optimizing early; focus on correctness & interpretability signals.
- Re-run integration tests after each major subsystem (metrics, trainer, exporters, replay) implementation.
- Consider adding a small synthetic mock env for ultra-fast unit tests if Gym Tetris proves slow.

END OF TASKS
