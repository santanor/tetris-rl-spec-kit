# Feature Specification: Tetris RL Interpretability Demo

**Feature Branch**: `001-a-demo-that`  
**Created**: 2025-09-15  
**Status**: Draft  
**Input**: User description: "A demo that trains and evaluates an RL agent on Tetris using the Gymnasium API, with a notebook-based, real‑time visualization of actions, rewards, and board health to make learning interpretable and engaging. Purpose: Showcase a complete RL workflow end‑to‑end on a familiar game, emphasizing transparency of agent behavior through live visuals and metrics. Provide a modular baseline that can be extended (reward shaping, observation choices, wrappers) while keeping a clean, reproducible interface."

## Execution Flow (main)

```text
1. Parse user description from Input
	-> DONE
2. Extract key concepts from description
	-> Concepts: RL training loop, Tetris environment (Gymnasium), real-time notebook visualization, actions, rewards, board health metrics, transparency, extensibility (reward shaping, observation variants, wrappers), baseline reproducibility
3. For each unclear aspect:
	-> Marked below in Requirements & Scenarios with [NEEDS CLARIFICATION]
4. Fill User Scenarios & Testing section
	-> DONE
5. Generate Functional Requirements
	-> DONE (with clarification markers)
6. Identify Key Entities
	-> DONE (Agent Session, Episode, Frame Visualization, Metric Stream, Configuration)
7. Run Review Checklist
	-> PENDING (needs stakeholder confirmation & removal of clarification markers)
8. Return: SUCCESS (spec ready for planning when clarifications resolved)
```

---

## ⚡ Quick Guidelines

- ✅ Focus on WHAT and WHY (educational interpretability demo)
- ❌ Avoid implementation specifics (no library choices beyond mandated Gymnasium API reference as part of feature scope)
- 👥 Audience: stakeholders evaluating educational value & extensibility of RL baseline

### Section Notes

All mandatory sections completed. Key optional detailing (entities) included due to data & visualization nature.

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

As an ML practitioner or learner, I want to run a single notebook that trains a baseline RL agent on Tetris and simultaneously shows live actions, rewards, and board health so I can understand how the agent is learning and diagnose ineffective reward shaping or observation choices early.

### Acceptance Scenarios

1. Given the notebook is opened fresh, When the user runs the "Initialize & Configure" cell, Then the baseline configuration (default environment, default reward function, default observation space) is loaded and displayed for confirmation.
2. Given training has started, When an episode is progressing, Then the notebook updates a live panel showing: current frame (board & falling piece), chosen action, step reward, cumulative episode reward, lines cleared, board height/holes metric.
3. Given an episode ends, When the next episode begins, Then prior episode summary metrics (total reward, lines cleared, duration (steps), average reward/step) are appended to a results table and plotted cumulatively.
4. Given training reaches a predefined max episodes or early stop condition, When training loop finishes, Then the notebook displays: final aggregate metrics, learning curves (reward vs episodes, lines cleared vs episodes), and a selectable replay of notable episodes (highest reward, lowest reward, most lines cleared).
5. Given the user modifies a reward shaping parameter cell (e.g., hole penalty weight) and re-runs the training cell, When training restarts, Then previous run results remain available for comparison and new run is labeled with a distinct configuration identifier.

### Edge Cases

- What happens when training is interrupted manually? → System SHOULD gracefully stop loop and still finalize metrics for the current partially completed episode. [NEEDS CLARIFICATION: Should partial episodes be excluded or tagged?]
- How does system handle environment reset failures? → [NEEDS CLARIFICATION: Define expected fallback if environment raises an exception]
- Extremely long episodes (agent stalls) → System SHOULD detect stagnation (no lines cleared over X steps) and terminate episode early. [NEEDS CLARIFICATION: Value of X]
- Zero-improvement runs (flat reward curve) → System SHOULD surface a warning annotation in plots when rolling mean reward variance below threshold. [NEEDS CLARIFICATION: Threshold definition]
- Notebook run without display (e.g., headless CI) → Visualization components SHOULD degrade to stored static artifacts (images / JSON logs) instead of real-time updates. [NEEDS CLARIFICATION: Are headless runs required?]

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a single executable notebook that orchestrates configuration, training, live visualization, and evaluation.
- **FR-002**: The system MUST support initializing a Gymnasium-compatible Tetris environment using a default baseline configuration.
- **FR-003**: The system MUST stream per-step agent decisions (action taken and its meaning) during training to a live display area.
- **FR-004**: The system MUST compute and surface per-step reward, cumulative episode reward, and lines cleared.
- **FR-005**: The system MUST compute board health metrics (e.g., aggregate column height, number of holes, bumpiness) each step and display at least one trend indicator.
- **FR-006**: The system MUST allow users to adjust reward shaping parameters before (re)starting training.
- **FR-007**: The system MUST allow users to toggle or select between alternative observation representations (e.g., raw grid, engineered features). [NEEDS CLARIFICATION: List of initial observation modes]
- **FR-008**: The system MUST record each training run with a configuration identifier and persist summary metrics for later comparison.
- **FR-009**: The system MUST present episode-level summary metrics in a tabular form and cumulative plots (reward curve, lines cleared curve).
- **FR-010**: The system MUST provide a replay mechanism for selected episodes (frame sequence with timing or step-through controls). [NEEDS CLARIFICATION: Minimum required replay interactivity]
- **FR-011**: The system MUST highlight notable episodes (top reward, top lines cleared, worst reward) automatically after training completes.
- **FR-012**: The system MUST support an early stopping condition based on either max episodes or stagnation criteria. [NEEDS CLARIFICATION: Default stagnation parameters]
- **FR-013**: The system MUST surface warnings when training signals suggest ineffective learning (e.g., flat reward, high holes growth). [NEEDS CLARIFICATION: Exact heuristic definitions]
- **FR-014**: The system MUST provide a clearly separated section for future extensions (wrappers, alternative agents) without impacting baseline reproducibility.
- **FR-015**: The system MUST allow re-running training without overwriting prior run result artifacts within the same notebook session.
- **FR-016**: The system MUST summarize final training outcomes in a concise dashboard (key metrics, charts, run configurations list).
- **FR-017**: The system MUST make all metrics exportable (structured format: JSON or CSV) for offline analysis. [NEEDS CLARIFICATION: Preferred format(s)]
- **FR-018**: The system MUST retain deterministic reproducibility when a seed value is specified (environment, agent initialization, any stochastic elements). [NEEDS CLARIFICATION: Do users require cross-platform reproducibility guarantees?]
- **FR-019**: The system SHOULD gracefully handle manual interruption and still produce partial results summary. [NEEDS CLARIFICATION: Partial result inclusion rules]
- **FR-020**: The system SHOULD allow switching to an evaluation-only mode using a saved policy to produce interpretability visuals without training. [NEEDS CLARIFICATION: Are saved policies in scope for MVP?]

### Key Entities

- **Agent Session**: A single initiated training/evaluation run; attributes: session id, timestamp, seed, config params (reward weights, observation mode), status, aggregated metrics.
- **Episode**: A sequence of environment steps until terminal condition; attributes: episode index, steps, total reward, lines cleared, max height, holes trajectory, notable flags.
- **Frame**: A single environment state snapshot; attributes: step index, board representation, active piece, action chosen, immediate reward, computed board metrics.
- **Metric Stream**: Time-ordered collection of computed values (reward per step, holes count, lines cleared cumulative) bound to a session & episode.
- **Configuration**: Parameter bundle defining environment options, reward shaping weights, observation type, early stopping settings.
- **Replay Artifact**: Ordered list of Frames selected for playback plus metadata (episode reference, highlight reason).

---

## Review & Acceptance Checklist

GATE: Requires removal or resolution of all [NEEDS CLARIFICATION] markers before moving to implementation planning.

### Content Quality

- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness

- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---

## Execution Status

Updated by main() during processing.

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarifications)

---

## Outstanding Clarifications (Summary)

1. Partial episode handling upon manual interruption
2. Environment reset failure fallback behavior
3. Stagnation detection threshold (X steps with no lines cleared)
4. Flat reward variance threshold for warning
5. Need for headless (non-visual) execution support
6. Initial list of observation modes to include in MVP
7. Minimum replay interactivity (play/pause vs step vs speed control)
8. Default stagnation/early stop parameters
9. Heuristics for ineffective learning warnings (exact definitions)
10. Preferred export format(s) (JSON, CSV, both)
11. Cross-platform reproducibility requirement
12. Inclusion rules for partial results after interruption
13. Saved policy evaluation-only mode in MVP or deferred

---

## Out of Scope (Explicit)

- Implementing multiple advanced RL algorithms beyond a single baseline policy (e.g., multiple agent classes)
- Cloud deployment or multi-user sharing platform
- Persistent database storage of runs (artifact persistence limited to notebook session / local files)
- Advanced hyperparameter search / tuning automation
- Real-time multiplayer or competitive agent benchmarks
- Security / authentication layers (single-user educational context)

---

## Success Indicators (High-Level)

- User can run notebook end-to-end without modifying code cells (configuration via parameters only)
- Live visualization updates smoothly (per-step or batched) during training
- At least one interpretability insight surfaced (e.g., highlighting a high-hole state leading to negative reward trend)
- Ability to compare at least two distinct configuration runs in one session
- Exported metrics loadable externally for analysis

---

## Assumptions

- A Gymnasium-compatible Tetris environment is (or will be) available
- Users have minimal RL familiarity (baseline explanations beneficial)
- Single-user local execution environment (no concurrency needs)
- Notebook runtime has sufficient performance for live updates (reasonable step latency)

---

## Risks

- Visualization overhead may slow training (need batching strategy later—implementation detail deferred)
- Ambiguous reward shaping could mislead users if defaults poorly chosen
- Replay feature scope creep (keep minimal for MVP pending clarification)

---

This specification is ready for clarification review. Once the outstanding items are resolved, it can transition to planning.

