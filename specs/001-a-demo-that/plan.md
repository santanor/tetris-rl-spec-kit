# Implementation Plan: Tetris RL Interpretability Demo

**Branch**: `001-a-demo-that` | **Date**: 2025-09-15 | **Spec**: `specs/001-a-demo-that/spec.md`
**Input**: Feature specification from `/specs/001-a-demo-that/spec.md`

## Execution Flow (/plan command scope)

```text
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:

- Phase 2: /tasks command creates tasks.md
- Phase 3: Task execution (/tasks command creates tasks.md)
- Phase 4: Implementation (execute tasks.md following constitutional principles)
- Phase 5: Validation (run tests, execute quickstart, performance validation)

## Summary

Build a reproducible, interpretable RL baseline training a DQN-style agent on a Gymnasium Tetris environment with a notebook UI that streams per-step actions, rewards, and board health metrics, enables replay, and exports structured metrics (JSON/CSV). Research phase resolved all clarifications (reward shaping, stagnation detection, observation modes, evaluation-only mode).

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: gymnasium, torch, numpy, tqdm, matplotlib, ipywidgets, imageio[ffmpeg]  
**Storage**: Local filesystem for run artifacts (JSON/CSV/MP4)  
**Testing**: pytest (unit + integration), jsonschema validation for exports  
**Target Platform**: Local Linux/macOS (CPU baseline; optional GPU)  
**Project Type**: single  
**Performance Goals**: Maintain >200 env steps/sec headless (features mode) on CPU; UI refresh latency <250ms; replay load <2s for 500-step episode  
**Constraints**: Deterministic seeding best-effort; memory footprint under 1GB for 300 episodes with keyframe stride=4  
**Scale/Scope**: Single-user educational notebook; ~300 training episodes baseline

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:

- Projects: 1 (src + tests) — within limit
- Using framework directly: Yes (Gymnasium env directly; wrappers only for reward/obs shaping — justified)
- Single data model: Yes (session/episode/frame modeled via simple Python dataclasses or dicts)
- Avoiding patterns: No unnecessary repositories/UoW

**Architecture**:

- Feature as library: Training logic packaged under `src/tetris_rl/`
- Libraries: `tetris_rl` (env wrappers, agent, metrics, export)
- CLI: Minimal CLI (optional future) deferred; notebook primary interface (acceptable given educational scope)
- Library docs: Quickstart + inline docstrings; llms.txt can be added later

**Testing (NON-NEGOTIABLE)**:

- TDD: Plan to write export schema tests before export implementation
- Commit ordering: Will ensure failing tests precede code
- Order: Contracts (json schema) → integration (training loop small run) → unit (reward calc, metrics) → notebook smoke (optional)
- Real dependencies: Yes (actual gym env)
- Integration tests: training loop for 5 episodes verifying output files
- Forbidden behaviors acknowledged

**Observability**:

- Structured logging: JSON lines optional (stdout) + in-memory buffer for UI
- Frontend logs: Notebook prints curated metrics; no separate backend
- Error context: Exceptions caught and summarized into session status

**Versioning**:

- Initial version: 0.1.0
- Build increments: To be applied on meaningful changes
- Breaking changes: Not expected in MVP; future config migrations may add version key

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

```text
Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (single project)

## Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved (COMPLETED)

## Phase 1: Design & Contracts

Prerequisite: research.md complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/bash/update-agent-context.sh copilot` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, quickstart.md (schema tests & agent file pending implementation phase; failing tests not created in /plan scope)

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/bash/update-agent-context.sh copilot` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, quickstart.md (schema tests & agent file pending implementation phase; failing tests not created in /plan scope)

## Phase 2: Task Planning Approach

This section describes what the /tasks command will do - DO NOT execute during /plan.

**Task Generation Strategy**:

- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:

- TDD order: Tests before implementation
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

**Task Generation Strategy**:
   - (duplicate content removed above)

**Ordering Strategy**:
   - (duplicate content removed above)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

These phases are beyond the scope of the /plan command.

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

Fill ONLY if Constitution Check has violations that must be justified.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | N/A | N/A |


## Progress Tracking

This checklist is updated during execution flow.

**Phase Status**:

- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - approach documented)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented (N/A currently)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
