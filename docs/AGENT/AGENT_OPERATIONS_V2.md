# Blender Animation Workbench — Agent Operations V2

> Updated: 2026-09-27 KST
> Authority: Main / Sol
> Runtime: Harness Lite V2 + Goose G1-G8
> Purpose: single current authority for planning, research, implementation, debugging, review, verification, and worker allocation.

## 1. Core operating model

AWB plans **the work and the workforce together**.

For every substantial task, Main first creates the complete high-level execution plan (`H1..Hn`) and assigns ownership before implementation starts.

Each planned step records, as applicable:

- owner: `MAIN` or `GOOSE`;
- parallel group and dependencies;
- criticality and difficulty;
- mutation ownership / forbidden overlap;
- research and web-research needs;
- expected artifact/evidence;
- verification requirement;
- debug class when the task is a regression.

Do not wait until each step begins to decide whether a worker is useful. Worker allocation is part of design.

The plan may be revised only when new evidence invalidates the original decomposition or dependency assumptions.

## 2. Main and worker responsibility

### Main / Sol

Main owns the work where continuous judgment is the speed advantage:

- product/architecture decisions;
- critical-path implementation;
- high-complexity or ambiguous implementation;
- cross-subsystem integration;
- ambiguous root-cause analysis;
- mutation ownership when several systems meet;
- final disposition of external review findings;
- authoritative Blender/runtime validation;
- user-facing acceptance decisions;
- final integration and Git checkpoint.

Main does not hand off an important step merely to keep workers occupied.

### Goose G1-G8

Goose is the normal Harness-managed worker pool.

Use Goose aggressively for independent work that can leave the Main critical path:

- bounded implementation steps with clear contracts;
- lower-complexity refactors;
- source/callsite/control-flow investigation;
- research and public web research;
- regression inventory and contract checking;
- test/verifier construction;
- independent failure-path analysis;
- evidence preparation and documentation updates.

A Goose lane is ephemeral. G1-G8 do not have permanent social roles. The Task/Step contract defines the role for that run.

One coupled production seam has one mutation owner. Several workers may inspect the same seam, but simultaneous competing mutation is opt-in, not default.

### Non-Goose runtimes

Standalone Luna/Codex and OpenCode are **not automatic routing or fallback backends**.

They remain exceptional/manual capabilities only when Main has a concrete reason to use them. Goose failure returns evidence to Main; it does not silently fall through to another worker runtime.

External Web Bridge is not an implementation worker. It is an independent PRE/POST challenger surface.

## 3. Harness V2 execution shape

Simple deterministic work may stay on the Native/Main fast path.

Substantial work uses the Harness V2 concepts as needed:

`Goal -> H-plan + workforce plan -> Task Capsule/Steps -> Main + Goose parallel execution -> Artifact/Evidence -> deterministic integration -> verification -> bounded repair -> acceptance`

Important rules:

- decompose by real dependency and ownership, not by desired lane count;
- completed sibling work survives a failed step;
- retry only failed/incomplete units;
- workers return artifacts/evidence rather than whole-conversation dependency;
- workers never recursively create an uncontrolled agent tree;
- Main continues productive critical-path work while Goose lanes run.

### AWB active-project invariant for worker orchestration

While Main is executing an AWB task, the selected/active Harness project remains `blender_animation_workbench` for the whole AWB session, including Goose/Hive worker launch, status polling, result/history collection, review, and approval handling.

Do **not** switch the selected project to `aiprojectharness_lite` merely to use Harness Lite as the orchestration runtime. Harness Lite is the orchestration engine; AWB is the task project and must remain the active project.

When a worker launch needs Harness Lite code/runtime, invoke it explicitly while AWB stays selected:

- pass `project_id="blender_animation_workbench"` as the worker target;
- use explicit/absolute Harness Lite runner, interpreter, or runtime paths when a local launcher is required;
- bind Goose G1-G8 workspaces, worker status/history, and approvals to the AWB target project;
- keep source mutation, verification, Git state, and Main integration rooted in the AWB workspace.

Switch the selected project to `aiprojectharness_lite` only when the task is actually to inspect, modify, test, or repair Harness Lite itself. That is a separate project task, not normal AWB worker orchestration.

Parallelism is **aggressive where independent, conservative where ownership overlaps**.

## 4. Planning and research policy

Research is part of the initial execution plan, not a serial prelude performed by Main alone.

For a substantial new design or implementation slice, Main should normally fan out independent research while building the architecture:

- current source / callsite / ownership map;
- Blender 5.2.1 native/API behavior;
- existing AWB contracts and regression evidence;
- public/official implementation references when useful;
- risk / failure-mode analysis.

Goose research lanes may use web search when the current Goose runtime proves the required web-search capability is available. Enable only the capabilities needed for the assigned research step.

Harness owns G1-G8 orchestration. Do not enable nested Goose delegation/subagent trees as an alternative orchestration layer.

Main synthesizes research; no worker research result is product authority by itself.


## 5. Skill injection and worker learning

Worker skills are part of execution planning, not an optional afterthought.

When Main builds the H-plan, every Goose step must declare the smallest relevant `required_skills` set before dispatch.

Before a Goose launch:

1. inspect the current AWB skill catalog;
2. select only skills materially relevant to that Step;
3. inject those skills into the worker request/runtime;
4. if no skill applies, record an explicit no-relevant-skill reason rather than silently sending an empty set.

Do not attach every skill mechanically. Skill injection exists to give the worker durable project-specific rules without flooding context.

After every meaningful Goose result, Main performs the learning loop:

`CANDIDATE -> EVIDENCE -> Main REVIEW -> APPLY / DEFER / REJECT`

- **CANDIDATE** — a potentially reusable lesson from the worker result;
- **EVIDENCE** — source/runtime/test evidence proving the lesson is broader than one accidental case;
- **APPLY** — update the smallest relevant skill;
- **DEFER** — plausible but not yet durable enough;
- **REJECT** — incorrect, too local, superseded, or harmful.

An APPLY decision is incomplete until the updated skill is injected into the **next relevant worker Step**. This closes the learning loop and proves the new guidance is actually being used.

Worker cleanup/finalization happens only after result disposition and learning review are complete.

External Web Bridge challengers do not use Goose skill injection. Their context/evidence packaging is governed separately by the PRE1/POST1 review rules and provider profiles.

## 6. External Web Bridge — PRE1 / POST1

Every non-trivial implementation or stabilization slice gets exactly:

1. **PRE 1** — one External Web Bridge challenger before implementation;
2. **POST 1** — one External Web Bridge challenger after the first complete implementation/fix round.

Do not use the former PRE2/POST2 pattern.

Main chooses the provider from DeepSeek / Qwen / Gemini Web (GLM fallback) according to task characteristics, current availability, and observed provider strengths. Do not permanently bind one provider to PRE or POST.

PRE receives the decision-critical design/contract plus exact relevant source evidence.

POST receives the actual completed source/diff plus the frozen contracts and required regression evidence.

If POST finds blockers, Main fixes and verifies them directly. Do **not** request another POST merely because the first POST produced findings. A new PRE/POST cycle begins only for a genuinely new implementation/design slice.

A sent PRE/POST request must be consumed and dispositioned before that gate closes.

## 7. Implementation allocation rule

When Main creates `H1..Hn`, assign every meaningful step before execution.

Default decision:

- critical architecture / high ambiguity / high coupling / hard implementation -> MAIN;
- bounded low-to-medium complexity implementation with clean ownership -> GOOSE;
- independent research / regression / test / evidence steps -> parallel GOOSE;
- final integration / authority decisions -> MAIN.

Example:

```text
H1 architecture + ownership contract      MAIN
H2A callsite cleanup                      G1
H2B verifier additions                    G2
H2C independent regression inventory      G3
H3 critical solver implementation         MAIN
H4A failure-path tests                    G4
H4B docs/evidence update                  G5
H5 integration + authoritative runtime    MAIN
```

Main and workers run concurrently whenever dependencies permit.

Do not shrink worker use merely because a task is small. A small step should be parallelized when its startup/context/integration cost is lower than the critical-path time it removes.

## 8. Debug classification

Top-level product-modification authority:

`docs/DEBUG/BLENDER_USER_FIRST_FINAL_TEST_PROTOCOL.md`

Bug-fix classification router:

`docs/DEBUG/BUG_FIXING_GUIDE.md`

Main first freezes the USER defect / last accepted USER boundary under the USER FIRST protocol, then classifies the regression as **SMALL / MEDIUM / LARGE** using reproduction certainty, ownership/coupling, and protected-regression exposure.

- **SMALL** — Main-first fast repair; trace/replay/probe only as needed.
- **MEDIUM** — default serious debugging class; Main owns causal synthesis and patch while 1-3 Goose lanes parallelize independent analysis/verification.
- **LARGE** — preplanned debug H-plan/workforce, recovery/evidence discipline, and broader General Debugger use.

The bug-fixing guide owns escalation rules, repair limits, PRE1/POST1 expectations, and verification depth. Do not duplicate those detailed rules here.

## 9. General Debugger usage

The General Debugger remains evidence-driven rather than mandatory. Use `docs/DEBUG/BUG_FIXING_GUIDE.md` to decide how deeply to escalate, then use `docs/DEBUG/AWB_DEBUG_FLIGHT_RECORDER.md` for the actual trace/replay/incident procedure.

Never stack speculative sign/axis/hotfix patches. When evidence invalidates the causal model or repeated focused repair contaminates the patch stack, return to the newest accepted recovery boundary and re-plan.

## 10. Major-unit operations plans

Permanent policy and milestone execution plans are separate.

For every large work unit, create or update exactly one plan under:

`docs/PHASE4/planning/orchestrator/`

That plan owns the unit's:

- H-step decomposition;
- MAIN/GOOSE assignment;
- Step-level `required_skills` selection;
- parallel groups and dependencies;
- research/web-research fan-out;
- mutation ownership;
- PRE1/POST1 placement;
- verification and USER acceptance boundaries.

Do not grow this permanent policy with RC-, feature-, or milestone-specific worker allocations.

Current plan index:

`docs/PHASE4/planning/orchestrator/README.md`

## 11. Verification and repair

Worker self-report is not acceptance evidence.

Authority order:

`actual Blender/runtime evidence > frozen AWB contract > current Main source/diff > official docs > worker reasoning/self-tests`

Use the Harness V2 failure taxonomy and bounded repair model.

A reproducible verification failure may receive one focused repair attempt on the failing state. Re-run the same verifier. If still failing or ambiguous, return evidence to Main instead of entering an unlimited repair/reviewer loop.

USER visual/perceptual acceptance remains user-owned when no deterministic oracle can replace it.

## 12. Worker result lifecycle

Every meaningful worker result is dispositioned by Main:

- APPLY — integrate/use;
- DEFER — useful but not for the current slice;
- REJECT — wrong, superseded, or not worth Main repair.

Durable worker learning follows:

`CANDIDATE -> EVIDENCE -> Main REVIEW -> APPLY/DEFER/REJECT`

Only evidence-backed reusable lessons enter a skill/profile.

Clean or finalize lanes after their result has been consumed/dispositioned and any learning decision is complete.

## 13. Document authority

This file is the current AWB agent/worker operations authority.

Supporting documents:

- `AGENTS.md` — short project bootstrap and routing;
- `MAIN_WORKFLOW.md` — Main-specific execution checklist;
- `WORKER_GUIDE.md` — bounded Goose worker contract;
- `BLENDER_RUNTIME_RULES.md` — Blender/runtime evidence rules;
- `../DEBUG/BLENDER_USER_FIRST_FINAL_TEST_PROTOCOL.md` — top-level product-modification and USER FIRST/FINAL authority;
- `../DEBUG/BUG_FIXING_GUIDE.md` — subordinate SMALL/MEDIUM/LARGE bug-fix routing;
- `../DEBUG/AWB_DEBUG_FLIGHT_RECORDER.md` — General Debugger operational detail;
- `WORKER_PROFILES/*` — current Goose and External Web provider profiles.
- `../PHASE4/planning/orchestrator/*` — current Phase 4 major-unit execution/workforce plans.

Older OpenCode/Luna-first routing, fixed eight-lane occupancy, two-reviewer PRE/POST minimums, and PRE2/POST2-style review practice are superseded by this document.
