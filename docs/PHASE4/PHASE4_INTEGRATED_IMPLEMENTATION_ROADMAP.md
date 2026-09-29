# AWB Phase 4 — Integrated Implementation Roadmap

> Updated: **2026-09-29 KST**
> Status: **ACTIVE — Post-RC3 Character Production Expansion**
> Role: **single canonical Phase 4 execution/dependency roadmap**
> Runtime target: Blender 5.2.1 LTS
>
> Current pointer: `PHASE4_CURRENT_SESSION.md`
> Product behavior: `planning/rigped/PHASE4_RIGPED_PRODUCT_CONTRACTS.md`
> Verification: `verification/PHASE4_VERIFICATION_STRATEGY.md`

## 1. Large-axis model

```text
AXIS 1  Rigped Foundation / Fit                      CLOSED
AXIS 2  Rigped Animate Core                          CLOSED
AXIS 3  Rigped Core Closure RC0-RC3                  CLOSED / CORE FREEZE PASS
AXIS 4  Post-RC3 Character Production Expansion      NEXT
AXIS 5  Phase 4 Integration / Release                LATER
AXIS 6  Deferred / Optional Product Branches         OPTIONAL
```

Historical A/AK/B/E/F/I/X/RC implementation ledgers are recoverable from Git history. They do not own current sequencing.

## 2. Frozen dependency — AXIS 1-3

### AXIS 1 — Rigped Foundation / Fit

Accepted foundation:

- generated Rigped core and semantic publication;
- default modular generator seam;
- global transform gizmo;
- View3D selection system;
- Object-hosted semantic Fit Mode;
- Fit F1-F6 USER PASS;
- atomic Fit Apply / rollback / native Undo boundary;
- legacy live EditBone Fit authority retired.

### AXIS 2 — Rigped Animate Core

Accepted animator-facing core:

- A1-A4 FK/keying foundation;
- A5 Free/Sliding + E1-E12 stabilization;
- A6 Planted;
- explicit `C` semantic authoring;
- AUTO transform-release authoring;
- semantic W/E/R;
- Free / Sliding / Planted Track Bar semantics;
- supported multi-selection/domain partitioning;
- atomic failure/cancel rollback and native Undo/Redo;
- Blender Action/FCurve replay authority.

There is no active A7 slice.

### AXIS 3 — Core Closure

RC0, RC1, RC2, and RC3 are complete. The Rigped Core is a stable dependency.

Durable closure authority:

- current user-visible contract: `planning/rigped/PHASE4_RIGPED_PRODUCT_CONTRACTS.md`;
- frozen USER evidence: `debug/user_final_tests/`;
- final RC3 receipt: `debug/user_final_tests/RC3_CORE_FREEZE.md`;
- current regression tests/verifiers and Git history.

Final RC3 gate: **1194 passed / 63 skipped / 106 warnings + Ruff PASS**, with compact USER smoke and fresh-process save/reopen accepted.

Frozen semantics reopen only for a concrete regression or explicit product-contract change.

The bounded four-control lower-link Rotate follow-up is closed: fresh 2026-09-29 USER PASS covers simultaneous Free/Sliding LOCAL X/Y/Z under the current LOCAL-only contract, with Calf LOCAL Y locked/no-op. No GLOBAL follow-up remains.

## 3. AXIS 4 — Character Production Expansion

Every new AXIS 4 mutation starts from a fresh focused contract/plan. Do not revive old stage-local `CURRENT/NEXT` text.

### 4A. Rigped presentation / solid-shell modeling

Goal: improve readable Biped-style presentation without changing animation authority.

Rules:

- consume frozen semantic roles;
- presentation should remain derived/disposable where practical;
- no mesh/display layer becomes solver or animation authority;
- do not redefine Fit/Animate semantics.

### 4B. Modular topology expansion

Fresh planning is required for:

- variable spine count;
- variable neck count;
- finger semantic/generator support;
- generalized toe/digit topology;
- explicit atomic structure-rebuild transaction;
- expanded presets.

The retained modular-generator plan is reference material, not an active implementation order.

### 4C. Production pipeline

Default release-path order:

```text
I21  World Offset GUI / confirmed animation mutation
-> I22  Mesh Prep
-> I23  Bind / Auto Weights / native Weight Mode
```

World Offset preview is non-authoritative until confirm. Mesh/weight work must preserve unrelated user data and fail safely.

## 4. AXIS 5 — Phase 4 Integration / Release

### I24 — Integrated Phase 4 acceptance

No new feature development belongs in I24.

Verify the production flow:

```text
Mesh Prep
-> Create Rigped
-> Fit
-> Bind / Weight
-> Animate
-> C / AUTO / All Key
-> Free / Sliding / Planted
-> semantic W/E/R
-> World Offset
-> save/reload / Undo / replay
```

Evidence spans static, Blender runtime/state, GUI input/screen, Undo/Redo, save/reload, failure safety, performance, and preserved earlier-phase behavior.

## 5. AXIS 6 — Deferred / Optional Product Branches

Valid but non-blocking ideas include:

- advanced Rigped display/visibility polish;
- Object-space/Replant/arbitrary-contact-point UX;
- continuous-contact precision expansion;
- COM Mirror and other animation operations;
- Blocking Mode;
- STEP / Hold / FPS / playback UX;
- Match & Bake;
- general Align/Pivot utilities;
- Secondary Motion;
- export / Root Motion / bake;
- advanced picker artwork;
- arbitrary full-body balance solver;
- generalized long-chain solver;
- external-space rewrite/detach/bake;
- NLA/tweak/multi-layer inverse editing.

Export may be pulled forward only when a concrete release target requires it.

## 6. Current order

```text
COMPLETED
Foundation / Fit
-> Animate Core
-> RC0-RC3 Core Closure

CURRENT
-> repository cleanup
-> bounded current-axis revalidation if still required

NEXT
-> one focused AXIS 4 plan
-> explicitly prioritized presentation / topology / production-pipeline slice

THEN
-> I24 Integrated Phase 4 acceptance

OPTIONAL
-> deferred utilities / export branches

AFTER PHASE 4
-> Phase 5 Pose / Motion / Layers
```

## 7. Documentation ownership

- `PHASE4_CURRENT_SESSION.md` — current gate and immediate action.
- `PHASE4_INTEGRATED_IMPLEMENTATION_ROADMAP.md` — this file; large-axis order/dependencies.
- `planning/rigped/PHASE4_RIGPED_PRODUCT_CONTRACTS.md` — user-visible Rigped behavior.
- completed Rigped core plans (`PHASE4_RIGPED_RC1_CORE_HARDENING.md`, `PHASE4_RIGPED_RC2_CORE_REFACTOR.md`, `PHASE4_RIGPED_RC3_CORE_FREEZE.md`) live under `planning/rigped/` with the subsystem they explain.
- retained completed subsystem plans — historical design references for the current system; speculative future plans are written fresh only when prioritized.
- `verification/PHASE4_USER_TEST_SCOPE.md` — USER/practical acceptance policy.
- `verification/PHASE4_VERIFICATION_STRATEGY.md` — common evidence/rollback/performance rules.

Do not create another competing Phase 4 roadmap. Keep completed plans beside the subsystem they explain; do not pre-write or retain stale implementation plans for unstarted future features.
