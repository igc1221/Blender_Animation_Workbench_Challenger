# Phase 4 Rigped — Stabilization + Sliding Integrated Execution Plan

> Updated: **2026-09-22 KST**
> Status: **A5 STABILIZATION CLOSED / E1-E12 USER PASS / CLOSED / A6 PLANTED NEXT**
> Runtime target: **Blender 5.2.1 LTS**
> Main authority: **Sol Main**
> Mandatory challenger policy: **every implementation/stabilization item receives a narrow pre-implementation challenger review and a narrow post-implementation challenger review; challenger output is advisory evidence and Main decides whether a finding blocks, improves, defers, or is rejected**
> Architecture escalation: **Astra only when Main + challenger evidence remains materially ambiguous**
> Product authority: `PHASE4_PRODUCT_CONTRACTS.md`
> Source plans merged here:
> - `PHASE4_RIGPED_SLIDING_IMPLEMENTATION_PLAN.md` (2026-09-18 Astra Sliding/COM plan)
> - `PHASE4_RIGPED_STABILIZATION_OWNERSHIP_PLAN.md` (2026-09-19 ownership gate)
> - `docs/AGENT/ASTRA_REVIEW_RESPONSE_PHASE4_STABILIZATION_20260919.md` (2026-09-19 Astra architecture response)

## 2026-09-22 execution override — E1-E6 COMPREHENSIVE-REVIEW STABILIZATION CLOSED / E7 UNBLOCKED

E1 through E6 remain **USER PASS / CLOSED**. The later Astra comprehensive review produced three accepted stabilization blockers before E7:

- **B1** — Sliding Rotate active public input was disconnected from hidden result derivation while FK feedback was muted.
- **B2** — final Contact feedback-authority mutation/update occurred after `journal.commit()`.
- **B3** — E6 hidden SolverSeed raw result mutation was not fully owned by gesture cancel/failure restore.

Main resolved B1-B3 as one coherent stabilization batch without adding a second solver or new animation authority. GLM N1 replay hardening is closed. GLM N2/N3 remain assigned to the first E7 implementation batch.

Closure evidence on current source:

- `awb-check`: **389 pytest PASS + Ruff PASS**.
- Blender 5.2.1 `awb-phase4-i12-runtime-verify`: **FULL PASS**, including L/R Arm+Leg FK↔IK, opposite-bend Contact playback, Sliding Rotate→C pose preservation, live red-pivot sync, exact rollback, limits, and singular fallback.
- Blender 5.2.1 `awb-phase4-i13-runtime-verify`: **FULL PASS**, including same-frame Free↔Sliding continuity, fault rollback, single-journal Contact closure, and Rigped K complete no-op.
- Clean-baseline frozen `E5` replay: **PASS**.
- Newly launched clean-baseline frozen `E6` replay: **PASS**.
- Figure/Fit current pointer was not repointed for this stabilization.
- POST round consumed exactly once with two independent reviewers:
  - **Qwen = CLEAR / no blockers**.
  - **DeepSeek = REVIEW / no source-backed product blocker**; its only material point was insufficient inline source evidence in the packet.
- Main accepted DeepSeek's packet-construction lesson into the `review-evidence` skill and independently verified the exact current-source callsites, including no fallible product-state mutation or required verification after `journal.commit()`.
- Mandatory post-review rerun: **389 pytest + Ruff PASS, I12 FULL PASS, I13 FULL PASS**.

The stabilization gate is therefore **CLOSED**. E7 later passed USER FIRST and frozen replay; E8 is now the active next item.

## 2026-09-22 E8 entry pre-audit

Before handoff to the next session, Main performed a narrow source audit only; **no E8 PRE request has been sent yet**.

Current-source facts:
- `rigped_ik_pivot_overlay.py` explicitly documents and implements the red point from the solved generated terminal pair `MCH_Hand/MCH_Foot`, so the current cue follows solved result rather than the true hidden IK target/reference.
- Generated `IK_Hand.*`, `IK_Foot.*`, `IK_Elbow.*`, and `IK_Knee.*` authored IK target/pole bones use `ROOT` as parent.
- E8 Root behavior must therefore preserve the current hierarchy: Root motion carries hidden target/pole exactly once. E8 must not add world locking, compensation, or a second delta.
- E8 red-cue work should change only the displayed source of the Sliding cue to the actual hidden IK target/reference, so reachable cases overlap and unreachable cases visibly distinguish target from solved terminal.
- Continue by finishing the E8 source audit, then send exactly one PRE round using two rotated first-tier providers. Do not implement E8 before all sent PRE responses are consumed.

## 2026-09-22 E7 — USER PASS / CLOSED

E7 first-batch scope is now implemented for AUTO OFF body/ancestor dependency:
- Direct Move consumes one frozen E1 `OperationDomainSnapshot`; no mid-gesture semantic ownership rescan.
- COM Move freezes one character-wide Sliding affected set for the gesture and treats those limbs as passive pinned-target dependencies.
- Body Rotate freezes the same character-wide Sliding set, preserves the existing active/passive partition, and enables E7 hard guards only when there are no active Sliding edit sessions and all selected roles are COM/Pelvis/Spine/Head/Clavicle.L/Clavicle.R.
- Root is explicitly outside the E7 hard-guard scope and remains E8.
- Hidden IK target/pole/pole-angle/runtime identity/native IK raw state are hard immutable for E7 body dependency; drift cancels/fails closed.
- Native result-target reach residual and public-result residual are diagnostic measurements only; unreachable native saturation is allowed. No body clamp, stretch, world pinning, or target transport was added.
- Cancel/failure restores body state, frozen authority raw state, and hidden seed state; restore disables reseeding.

Closure proof:
- `awb-check`: **396 pytest PASS + Ruff PASS** after the semantic-replay compatibility regression test was added.
- Blender 5.2.1 I12: **FULL PASS**.
- Blender 5.2.1 I13: **FULL PASS**.
- Blender 5.2.1 E7 dedicated runtime probe: **FULL PASS**:
  - COM Move pinned authority + reachable/unreachable saturation;
  - COM/Pelvis/Spine/Head/Clavicle.L/Clavicle.R Rotate dependency;
  - target/pole drift injection fails closed;
  - AUTO OFF / no new keys / exact restore.
- E7 POST round: **Gemini CLEAR / blocker 0; Qwen CLEAR / blocker 0**.
- Post-review rerun: **396 pytest + Ruff PASS, I12 FULL PASS, I13 FULL PASS, E7 runtime FULL PASS**.
- Frozen E5 replay: **PASS**.
- Frozen E6 replay: **PASS** after fixing the Direct Move semantic-replay path to consume the frozen `OperationDomainSnapshot` and E7 dependency guard.
- E7 USER FIRST session `98f3561750404c95910c95e7a0d99941`: **USER PASS**:
  - Foot.L Sliding + COM Move from max-extension/near-straight start;
  - bent Foot.L continuation + COM Move;
  - Hand.L Sliding + Spine/Head Rotate while Foot.L remained Sliding;
  - Pelvis / Clavicle.L Rotate;
  - ESC cancel at trace seq 1871 and 1959.
- User-session trace: **1957 events / 58 committed semantic actions / 0 authority drift / 0 transform fail / 0 operator error**.
- Frozen `debug/user_final_tests/E7/user_final_replay.json`: clean-baseline `awb-replay-user-final` **PASS / 58 actions**.
- Current user-final pointer: `debug/user_final_tests/current.json -> E7`.

E7 is therefore **USER PASS / CLOSED**. E8 is now the next implementation item.

This override does not alter the normal narrow PRE/POST challenger policy for later implementation items.

## 0. Purpose

This document is the execution authority for the next Rigped stabilization sequence.

It merges two valid but different layers of work:

1. the 2026-09-19 ownership review, which fixes **who owns one operation, one domain, one transaction, and one replay meaning**;
2. the 2026-09-18 Sliding implementation plan, which fixes **how Sliding/COM/Root evaluation actually behaves in Blender**, including max-extension / near-straight limb seeding, native matrix conversion, live body dependency, and multi-limb behavior.

The newer ownership review does **not** replace the older Sliding/COM runtime findings. The older line numbers/source hashes are historical evidence only, but its P1 behavior findings and verification matrix remain active until re-proven on current source.

No item below is considered closed from source inspection alone. This stabilization uses a **user-first debug workflow**, not Main-run acceptance testing.

```text
PRE-IMPLEMENTATION CHALLENGER
-> Main judges the challenger findings
-> Main implementation
-> POST-IMPLEMENTATION CHALLENGER on the exact diff/source
-> Main judges the challenger findings
-> install / fresh Blender debug-mode handoff to user
-> USER FIRST DEBUG TEST
-> Main reads Flight Recorder / runtime logs
-> Main diagnoses and patches
-> replay verification loop on the user-captured regression
-> if replay exposes another defect: patch -> replay again
-> when the captured regression loop is stable
-> USER FINAL CONFIRMATION
-> item closes
```

**Main does not run its own pre-user Blender acceptance/repro/GUI/fault-injection test pass for these items.** The first behavioral test after implementation belongs to the user in debug mode. Main may inspect source, prepare instrumentation, install the build, and launch the fresh baseline, but those actions are not acceptance evidence.

After the user's first debug run has produced logs/capture, Main may use the resulting Flight Recorder/replay artifact repeatedly to diagnose and verify fixes without asking the user to manually recreate the same gesture. Replay verification is a debugging/verification loop derived from the user's real run, not a substitute for the user's final confirmation.

Each item’s user-confirmed final replay is frozen under `debug/user_final_tests/<ITEM>/user_final_replay.json`. `debug/awb_replay_latest.json` remains a rolling capture and must not replace an existing frozen final replay unless the user-test behavior itself changes. After a code fix, Main reuses the frozen item replay on a clean baseline before asking the user to repeat any already-captured gesture.

A challenger may reject a design or identify missing proof, but **user-observed Blender behavior + Flight Recorder/replay evidence + frozen product contract + current source remain the acceptance authority**.

## 1. Frozen semantics for the entire sequence

Do not change these while executing this plan.

- Rigped `K / KEY` = complete no-op.
- Rigped `C` = explicit Rigped semantic authoring command.
- Broad mixed `C` containing Contact limbs plus supported direct Rigped controls must author both in one coherent operation.
- Direct controls authored by broad mixed `C` remain Free.
- Direct-only `C` is **not frozen**; keep it fail-closed/no-op during this stabilization unless separately approved.
- Free = FK authority.
- Sliding = hidden IK/result authority.
- Planted remains deferred.
- Hidden IK targets/poles remain hidden and unselectable.
- Visible ForeArm/Calf never receive Blender native IK solver constraints directly.
- AUTO triggers existing semantic authoring on successful transform release; AUTO does not call C or cycle Contact state.
- Action/FCurve remains animation/replay authority.
- One successful gesture = one Undo.
- ESC/RMB/failure restores exact operation start and writes zero keys.
- Same-limb multi-selection deduplicates to one semantic limb domain.
- A user operation must not leave a partial authored result.
- Accepted gizmo drawing/hit/input behavior remains closed unless a concrete regression appears.

### Sliding body-space semantics frozen for this plan

- COM / Pelvis / Spine / Head / Clavicle body motion may change limb roots while an existing Sliding IK reference remains independent of that body motion.
- For a **reachable** Sliding limb, native solved Hand/Foot should remain on its target/reference and the public limb must show that solved result live.
- When the body moves beyond reachable range, **target/reference authority remains fixed but the native solved result may diverge due to reach limit**. Do not silently world-pin, stretch, full-body compensate, or invent Planted behavior.
- Root translation preserves the current hierarchy: Sliding target/pole follows Root exactly once. Root Move is not a world-lock command.
- The red Sliding point is the animator-facing **actual IK target/reference position**, not merely the solved MCH terminal result.

## 2. Source-state boundary at plan creation

Current reviewed source snapshot:

```text
phase4_contact_authoring.py = e5e422e00439e47f72adb02d812fd9f3f8bbda7afec3b8ee4f5f176de7777994
rigped_transform.py         = 472fb57734e3d465b4ead1d102ed0e07eb87b41dbf673cff04ba08b474d02683
debug_replay.py             = f270108923df5a25dc8ad2f128601ffba7226130a718cf6fd595e0ea631c66b2
phase4_preflight.py         = 484f1f4603095aa6ed665cb1115c6c6869b7f4316a57c19f6d7daa1c35cbe09f
phase4_writer.py            = 38a85897d47f3e44e5e9fd206461fca6f1c09805bb8c310fa6057900068107ba
```

Current recoverable Git checkpoint at review time:

```text
b6427cb8b609ef1a9f53e5a8127097ede46f094e
```

The 2026-09-18 Sliding plan audited older source. Treat its source locations as historical guidance, not literal current implementation positions.

## 3. Challenger review invariant

Every implementation/stabilization item E0-E12 receives **two narrow CHALLENGER REVIEWS** when code is involved:

1. **PRE-IMPLEMENTATION CHALLENGER** — reviews the intended boundary, frozen contract, current source, and proposed change before Main edits production code.
2. **POST-IMPLEMENTATION CHALLENGER** — reviews the exact resulting diff/source before the build is handed to the user for behavioral testing.

For an evidence-only item with no product-code mutation, one pre-decision challenger is sufficient unless the evidence changes the architecture boundary.

Default challenger pool:
- DeepSeek / Qwen / Gemini Web Bridge are co-equal first-tier reviewers; rotate pairs across PRE/POST rounds for diversity.
- GLM is fallback when a fourth perspective or provider replacement is needed.
- Each request is narrow and decision-boundary specific.
- PRE packet contains current frozen contract, current relevant source, known evidence, and the proposed implementation boundary.
- POST packet contains the same frozen contract plus the exact Main diff/source and asks whether the implementation actually preserves the intended boundary.
- Do not send a broad “review everything” request.
- Main consumes and evaluates each response; challenger recommendations remain advisory.

Main classifies every meaningful challenger finding by evidence quality and scope:

```text
ACCEPTED BLOCKER
- concrete frozen-contract conflict, current-source defect, or runtime risk capable of invalidating the item
- Main stops closure of that item, proves or repairs the issue, and re-challenges only if the material decision boundary changed

ACCEPTED IMPROVEMENT
- valuable correction or stronger design/proof that improves the current item but is not itself a correctness blocker
- Main integrates it when proportionate, or records the exact follow-up while continuing if current acceptance remains valid

VALID BUT DEFERRED
- evidence-backed point outside the current item's required scope
- Main records it in the appropriate later item/debt location and continues

REJECTED
- preference, unsupported speculation, stale-source reasoning, contract mismatch, or a larger redesign without evidence that the current item is unsafe
- Main records the rejection reason and continues

ESCALATE
- Main and challenger have materially conflicting source/runtime/contract evidence on an architectural decision
- escalate only that narrow decision to Astra
```

A challenger `BLOCK` recommendation does **not** automatically block progress; Main must accept the underlying evidence as material. A challenger `PASS` recommendation also does **not** make an item pass. Item closure is owned by Main and requires the listed code/runtime evidence.

Do not re-challenge every mechanical micro-fix. Re-challenge when the accepted finding causes a material change to the boundary the challenger reviewed.

If the bridge is unavailable, record `CHALLENGER UNAVAILABLE`; continue only from authoritative Main + Blender evidence and do not describe the item as challenger-reviewed.

---

## 3.1 User-first debug / replay validation protocol

This protocol overrides any older wording in the source plans that says Main should run an internal Blender acceptance matrix before user testing.

### Before implementation

- Main inspects current source/log evidence and defines one bounded item.
- Send PRE-IMPLEMENTATION CHALLENGER.
- Main classifies findings as ACCEPTED BLOCKER / ACCEPTED IMPROVEMENT / VALID BUT DEFERRED / REJECTED / ESCALATE.
- Only accepted material findings alter the implementation boundary.

### After implementation, before user test

- Main does **not** execute a self-acceptance Blender scenario.
- Send POST-IMPLEMENTATION CHALLENGER with the exact diff/current source.
- Main resolves valuable findings.
- Install current development build.
- Close the previous AWB Blender process when needed.
- Launch a fresh existing manual baseline with debug/Flight Recorder instrumentation ready.
- Hand control directly to the user.

### User first debug test

- User performs the requested gesture/scenario manually.
- The user's observed behavior is the first behavioral acceptance signal.
- Main does not ask the user to recreate already captured behavior merely to obtain more evidence if Flight Recorder/replay can reproduce it.

### Log -> patch -> replay loop

When the user reports a defect:

```text
user test
-> inspect Flight Recorder/runtime log
-> identify ownership/solver/writer/replay cause
-> patch
-> install/update as required
-> replay the captured sequence
-> inspect replay/state/key evidence
-> if still wrong: patch -> replay again
```

During this loop Main may run replay repeatedly because it is reproducing the user's captured real gesture. Do not replace this with unrelated broad automated suites.

If the patch materially changes the architecture boundary reviewed by the previous POST challenger, send another narrow POST challenger. Mechanical fixes inside the already-reviewed boundary do not require challenger spam.

### Repeated-failure escalation rule

For the **same bug family / same root-cause boundary**:

- if Main has made **three repair attempts** and the defect is still not closed, stop local patching;
- if the defect was considered fixed and then **reappears from the same underlying ownership/solver/writer/replay cause**, treat that as an escalation trigger even if fewer than three new edits were made;
- do not continue tolerance tuning, extra guards, or adjacent special cases merely to keep the current design alive.

Escalation path:

```text
same defect family reaches 3 failed repair attempts
OR same root-cause defect reappears after a claimed fix
-> freeze further local patching
-> collect current diff + user capture + replay + logs + exact failed invariant
-> re-check frozen product contract and operation ownership from first principles
-> use Astra for a narrow architecture/root-cause review when that can add value
-> Main decides KEEP / REVERT / REDESIGN / NEW PROOF
-> only then resume implementation
```

The count is by **root-cause family**, not by every tiny edit or typo. A mechanical mistake in applying an otherwise valid fix does not consume a separate architecture strike. Conversely, three superficially different symptoms that come from the same ownership defect count toward the same escalation threshold.

### Final confirmation

When replay and logs show the captured defect is resolved:

- leave a fresh/current Blender baseline ready;
- tell the user exactly what to check;
- user performs final confirmation;
- only the user's final confirmation closes the user-visible item.

### Forbidden shortcut for this stabilization

Do not mark an item complete from any of the following alone:

- Main self-run Blender scenario;
- worker/challenger PASS;
- unit/static tests;
- replay without an originating user capture when the behavior is user-facing;
- source inspection.

---

# E0 — Baseline, existing-evidence inventory, and debug instrumentation readiness

## Goal

Freeze the current baseline, inventory already-captured evidence, and make sure debug/Flight Recorder instrumentation can capture the exact scenarios needed after implementation. Do not run a new Main behavioral repro pass here.

## Main work

- Confirm existing manual baseline remains `build/rigped_animate_manual_baseline.blend`.
- Preserve existing Flight Recorder captures.
- Add or refresh focused probes only where current evidence is missing.
- Record current direct Move/Rotate routes and current Sliding active/passive membership.
- Record target, pole, MCH result, public matrices, Contact state, Action/Slot/ChannelBag, and current keys for the focused cases.
- Do not modify product behavior.

## Required user debug scenario inventory

1. broad mixed C: Root/COM/Pelvis/Spine/Head/Clavicles + four limbs;
2. same-frame repeated broad C;
3. one Contact limb + direct controls;
4. direct-only C;
5. bent Sliding foot + COM W;
6. **max-extension / near-straight Sliding foot + COM W**;
7. Sliding hand + Spine/Head Rotate;
8. COM/Pelvis Rotate with Sliding limb;
9. selected four-limb Sliding Rotate twice;
10. Root W with all four limbs Sliding;
11. captured historical replay whose C semantics changed.

## CHALLENGER GATE E0

Question:

> Does this repro/measurement matrix distinguish target authority, native solved result, public display, authored write footprint, and replay meaning well enough to prevent a false PASS? Identify only missing proof that can change the next architecture decision.

## Exit

- Repro set and measurements are sufficient.
- No product mutation yet.

---

# E1 — Partial revert of unsafe mixed-C WIP + immutable operation-domain resolver

## Goal

Stop domain ownership from being rediscovered differently by Contact, direct writer, transform, and replay code.

## Main work

1. Revert only the unsafe WIP seam:
   - full-selection direct planning followed by row filtering;
   - deep propagation of direct meaning into Contact single/batch internals;
   - newly activated direct-only C branch.
2. Preserve accepted Contact behavior and accepted Sliding fixes.
3. Add a small read-only operation-domain snapshot/resolver, proposed location:
   - `extension/blender_animation_workbench/rigped_operation_domain.py`
4. The resolver returns stable operation identity for:
   - selected Contact mappings;
   - selected supported direct bindings;
   - unsupported/ambiguous bindings;
   - active binding;
   - current Sliding mappings where needed by transform;
   - rig/setup/frame/animation binding identity.
5. Resolver performs no mutation, no writer work, no solver work, no UI work.

## Required invariant

```text
each selected binding
= exactly one limb domain
OR supported direct domain
OR explicit rejection
```

Do not interpret “not Contact” as automatically direct.

## CHALLENGER GATE E1

Questions:

> Is the resolver strictly read-only and identity-oriented, or has semantic writer/solver responsibility leaked into it?

> Can any selected binding still silently disappear between native selection and the final planned operation domain?

## User debug target after implementation

- same-limb selected controls dedupe once;
- independent mappings remain distinct;
- ambiguous selection fails before mutation;
- no keys/pose/Undo change from resolver-only calls.

## Exit

Operation-domain identity is shared and frozen per operation.

---

# E2 — Explicit direct-subset planning

## Goal

Direct planning must inspect and allocate only the direct bindings that the operation actually owns.

## Main work

- Add a narrow `binding_ids`/equivalent subset API to `build_direct_key_plan()`.
- Keep full selection identity separately as freshness/provenance.
- Refuse subset IDs that:
  - are not in the frozen selection;
  - cannot resolve completely;
  - are owned by a Contact limb domain;
  - are unsupported/ambiguous.
- Remove broad-plan-then-filter behavior.
- Do not temporarily mutate Blender selection to trick the planner.

## CHALLENGER GATE E2

Question:

> Does the new subset API separate command provenance from direct write target cleanly, without creating a second selection authority or allowing Contact-owned bindings into direct planning?

## Proof

- broad mixed selection plans only actual direct controls as direct;
- Contact controls do not fail direct preflight merely because they are in the same native selection;
- stale/rebound selection fails before write;
- no mutation on plan failure.

## Exit

Direct planning footprint is explicit before allocation/write.

---

# E3 — Atomic broad-mixed C coordinator

## Goal

One C press authors the complete selected supported domain or nothing.

## Main work

At the C command boundary:

```text
freeze operation-domain snapshot
-> choose ONE Contact group target
-> prepare Contact fragment(s)
-> prepare actual direct Free fragment
-> validate combined storage/dependency closure
-> one existing MutationJournal
   -> direct semantic backing-property preparation
   -> existing Contact write primitives
   -> existing direct write primitives
   -> native evaluation
   -> representation reconciliation
   -> full operation verification
-> one commit
-> one outer Undo
```

Rules:

- coordinator is not a new semantic writer;
- Contact code still owns Contact transition/hidden IK/pole/scalar meaning;
- direct writer still owns direct P/R + Free marker/backing property meaning;
- no second independent transaction;
- no nested commit;
- direct-only C remains fail-closed/no-op;
- repeated same-frame mixed C cycles Contact group but re-upserts current direct pose as Free.

## Backing-property requirement

Reuse the existing `phase4_writer.py::_prepare_semantic_state_property()` behavior under the shared journal.

Failure must restore:
- newly created property -> absent again;
- existing property -> previous value;
- created Action/Slot/Bag/FCurve/key -> removed if operation created it;
- previous key/handle/interpolation/etc. -> exact previous data;
- Contact guidance/latch changes caused by this operation;
- transient public/hidden preparation state.

## CHALLENGER GATE E3

Questions:

> Is there exactly one persistent transaction/rollback authority for Contact + direct broad C?

> Is any failure-capable representation repair still happening after persistent commit, creating a partial-success path?

## Proof

- broad C with direct backing property initially absent;
- same-frame repeat;
- active direct control variant;
- one Contact limb + direct controls;
- direct-only C = zero mutation + zero Undo.

## Exit

Broad mixed C is structurally atomic before fault injection.

---

# E4 — Mixed-C authoring coverage and rollback boundary review

## Goal

Review and instrument rollback boundaries now, while keeping artificial fault-injection testing out of this user-first core stabilization pass. Real failures observed during user debug are captured and verified through replay.

## Main work

- Inspect the rollback coverage for the known high-risk boundaries:
  1. direct backing-property preparation;
  2. partial Contact write;
  3. partial direct write;
  4. final native evaluation/representation verification;
  5. stale Action/Slot/frame/domain identity.
- Add/retain Flight Recorder detail needed to tell whether residue survives if one of these boundaries fails during real use.
- Do not run an artificial fault-injection suite in this stabilization pass.
- If the user's debug run naturally exposes one of these failures, capture it once and use replay to verify exact restoration after the patch.
- Dedicated systematic fault injection belongs to **RC1 Hardening** after core feature completion unless a current user-captured defect makes it necessary earlier.

## CHALLENGER GATE E4

Question:

> Is there any persistent or transient residue that survives one of these failures and can change the next C, scrub, Undo, or transform result?

## User debug target after implementation

User debug target:
- complete Contact bundles;
- complete selected direct Free bundles;
- exact-time duplicate prevention;
- scrub away/back;
- one Undo/Redo for the whole C.

If a real failure is captured:
- replay must show zero partial keys after the fix;
- zero stray property;
- zero stale Contact guidance;
- exact pose/hidden/public restoration;
- no success trace on partial failure.

## Exit

Mixed C ownership gate is closed.

---

# E5 — One Sliding gesture lifecycle: active vs passive ownership

## Goal

Keep the accepted active/passive split while making one operation responsible for coverage, cancel, completion, and verification.

## Main work

Use the E1 operation-domain identity at gesture start.

Required invariant:

```text
active_sliding ∩ passive_sliding = empty
active_sliding ∪ passive_sliding = affected_current_sliding
```

- selected Sliding domains: existing active solver/sync only;
- other current Sliding domains affected by body/ancestor evaluation: passive representation maintenance only;
- passive path must not author keys;
- passive path must not independently move target/pole to “fix” display;
- active domain must not re-enter passive guard through a fresh selection scan;
- ESC/failure restores public, hidden, temporary constraint/hinge, and gesture state.

## CHALLENGER GATE E5

Questions:

> Can the same Sliding mapping still be mutated by two preview owners in one event?

> Is passive maintenance truly derived-display work, or can it alter hidden authority and silently become a second solver?

## Proof

- four Sliding limbs Rotate twice consecutively;
- active/passive IDs logged per gesture;
- no convergence failure from duplicate ownership;
- cancel exactness;
- no key creation with AUTO OFF.

## Exit

One gesture owns one complete Sliding lifecycle.

---

# E6 — Sliding evaluation foundation: straight seed + inheritance-aware native conversion

## Goal

Close the two major 2026-09-18 P1 findings that current passive overlay does not prove fixed.

### P1-A still requiring proof/fix

Straight fully extended Sliding arm/leg can stall native IK when COM/body moves even though target stays fixed.

### P1-B still requiring proof/fix

Current public reconstruction helpers still use manual parent/rest math and may be wrong with Blender inheritance flags such as `use_inherit_rotation=False`.

## Main work

- Introduce/complete a shared Sliding evaluation service, e.g. `rigped_sliding_evaluation.py`, but keep it writer-free.
- Distinguish:
  - AuthoredReference;
  - transient SolverSeed;
  - native SolvedResult;
  - derived PublicDisplay.
- Use deterministic two-bone preferred-bend seeding only to escape the straight singularity; native Blender IK remains the final solver.
- Preserve no-stretch behavior.
- Replace duplicated custom reconstruction where necessary with Blender-native/inheritance-aware conversion behavior proven in 5.2.1.
- Do not redesign rig hierarchy or constraints.
- Do not key the derived public result.

## CHALLENGER GATE E6

Questions:

> Does the straight-limb seed merely initialize Blender native IK, or has it accidentally become a second final solver?

> Does the proposed conversion correctly respect generated Rigped inheritance flags and preserve twist/roll/reference meaning?

## User debug target after implementation

- product max-extension / near-straight + bent L/R arm and leg;
- COM down/up/side;
- COM/Pelvis rotation;
- non-identity Root/Object placement;
- public=result when reachable;
- target/pole unchanged by body-only dependency;
- deterministic same-input repeated evaluation;
- no stretch;
- branch continuity.

## Exit

Max-extension / near-straight and rotation cases are native-solve correct before broad body integration.

---

# E7 — Live COM / Spine / Head dependency behavior

## Goal

Make body/ancestor transforms maintain Sliding authority during every preview event, not only release.

## Main work

For supported direct body gestures:

- COM W;
- supported COM/Pelvis E;
- Spine E;
- Head E;
- Clavicle E where dependency exists;

resolve affected Sliding domains from the frozen gesture snapshot.

For each preview event:

```text
apply body transform
-> evaluate hierarchy/native IK
-> if straight singular seed is needed, apply bounded transient seed
-> re-evaluate native result
-> project derived public display
-> verify target/pole authority preservation
```

Do **not** use direct Sliding W target-transport behavior for COM/body dependency.

Do **not** infer authored target intent from public/result mismatch.

### Reach behavior

- target/reference remains fixed under COM/Spine/Head dependency;
- reachable result should stay at target;
- unreachable native result may diverge;
- do not secretly clamp body motion unless a separate product decision explicitly adopts that behavior;
- do not world-pin Root-relative targets.

## CHALLENGER GATE E7

Questions:

> Does body dependency preserve target authority without converting Sliding into Planted/full-body compensation?

> In the unreachable case, does the implementation distinguish target drift from legitimate native reach saturation?

## User debug target after implementation

- one Sliding foot + COM down/up/side/return;
- straight-start and bent-start;
- one Sliding hand + Spine/Head rotate;
- reachable and intentionally unreachable cases;
- AUTO OFF;
- ESC;
- repeated drag;
- target/pole local + world before/after;
- public/result residual;
- result itself before/after overlay projection.

## Exit

“COM moves while Sliding hand/foot contact remains authoritative” is runtime-proven with correct reach semantics.

---

# E8 — Root semantics + red target cue

## Goal

Keep Root placement semantics distinct from COM-independent Sliding contact, and make the red cue show the actual reference.

## Main work

### Root

Root W:
- Root moves the rig;
- Root-parented hidden IK target/pole follows exactly once;
- no extra world pin;
- no double delta;
- existing target local animation remains unchanged unless Root itself is authored.

### Red cue

Change/verify the Sliding red point so it is sourced from the actual hidden IK target/reference, not merely `MCH_Hand/MCH_Foot` solved result.

Reachable:
- target and result overlap visually.

Unreachable:
- red target remains at the true reference;
- solved Hand/Foot may visibly fall short.

## CHALLENGER GATE E8

Questions:

> Does Root motion preserve existing hierarchy rather than silently introducing world-space Sliding?

> Does the red cue expose true target/reference authority in reach-limit cases instead of hiding target/result divergence?

## User debug target after implementation

- Root translation with all limbs Sliding;
- non-identity Root/Object placement;
- target/pole local unchanged, world follows Root once;
- COM W contrast case: target world fixed relative to Root while COM moves;
- unreachable body pose: red cue stays on actual target.

## Exit

Root-relative Sliding and target cue semantics are unambiguous.

---

# E9 — Multi-limb Sliding body dependency

## Goal

Scale the now-proven one-limb body dependency to independent limbs without inventing full-body IK.

## Main work

Order:

1. left Sliding / right Free;
2. both feet Sliding;
3. one foot + one hand Sliding;
4. both hands;
5. all four limbs Sliding.

Rules:

- each limb keeps independent hidden native IK;
- shared body gesture input is one operation;
- selected edit domain and dependency-maintenance domain remain separate;
- no limb writes body compensation;
- one limb failure cancels the whole supported gesture when the operation contract requires all-or-none;
- no hidden full-body balance solve.

## CHALLENGER GATE E9

Questions:

> Does multi-limb dependency remain a union of independent native IK domains, or has shared body handling introduced hidden cross-limb solving/compensation?

> Can failure of one limb leave another limb or public overlay partially advanced?

## User debug target after implementation

- both feet COM down/side;
- foot + hand;
- all four limbs;
- one deliberately reach-limited limb;
- cancel/failure;
- AUTO OFF;
- one Undo for successful supported gesture.

## Exit

Four-limb passive Sliding dependency is stable.

---

# E10 — Deterministic replay modes

## Goal

Separate historical semantic-result replay from current command-UX replay.

## Main work

Use one backend with explicit replay mode:

### recorded-result mode

- recorded Contact target/mapping/result is supplied explicitly;
- current C latch/cycle does not reinterpret it;
- expected mapping/coverage present -> actual None/missing is failure;
- if historical trace lacks enough semantic result/coverage, report incomplete context instead of guessing.

### command mode

- executes current C command semantics intentionally;
- used to test current UX behavior, not historical equivalence.

Do not silently add direct keys to old captured broad-C history if they were absent in the original capture unless the replay is explicitly a migrated/new-contract test.

## CHALLENGER GATE E10

Question:

> Can the same capture now be clearly identified as “historical recorded result” versus “current command behavior” without either mode borrowing hidden assumptions from the other?

## User debug target after implementation

- replay an existing captured sequence that previously stopped after C-cycle rule changes;
- reach the later regression point;
- separately run command mode under current C semantics;
- verify no write/repair during ordinary playback beyond intended replay execution.

## Completion ledger — 2026-09-22

Status: **USER PASS / CLOSED**.

- Added explicit replay authority split: `RECORDED_RESULT` vs `COMMAND`; `run_semantic_replay` requires the mode explicitly.
- `RECORDED_RESULT` bypasses current C latch/cycle membership and executes explicit historical Contact targets through the existing single/batch Contact planners and atomic writers.
- Missing historical mapping/direct coverage fails closed instead of guessing. Legacy broad-C with direct controls and no recorded direct footprint reports incomplete context; recorded direct binding coverage without a frozen transform payload reports unsupported direct replay rather than synthesizing current broad-C keys.
- Single Contact results now expose `mapping_contact_types`, and recorded-result replay validates actual single/multi mapping coverage after execution.
- Existing semantic-replay manifests were migrated to explicit authority: E2/E7/E8/E9 = `RECORDED_RESULT`; E3/E4/E5/E6 = `COMMAND`.
- New Contact captures record explicit `direct_binding_ids` coverage when the frozen operation domain contains supported direct controls.
- `resolve_operation_domain` was audited as read-only selection/rig-domain classification; it does not read current C latch/cycle/enabled-type authority.
- Global pre-E10 Undo/Redo regression on semantic Sliding Move and FK Move passed one gesture -> one Undo -> one Redo with no Contact rollback or pose residue.
- Static gate after final fixes: **419 pytest PASS + Ruff PASS**.
- POST re-review: **DeepSeek USER-FIRST READY + Gemini USER-FIRST READY**.
- USER FIRST historical replay: E7 completed **58/58 actions** in `RECORDED_RESULT`.
- Main follow-up after USER FIRST: same E7 capture completed **58/58 actions** in `COMMAND`; ordinary frame playback preserved the exact persistent animation signature (**58 FCurves / 58 keys before and after**), proving no passive repair/write.
- Durable E10 rule: semantic replay callers/manifests must declare replay mode explicitly; do not restore an implicit COMMAND fallback.
- Durable coverage rule: single-mapping historical replay must validate actual `mapping_contact_types`, not scalar `contact_type` alone.

## Exit

Flight Recorder captures remain useful after semantic-rule changes.

---

# E11 — AK4 Sliding Auto / multi-key matrix

## Goal

Finish the remaining Sliding Auto and multi-key authoring only after ownership/evaluation is stable.

## Main work

Reuse existing semantic writers and ANCHOR/transform closure.

Do not add:
- Auto-specific solver;
- Auto-specific key schema;
- Auto call to C;
- second Sliding closure planner.

Required cases:
- direct Sliding W Auto;
- direct Sliding E Auto;
- body-only COM/Root Auto;
- repeated same-frame replace;
- first-key baseline policy only where already allowed;
- scrub/replay;
- multi-limb;
- AUTO OFF;
- ESC;
- Undo/Redo.

Body-only Auto must not key derived limb public display, target, pole, or Contact state merely because body motion caused a solve.

## CHALLENGER GATE E11

Questions:

> Does AUTO record only the authored authority of the successful gesture, or is any derived Sliding display/result being accidentally persisted?

> Can AUTO failure leave Contact/target/body keys partially mixed across transactions?

## User debug target after implementation

The user runs the focused AK4 debug matrix. Any defect is captured once, then Main uses log/replay loops until stable. Artificial fault injection is deferred to RC1 Hardening unless a current captured defect specifically requires it.

## Closure evidence — 2026-09-22

Status: **USER PASS / CLOSED**.

- PRE: Qwen + DeepSeek completed. Both identified cross-writer Auto atomicity risk; DeepSeek additionally identified release-time AUTO OFF leakage.
- Implementation: existing Contact/direct writers gained deferred journals; mixed gesture writers prepare while OPEN, then `MutationJournal.commit_group()` validates the full set before an assignment-only COMMITTED state transition. Release-time AUTO state is re-read before every relevant modal commit. No Auto-specific solver, key schema, C call, or second Sliding closure planner was added.
- POST: the first DeepSeek POST correctly found a sequential-finalize hole. Main replaced per-journal final commit with `commit_group()`; blocker-fix re-review then passed **DeepSeek USER-FIRST READY + Gemini USER-FIRST READY**.
- Static gate: **440 pytest PASS + Ruff PASS**.
- USER FIRST / live Blender:
  - direct Sliding W Auto: frame 10 stayed at the 29-row Sliding authored bundle, exact-frame duplicates = 0, body-row leakage = 0;
  - direct Sliding E Auto: same 29-row authored bundle, exact-frame duplicates = 0, body-row leakage = 0;
  - COM body-only Auto: frame 20 wrote only COM location XYZ + semantic marker (4 rows); no Foot.L Contact/IK/public-derived rows;
  - Root body-only Auto: frame 30 wrote only Root location XYZ + semantic marker (4 rows); IK target/pole local basis stayed unchanged while their world translation followed Root once;
  - live one Undo removed the Root pose/key bundle and one Redo restored it exactly;
  - passive scrub 0↔30 preserved the exact animation signature: 37 FCurves / 51 keys before and after.
- Isolated Blender 5.2.1 runtime: `P4_E11_MULTILIMB_GROUP_COMMIT_OK`, `P4_E11_AUTO_OFF_RELEASE_NO_WRITE_OK`, `P4_E11_ESC_NO_WRITE_RESTORE_OK`, `P4_E11_SCRUB_NO_WRITE_OK`, final `P4_E11_SLIDING_AUTO_RUNTIME_OK`.
- E7 and E8 runtime regressions both passed after E11, including COM dependency authority and Root-relative target-follow-once behavior.
- The first isolated-verifier invocation failed only in verifier code by passing a Rotate-only dataclass field to a Move plan; the verifier was corrected and rerun without product-code changes.

## Exit

AK4 Sliding Auto is **USER PASS / CLOSED**.

---

# E12 — Integrated A5 stabilization regression + user acceptance

## Goal

Close the current A5 core stabilization before A6 Planted.

## Main regression

Required integrated coverage:

- Free/Sliding C;
- broad mixed C;
- same-frame mixed C;
- direct-only C expected no-op;
- Free multi Move/Rotate;
- mixed Free + Sliding Move route;
- direct body Move/Rotate with passive Sliding;
- max-extension / near-straight Sliding body solve;
- multi-limb Sliding;
- Sliding W/E;
- AUTO ON/OFF;
- Track Bar scrub;
- Move/Clone/Delete/Selection Range relevant Contact edits;
- replay recorded-result mode;
- Undo/Redo;
- red target cue;
- Root vs COM semantic contrast;
- K/KEY complete no-op.

## CHALLENGER GATE E12

Question:

> Given the final diff and proof ledger, is there any remaining contract boundary where selected authoring domain, dependency-maintenance domain, persistent write footprint, or replay authority are still conflated?

Final corrected inline-source POST:
- DeepSeek: CLEAR. One apparent `MutationJournal.commit_group` validation concern was rejected after direct source inspection showed a packet-rendering concatenation artifact rather than product code.
- Qwen: CLEAR. Hypothetical unsupported journal states were rejected because the current enum has no such states. Mid-gesture AUTO toggle semantics were recorded as VALID BUT DEFERRED RC1 hardening; E11 intentionally freezes release-time AUTO authority and no E12 blocker remained.

## Integrated closure evidence — 2026-09-22

Status: **USER PASS / CLOSED**.

- Static/regression gate before USER FIRST: `awb-check` **440 pytest PASS + Ruff PASS**.
- Blender 5.2.1 regressions passed: E7, E8, I12, I13, I16, I8, fresh E9 + E11, plus E9 `RECORDED_RESULT` replay (17 actions).
- I14 runtime verifier was stale against the already-frozen Contact Track Bar bundle contract. Main changed only `scripts/verify_phase4_i14_contact_integrity.py` to verify atomic Contact Move/Clone/Delete/Selection Range behavior; product extension source was unchanged. Corrected I14 runtime then FULL PASS.
- Frozen E5 COMMAND replay passed 12 actions across Sliding Rotate orientations/active-passive ownership. Frozen E6 COMMAND replay passed 34 actions including multi-limb Sliding, COM Move/Rotate, max-extension/near-straight and nonidentity cases.
- Fresh final handoff passed `awb-install-dev`, prior AWB Blender close, `awb-launch`, and `awb-prepare-rigped-user-baseline`; golden baseline SHA-256 remained `2d71e17f2849ec58dd5d7acaa946861fb89b99a7974406e049d78e5adefa4fe4`. Main did not pre-run the acceptance scenario.
- USER FIRST fresh session `24d9e9484f7a4054a2e84fee98e6e477` passed:
  - Foot.L C -> Free, then C -> Sliding;
  - Sliding W and E;
  - COM Move with fixed Sliding target dependency;
  - Root Move with Root-relative Sliding target transport;
  - AUTO ON -> Sliding Move `keyed:true`;
  - one Ctrl+Z + one Ctrl+Shift+Z live Undo/Redo user confirmation;
  - AUTO OFF -> Sliding Move `keyed:false`;
  - Rigped K complete no-op;
  - Root-only C fail-closed with zero persistent writes;
  - Track Bar scrub completed without errors.
- The first Track Bar scrub visibly returned Foot.L once. Investigation proved this was not a scrub regression: immediately before scrub, USER FIRST intentionally made an AUTO OFF `keyed:false` Foot.L pose edit, so the next frame evaluation returned the transient unkeyed pose to the existing frame-0 authored state. All relevant Thigh.L/Calf.L/Foot.L/IK_Foot.L curves contained frame-0 keys only; direct evaluation at frames 0/5/10/15/20/25 produced identical Foot.L/IK_Foot.L/MCH_Foot.L positions. A second 0↔25 scrub with no intervening pose edit completed normally and the user reported no repeated motion.
- Final scrub trace: `SCRUB_BEGIN -> SCRUB_END`, `cancelled:false`, max frame 25; no new error-channel events.

## Exit

E12 USER acceptance is complete. **A5 stabilization is CLOSED.**

Next:
- proceed to A6 Planted;
- after A6 acceptance, run RC0 Core Feature Complete -> RC1 Hardening -> RC2 Refactor -> RC3 Core Freeze before later roadmap work.

---

## 4. Challenger packet template for every item

Each item gets a separate request using this structure:

```text
ROLE
Independent narrow challenger. Do not implement unless explicitly asked.

ITEM
E# — exact item title.

CURRENT SOURCE
Exact SHA / diff after Main implementation for this item.

FROZEN CONTRACT
Only the semantics relevant to this item.

PROVEN BEFORE THIS ITEM
Blender/runtime facts already established.

MAIN CHANGE
What changed and why.

ONE OR TWO QUESTIONS
The exact decision boundary from this document.

REQUIRED RESPONSE
1. PROVEN CONTRACT VIOLATION
2. SOURCE-BACKED RISK REQUIRING PROOF
3. OPTIONAL ALTERNATIVE
4. NOT ENOUGH CONTEXT
5. CLEAR / REVIEW / BLOCK recommendation for this item only

The recommendation is advisory. Main classifies each finding as ACCEPTED BLOCKER / ACCEPTED IMPROVEMENT / VALID BUT DEFERRED / REJECTED / ESCALATE using current source, frozen contracts, and Blender evidence.

STOP RULE
Do not redesign adjacent Rigped systems.
Do not reopen accepted gizmo/UI.
Do not invent Planted/full-body balance/world-lock semantics.
```

Main records the challenger result before moving to the next item.

## 5. Evidence ledger per item

For every E0-E12 item record:

```text
ITEM:
MAIN DIFF / SHA:
CHALLENGER REQUEST:
CHALLENGER RESPONSE:
MAIN VERDICT:
CODE/STATIC:
BLENDER API/STATE:
GUI INPUT/SCREEN:
UNDO/REDO:
USER PASS:
OPEN RISK:
NEXT ITEM ALLOWED: YES / NO
```

Do not mark an item PASS from worker prose alone.

### E1 evidence ledger — 2026-09-20 USER PASS / CLOSED

```text
ITEM: E1 — ownership-safe mixed-C partial revert + immutable/read-only operation-domain resolver
MAIN DIFF / SHA: 24785d5 (feat: add rigped operation domain boundary)
CHALLENGER REQUEST: DeepSeek PRE awb-e1-pre-20260920-0018; Qwen POST awb-e1-post-20260920-0021; GLM POST-FIX awb-e1-postfix-20260920-0022
CHALLENGER RESPONSE: DeepSeek REVIEW; Qwen REVIEW; GLM REVIEW. Main accepted and fixed the source-backed failed-limb-capability -> DIRECT fallthrough risk, rejected the false selection-equality concern after source proof, and closed GLM's residual superset concern with an explicit invariant test.
MAIN VERDICT: CLEAR / CLOSED. Challenger recommendations remained advisory; no unresolved E1 blocker remains.
CODE/STATIC: 30/30 focused E1 + Rigped contract + representation tests PASS; changed files py_compile PASS; git diff --check PASS; unsafe Contact direct seam symbols all 0.
BLENDER API/STATE: USER-FIRST runtime checks passed on the fresh build/rigped_animate_manual_baseline.blend session.
GUI INPUT/SCREEN: USER PASS. Tested direct-only C no-op; same-limb UpperArm.L+ForeArm.L+Hand.L dedupe; independent UpperArm.R+Thigh.L mappings; exact same-frame second C cycling both mappings Free=1 -> Sliding=2; mixed UpperArm.L+Root+COM where only Contact authored and Root/COM direct keys remained absent.
UNDO/REDO: direct-only C produced no meaningful Undo; Contact cases produced one BAW_OT_contact operation as expected. Full E1 acceptance did not require a separate Undo/Redo stress matrix.
USER PASS: YES — user reported no visual pose jump, popup, or other anomaly after the focused sequence.
OPEN RISK: none that blocks E1. E2 explicit direct-subset planning and E3 atomic mixed coordination are intentionally still pending and now own the next mixed-C behavior expansion.
NEXT ITEM ALLOWED: YES — proceed to E2 only.
```

### E5 evidence ledger — 2026-09-20 USER PASS / CLOSED

```text
ITEM: E5 — one Sliding gesture lifecycle: active vs passive ownership
MAIN DIFF / SHA: c02b252600202fec367122313a4e45507007e25b
CHALLENGER REQUEST: DeepSeek PRE awb-e5-pre-deepseek-20260920-01; GLM Agent POST awb-e5-post-glm-agent-20260920-01
CHALLENGER RESPONSE: DeepSeek PRE found a real cancel/failure public-state restoration gap and the missing active/passive mapping-ID proof; it also over-promoted the final broad display reconciliation to a blocker before the passive helper body/commit-phase meaning was proved. GLM Agent fetched public mirror b72940feccf02ab5ad9442afb88736009bbe8877, reproduced 284 pytest passes, source-audited the final E5 Rotate lifecycle, and reported NO E5 BLOCKER.
MAIN VERDICT: CLOSED. Accepted the cancel exactness finding and ID instrumentation; replaced final dynamic rescan with the frozen affected set to remove ownership ambiguity; rejected the claim that the passive display helper mutates hidden IK/Contact after source inspection proved public-display-only writes. Sibling MOVE/FK_MOVE no-arg display refreshes remain a later ownership-hardening note, not an E5 Rotate blocker.
CODE/STATIC: 284/284 tests PASS; Ruff 0 on Main. New tests/test_rigped_transform_e5.py freezes OperationDomainSnapshot active ownership, disjoint/union coverage, active/passive/affected trace IDs, complete public+hidden cancel restoration, and frozen final reconciliation.
BLENDER API/STATE: USER-FIRST fresh Blender 5.2.1 session 9661249eed604aa68218e2955e2c71e7. Four limbs were authored FREE then SLIDING. User performed 9 committed four-limb Sliding Rotate gestures across multiple axes/orientations, each logging active=4 passive=0 affected=4. User then performed one committed UpperArm.L-only Rotate logging active=1 passive=3 affected=4.
GUI INPUT/SCREEN: USER PASS. User reported no visible anomaly through repeated four-limb Rotate, single-active/passive-three Rotate, and final overall check.
UNDO/REDO/CANCEL: ESC cancel on UpperArm.L-only Rotate logged TRANSFORM_CANCEL reason=ESC with active=1 passive=3; user confirmed the active arm returned exactly to pre-gesture pose and passive limbs showed no visible jump/change.
AUTO OFF: both AWB Auto and native Auto Key were OFF. Post-sequence state inspection showed 116 FCurves / 116 total keys / 116 frame-0 keys / max one frame-0 key per FCurve, exactly matching the preceding four-limb SLIDING C footprint; accepted Rotate gestures created zero additional keys.
FROZEN REPLAY: debug/user_final_tests/E5/user_final_replay.json
FROZEN REPLAY SHA-256: f7755ff7370ffd073594b74535f98468d3e6e878c25466c80cf7719957edc69c
OPEN RISK: none blocking E5. E6 still owns straight-limb singularity seed and inheritance-aware native conversion. E7 owns live body/ancestor dependency. Sibling MOVE/FK_MOVE post-write dynamic display rescans are recorded for later ownership hardening unless they become relevant earlier.
NEXT ITEM ALLOWED: YES — proceed to E6 only.
```

### E8 evidence ledger — 2026-09-22 USER PASS / CLOSED

```text
ITEM: E8 — Root semantics + red target cue
CHALLENGER REQUEST: DeepSeek PRE awb-e8-pre-deepseek-20260922-01; Qwen PRE awb-e8-pre-qwen-20260922-01; DeepSeek POST awb-e8-post-deepseek-20260922-01; Gemini POST awb-e8-post-gemini-20260922-01
CHALLENGER RESPONSE: no reviewer produced a proven E8 blocker. Qwen's PRE matrix-space objection was rejected after source/API-semantics audit: PoseBone.matrix translation is already the evaluated bone origin in armature-object space before rig.matrix_world conversion. POST reviewers found no blocking contract violation.
MAIN VERDICT: CLOSED. The red Sliding cue now reads the authored hidden IK target (IK_Hand/IK_Foot), while Root transport remains the existing hierarchy behavior with no world pin and no extra compensation delta.
CODE/STATIC: 399 pytest PASS; Ruff PASS. tests/test_rigped_ik_pivot_e8.py freezes exact state-owner -> hidden-target mapping and evaluated target world-position use.
BLENDER API/STATE: Blender 5.2.1 E8 runtime FULL PASS: reachable cue == target/result baseline; Root translation and rotation carry hidden target/pole exactly once while child matrix_basis remains unchanged; unreachable authored target remains unclamped, solver result may fall short, cue stays on the authored target; use_stretch remains false.
REGRESSION: E7 runtime PASS; I12 runtime PASS; I13 runtime PASS.
GUI INPUT/SCREEN: USER FIRST PASS. Foot.L Free -> Sliding showed the red cue. User then translated and rotated Root and reported the cue followed Root with no world-pinned residue, double motion, or visible anomaly.
AUTO OFF: AWB Auto Key OFF and Blender native Auto Key OFF throughout the accepted user session.
FROZEN REPLAY: debug/user_final_tests/E8/user_final_replay.json
FROZEN REPLAY: clean-baseline awb-replay-user-final PASS / 24 committed semantic actions.
OPEN RISK: none blocking E8. E9 owns multi-limb Sliding body dependency; the separate I19 continuous-contact broad-smoke residual remains debt.
NEXT ITEM ALLOWED: YES — proceed to E9 only.
```

### E9 evidence ledger — 2026-09-22 USER PASS / CLOSED

```text
ITEM: E9 — Multi-limb Sliding body dependency
MAIN DIFF / SHA: multi-limb architecture remained the existing E7/E8 tuple/guard design. USER FIRST exposed one production Undo/Redo defect in BAW_OT_rigped_direct_move_axis; final rigped_transform.py SHA-256 is 059fbda3ed35f8e80ccd828e777742088d14bbe54594662259b02eaa57991d37. Direct Move now uses Blender native operator bl_options {REGISTER, UNDO, BLOCKING} with zero manual bpy.ops.ed.undo_push calls.
PRE REVIEW: Qwen awb-e9-pre-qwen-20260922-01 + Gemini awb-e9-pre-gemini-20260922-01; GO after Main adjudication, with no new coordinator/body compensation/full-body solver accepted.
POST REVIEW: corrected inline packet docs/AGENT/E9_POST_UNDO_INLINE_SOURCE_PACKET_20260922.md reviewed by DeepSeek awb-e9-post-deepseek-20260922-01-corrected and Qwen awb-e9-post-qwen-20260922-01-corrected; both CLEAR. An earlier POST transport used a stale challenger mirror, was consumed, and was explicitly not counted.
MAIN VERDICT: USER PASS / CLOSED. Existing independent hidden native-IK domains remain the authority model; no body compensation, cross-limb solve, or full-body balance layer was added.
CODE/STATIC: 410 pytest PASS; Ruff PASS. E9 source tests now also freeze the native operator UNDO boundary and absence of manual undo_push in direct Move.
BLENDER API/STATE: Blender 5.2.1 E9 runtime FULL PASS after POST. Cases passed: left Sliding/right Free; both feet; foot+hand; both hands Rotate; all four; deliberate reach limit; injected one-limb authority drift whole-batch restore; AUTO OFF; native-UNDO source contract.
REACH-LIMIT EVIDENCE: deliberate Foot.L saturation produced target/result gap 1.357945 while Hand.R remained independently near target at 0.056574; no COM/body compensation or abort was introduced.
REGRESSION: E7 runtime PASS; E8 runtime PASS; I12 runtime PASS; I13 runtime PASS.
GUI INPUT/SCREEN: USER FIRST PASS. Accepted cases: both feet Sliding + COM Move; foot+hand mixed Sliding/Free + COM Move; both hands Sliding + Spine2 Rotate; all four Sliding body gesture; deliberate reach limit; four-limb ESC cancel; AUTO OFF; one Undo + one Redo.
UNDO/REDO REGRESSION + FIX: initial manual undo_push bookkeeping made the first Ctrl+Z a same-pose no-op; start-only bookkeeping restored Undo but lost Redo. Final native-UNDO patch was installed/reloaded. One Ctrl+Z changed COM by 1.7055584037 m from captured pose B and changed all observed limbs with redo_poll=true; one Ctrl+Shift+Z restored COM/Hand.L/Hand.R/Foot.L/Foot.R to pose B with exact measured delta 0.
BASELINE SAFETY: immutable authority is baselines/golden/rigped_animate_manual_baseline_v1.blend, manifest SHA-256 2d71e17f2849ec58dd5d7acaa946861fb89b99a7974406e049d78e5adefa4fe4. build/rigped_animate_manual_baseline.blend is disposable working state recreated by awb-prepare-rigped-user-baseline; .blend1 is never rollback authority.
AUTO OFF: AWB Auto Key OFF and Blender native Auto Key OFF throughout accepted USER FIRST. DeepSeek's AUTO ON Undo/Redo suggestion is deferred follow-up, not an E9 blocker because the canonical E9 USER FIRST scope is AUTO OFF.
FROZEN USER FINAL: debug/user_final_tests/current.json -> E9. Curated clean-baseline semantic replay PASS / 17 actions; native cancel and Undo/Redo remain preserved as live evidence because they are UI/history outcomes rather than semantic replay actions.
OPEN RISK: no E9 blocker. Separate I19 continuous-contact broad-smoke residual remains debt.
NEXT ITEM ALLOWED: YES — E10 may proceed.
```

## 6. Explicit non-goals until this plan closes

Do not start or redesign:

- A6 Planted;
- world-locked Root Sliding;
- full-body balance/compensation;
- native IK constraints on visible ForeArm/Calf;
- new key database;
- new general transaction framework;
- global depsgraph “repair everything” callback;
- arbitrary hierarchy migration;
- accepted gizmo visual/hit behavior;
- unrelated Track Bar UI redesign.

## 7. Immediate next action

**E1 through E12 are USER PASS / CLOSED. A5 stabilization is CLOSED.**

Proceed to **A6 — Planted** using the current Phase 4 product contracts and the next focused A6 implementation plan.

Do not reopen E1-E12 without new concrete regression evidence. The separate I19 continuous-contact broad-smoke residual remains debt and does not reopen A5.
