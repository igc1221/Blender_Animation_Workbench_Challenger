# AWB Phase 4 — Product Contracts

> Updated: **2026-09-23 12:46 KST**
> Role: **authoritative Phase 4 product-behavior contract**. This file owns user-visible Rigped semantics. Historical reviews, milestone proofs, and implementation experiments must not override it.

## 1. Authority and layering

```text
Blender native selection        = current edit-target authority
Blender native animation data   = animation/replay authority
Phase 3 Character semantics     = authored identity/relationship authority
Phase 2 semantic adapter        = native control / Action Slot / ChannelBag addressing path
Phase 4                         = Rigped operation semantics, bounded mutation, kinematics/contact behavior
```

No parallel hidden Character database, Contact-session cache, or independent animation authority is allowed.

## 2. Rigped lifecycle

Primary flow:

```text
New Rigped -> Figure/Fit -> Apply -> Animate
```

Current practical baseline:

- Figure/Fit F1-F6 is USER PASS / CLOSED and is the structural editing baseline.
- live Figure interaction is Object-hosted and semantic; native EditBone/rest mutation is bounded to atomic Apply.
- Animate A1-A6 is the accepted first animation-core baseline.
- Figure and Animate use the same AWB Move/Rotate/Scale visual language where the corresponding operation is supported.
- Figure edits rest/setup structure. Animate edits pose/animation only; Animate must not silently change rest structure.
- Root, COM, and Pelvis are distinct authored roles.
- hidden mechanism, deform, IK target, and pole controls are never normal animator selection targets.
- post-animation structural Figure/rebuild migration remains outside the current frozen core and must not be inferred from raw FCurve preservation alone.

## 3. Selection grammar

AWB scene/Rigped selection grammar:

```text
Click                  = replace selection
Ctrl + Click           = add
Alt + Click            = remove
Double Click           = semantic Chain/Group whole-select
Ctrl + Double Click    = semantic Chain/Group add
Alt + Double Click     = semantic Chain/Group remove
```

`Ctrl`, not `Shift`, is AWB add-selection. Semantic double-click uses authored Phase 3 Chain/Group identity, not name guessing or raw hierarchy traversal.

## 4. Transform semantics

`W / E / R` are semantic animator gestures, not a promise to write Blender Location/Rotation/Scale of the selected bone.

### Animate

- ordinary accepted FK controls preserve FK hierarchy behavior.
- parent rotation carries authored descendants where the hierarchy contract says it should.
- ordinary Rigped animation Scale is locked/unsupported unless an explicit future capability opts in.
- connected limb controls must not be translated by blindly writing raw local Location.
- visible ForeArm/Calf must **not** own Blender native IK solver constraints directly. That experiment was reverted because it caused yellow IK-chain coloration and arm/leg inconsistency.
- hidden/MCH chains own solver implementation. User-facing controls remain Biped-like and do not expose a separate target/pole workflow.

### Fit

- Fit owns structural proportions, joint placement, and rest hierarchy edits.
- Fit may use fixed-length upstream/downstream structural solving where needed; this does not make Animate an EditBone/rest operation.
- current accepted Blender bone-axis convention remains: local Y = Head→Tail / bone-length axis.

## 5. Biped-style Contact / IK model

User-facing states:

```text
Free     = FK-authoritative limb state
Sliding  = IK-authoritative contact state
Planted  = IK-authoritative held/joined contact state
```

User-facing IK UX:

- animator never selects or drags a separate Blender IK target or pole.
- hidden `IK_Hand.*` / `IK_Foot.*` and pole bones remain implementation-only and hidden/unselectable.
- the visible red point is the Biped-style IK/contact pivot cue at the hidden IK target position.
- Free hides the red point.
- Sliding and Planted show the red point.
- confirmed Track Bar Contact colors:
  - Free = gray
  - Sliding = yellow
  - Planted = light blue
- do not put native IK constraints directly on visible ForeArm/Calf bones to make them follow the solver.

### Sliding / Planted distinction

- both are IK-authoritative.
- Sliding may establish/update a movable contact reference for its authored episode.
- Planted joins/holds the contact reference according to the supported plant contract.
- neither state authorizes hidden Root/COM compensation, full-body balancing, arbitrary stretching, or silent movement of external scene owners.

## 6. Rigped key-authoring contract — critical

This contract is intentionally explicit because earlier Phase 4 documents contradicted it.

```text
Rigped C
= the ONLY explicit Rigped key-authoring command for Free / Sliding / Planted
= creates a Contact key when none exists at the current frame
= same-frame repeated deliberate presses cycle:
  Free -> Sliding -> Planted -> Free
= when one broad mixed Rigped selection contains Contact limbs plus ordinary/direct authored controls,
  one C operation must author the touched Contact limb domains coherently AND author the selected direct controls as supported Free keys at that frame
= the direct controls remain Free while the selected Contact group advances its Contact semantic
= on supported direct-only body controls, C authors the normal Free/direct key bundle at that frame
= direct-only body C must not create Sliding/Planted Contact authority

Rigped K / KEY
= complete no-op
= writes ZERO Rigped keys
= no Free key
= no Sliding key
= no Planted key
= no Contact closure
= no ordinary Rigped transform key
= no pose mutation
= no selection mutation
= no Undo entry
```

Non-Rigped `K` remains governed by the general AWB Set Key system outside this Rigped contract.

`All Key` is a separate explicit feature and must never be used as evidence that public Rigped `K` applies.

### Rigped AUTO — existing Rigped authoring의 자동 trigger

`AUTO`는 새 Rigped key system이 아니다. 이미 검증된 Rigped FK/Contact/Sliding authoring이 확정한 결과를 **지원 transform release 시 자동으로 기록하게 하는 trigger**다.

Frozen ownership:

```text
existing Rigped FK / Sliding solver
    = pose와 실제 authored authority를 결정

existing Rigped semantic key writer
    = 그 authority를 Action/FCurve semantic bundle로 기록

C
    = Free/Sliding/Planted 상태를 명시적으로 변경하는 authoring command

AUTO
    = 상태를 변경하지 않고 현재 transform 결과를 기존 semantic writer로 자동 기록

Rigped K / KEY
    = complete no-op

All Key
    = separate explicit bundle authoring
```

따라서 AUTO 구현은 다음을 금지한다.

- C cycle/replant 로직 재구현 또는 AUTO에서 C command 호출.
- Free/Sliding 판정 규칙 재설계.
- FK Move/Rotate solver 또는 Sliding solver 재구현.
- 기존 Contact/FK closure를 AUTO 전용으로 다시 계산하는 두 번째 closure planner.
- 일반 Rotation/Location FCurve만 별도로 기록하여 Rigped semantic marker/state를 우회하는 writer.
- AUTO 전용 key color/state schema.
- AUTO 구현을 이유로 기존 C, Sliding, FK USER PASS 의미를 변경하는 것.

AUTO ON에서 transform이 성공하면 **기존 Rigped semantic row/closure 생성 규칙을 그대로 사용**한다. 현재 소스의 state-preserving Contact ANCHOR 및 기존 transform-row builder가 semantic bundle의 authority이며, AUTO는 그 low-level write path를 trigger-specific side effect 없이 호출할 수 있도록 필요한 최소 seam만 추가한다. C의 authoring latch, cycle, replant 같은 명시 authoring side effect는 AUTO에서 발생하지 않는다.

AUTO OFF는 기존 pose-only transform을 그대로 유지한다. ESC/RMB, failed solve, zero effective transform은 animation mutation 0개다. Preview/mousemove/depsgraph/scrub은 authoring trigger가 아니며, successful transform release 한 번만 기록한다.

Contact가 아직 명시적으로 작성되지 않은 limb의 현재 physical/effective authority가 기본 Free라면 AUTO는 **Free를 새로 선택하거나 C를 실행하는 것이 아니라 현재 Free authority를 semantic Free key로 기록**한다. Track Bar 색/marker도 기존 Rigped semantic key 규약을 그대로 사용한다.

First-key baseline은 AUTO만의 보조 정책이다. 아직 animation이 없는 Free/direct authored family에서 이후 frame의 첫 AUTO가 발생하면, 필요 시 변형 전 authored authority를 0F에 보존하고 현재 frame 결과를 기록할 수 있다. 이 baseline은 기존 Rigped semantic bundle과 동일 transaction 안에서만 존재하며 Sliding/Planted authority나 hidden IK state를 새로 발명하지 않는다.

일반 Object와 custom/external bone Auto는 기존 AWB 경로를 유지한다. generated Rigped + custom/external mixed gesture는 atomic ownership이 증명되기 전까지 fail-closed한다.

AUTO의 상세 구현 이력은 `PHASE4_RIGPED_AUTO_KEY_IMPLEMENTATION_PLAN.md`에 보존한다. 현재 AUTO 제품 계약과 RC0 동결 의미는 이 문서와 통합 로드맵이 우선한다.

## 7. C state-time behavior

- every deliberate valid C press is one authoring operation.
- held-key repeat events must not produce multiple cycles.
- each supported limb owns a lightweight **authoring latch** that remembers the last Contact semantic explicitly confirmed by successful C authoring.
- when the current frame has no Contact state key, C stamps that latched semantic regardless of whether the new frame is earlier or later on the timeline; when the latch is uninitialized, the first enabled type is used, normally Free.
- when the current frame already has a Contact state key, C replaces the complete current-frame bundle with the next enabled state and updates the authoring latch to that new type.
- deleting the final Contact bundle for a limb resets its authoring latch to UNINITIALIZED, so the next first C begins at Free again.
- the authoring latch is **not animation/replay authority**. Contact state replay still comes only from native Action/FCurve data, and Undo/Redo must keep authored state and latch synchronized.
- Contact state interpolation is discrete/constant.
- incomplete Contact bundles fail closed; playback/scrubbing never repairs them through frame handlers.

## 8. RC0 frozen practical core baseline

The first complete Rigped animator core is USER-accepted:

- Figure/Fit F1-F6 is closed and provides the structural editing boundary.
- A1-A4 FK/keying foundation is USER PASS.
- A5 Free/Sliding plus E1-E12 stabilization is USER PASS / CLOSED.
- A6 Planted is USER PASS / CLOSED.
- Free / Sliding / Planted Track Bar semantics and red-pivot visibility are accepted.
- C, AUTO, semantic W/E/R, supported multi-selection/domain partitioning, atomic rollback and native Undo/Redo are part of the RC0 core.
- supported direct-only body C writes normal Free/direct keys and no body Contact authority.
- direct native IK constraints on visible ForeArm/Calf remain rejected/reverted.
- Blender Action/FCurve remains authored replay authority.

RC0 is CLOSED and this section is the frozen animator-facing baseline. There is no A7 feature currently defined. RC1 practical-use hardening is active, followed by RC2 -> RC3 before unrelated feature expansion. The evidence map is `docs/PHASE4/verification/PHASE4_RC0_CORE_FREEZE_EVIDENCE.md`.

## 9. Multi-selection and solver-domain boundary

- multiple selected controls from one semantic limb deduplicate to one limb domain.
- independent limb domains may be batched only when dependency/write footprints are independent.
- overlapping/shared writable COM/Pelvis/full-body dependencies fail closed until a dedicated solver domain exists.
- selecting multiple visible controls is never permission to invent a hidden full-body IK solve.
- one user operation must either succeed coherently for its whole supported domain or leave no partial authored result.

## 10. Contact target-space boundary

Supported semantic target-space categories may include World, Character/Root, local/pose, and explicit external object spaces, but each must be runtime-proven before mutation.

- external scene objects remain externally owned.
- Animation World Offset must not silently move externally owned contact objects.
- unsupported cycles, scale/shear, singular reach, or ambiguous spaces fail closed.

## 11. Key references

Detailed Rigped Animate implementation spec and USER PASS ledger: `docs/PHASE4/planning/rigped/PHASE4_RIGPED_ANIMATE_IMPLEMENTATION_SPEC.md`

Execution/dependency status: `docs/PHASE4/PHASE4_INTEGRATED_IMPLEMENTATION_ROADMAP.md`

Current blocker/next action: `docs/PHASE4/PHASE4_CURRENT_SESSION.md`

Manual acceptance: `docs/PHASE4/verification/PHASE4_USER_TEST_SCOPE.md`

RC0 frozen evidence map: `docs/PHASE4/verification/PHASE4_RC0_CORE_FREEZE_EVIDENCE.md`

Verification rules: `docs/PHASE4/verification/PHASE4_VERIFICATION_STRATEGY.md`

Deep behavior/reference evidence only: `docs/PHASE4/research/PHASE4_BIPED_KINEMATICS_RESEARCH.md`
