# Blender Animation Workbench — Blender Runtime / Verification Rules

> Updated: **2026-09-15 KST**
> Runtime target: **Blender 5.2.1 LTS**

## 1. Native-first implementation

Implementation preference:
1. verify Blender 5.2.1 native behavior/API,
2. reuse native data/operations where possible,
3. wrap native mechanisms with AWB semantic/UX adapters,
4. custom Python only for missing behavior,
5. native/C++ candidates only after an actual measured Python/API limitation.

Do not reimplement functionality merely because the native mechanism was not investigated.

For version-sensitive behavior, prefer Blender 5.2 Manual/API and actual 5.2.1 runtime probes. Newer/current docs are supporting evidence only.

Blender 5.2 Action Slot/ChannelBag behavior must be considered where applicable; do not assume legacy `Action.fcurves` access is the product model.

## 2. Animation authority

Unless a task-specific canonical document explicitly states otherwise:
- Blender native animation data is authoritative replay state.
- Blender native selection is the edit-target source of truth.
- Hidden Python caches/state must not become authority for authored animation.
- Queries must remain read-only where the project contract says they are read-only.
- Unsupported, ambiguous, linked, or read-only mutation targets fail closed.

Task-specific frozen Phase 4 semantics override generic examples in this guide.

## 3. Mutation / rollback

Persistent mutation must preserve project atomicity contracts.

When a task uses a mutation journal/rollback path:
- every persistent write in the operation belongs to the intended atomic closure,
- injected faults must restore the exact prior state when exact rollback is required,
- interpolation, handles, key coordinates, constraint values, custom properties, and related native state must not be silently normalized on rollback,
- transient solver/snap changes must not leak into persistent authored state unless the operation explicitly commits them.

Undo/Redo behavior is part of the user-facing contract, not merely an implementation detail.

## 4. Research-first behavior decisions

For ambiguous animator UX, research before coding. When relevant compare:
- current 3ds Max official user behavior,
- Max public API/controller model,
- Blender 5.2.1 native behavior/API,
- intended AWB behavior.

Use external DCC behavior as a requirement/reference, not source code to copy. AWB production naming and implementation remain independent.

## 5. Verification levels

Keep these distinct:

```text
CODE PASS
= pytest / Ruff / static checks

BLENDER API PASS
= Blender 5.2.1 background/runtime verifier

STATE PASS
= expected native Blender FCurve/selection/frame/rig state proven

GUI INPUT PASS
= actual Windows mouse/keyboard operation reaches Blender UI

SCREEN PASS
= visual before/after evidence matches the expected result
```

A task only needs the levels required by its acceptance contract, but do not substitute a lower level for a required higher level.

GUI behavior is not complete from CODE PASS or background Blender alone.

## 6. GUI verification

Use the project's registered GUI/Blender verification path when available. Reuse the designated verification Blender/session rather than spawning competing GUI instances.

For GUI acceptance when required:
- use actual input semantics,
- verify resulting Blender-native state,
- verify Undo/Redo when part of the contract,
- check relevant AUTO/keymap/modal/repeat behavior when applicable,
- capture visual evidence when the verifier requires it.

Avoid broad manual shell automation when a project-registered verifier/scenario exists.

### 6.1 Blender MCP / bounded App-Control surface

The AWB development environment already includes a Blender MCP bridge with Blender-side Python execution. Do not regress to screenshot coordinate guessing when Blender can provide semantic state or exact UI geometry.

Use these layers deliberately:
- `awb.blender.v1` semantic queries are read-only observation/geometry APIs. The registered surface includes Track Bar, viewport target, selection, transform/gizmo, rig/bone/pose, constraints/IK, Rigped/Contact, keying/timeline, Action/FCurve, drivers, NLA, motion paths, shape keys, deformation/weights, character metadata, depsgraph, keymap/operator discovery, generic RNA inspection, and UI-region/layout queries.
- `interaction.recipe` converts common AWB semantic operations into exact bounded `click` / `drag` / `key` action batches using current Blender geometry. Prefer it over manual coordinate derivation.
- `app_control_act` remains the actual GUI-input path for GUI acceptance. A semantic query or direct Blender mutation does not count as GUI INPUT PASS.
- Named Task `awb-mcp-blender-exec` is the general Blender-internal development/mutation bridge for setup, inspection helpers, rigging, animation authoring experiments, and deterministic backend manipulation. Its request is `build/blender_mcp_request.json`; it accepts Blender-focused Python and statically blocks general OS/network/process imports and dangerous builtins. It must not be used to claim that a GUI interaction path passed.

For future rigging/animation work, do not add one-off queries just because a new RNA property is needed. Use `rna.inspect`, `fcurve.inspect`, `operator.catalog`, and `keymap.summary` first. Add a dedicated query only when a stable high-value semantic summary or geometry contract is genuinely useful.

For UI tests, query current geometry immediately before the action. A minimized/restored window may change client geometry; reattach/requery after restoration rather than reusing stale coordinates.

## 7. Scope-specific contracts

Before touching Phase 4 keying, Contact, rig representation, IK/FK, Track Bar, or GUI behavior, read the task-specific canonical plan/verification document named by Main. Those documents contain frozen semantics that this generic guide intentionally does not duplicate.

Do not infer future Phase behavior from reserved enum values, placeholders, or deferred roadmap items.
