# Blender Animation Workbench — Worker Orchestration

> Updated: **2026-09-24 22:00 KST**
> Authority: **Main / Sol**
> Purpose: turn model-specific strengths/weaknesses into better task packets and lower Main repair cost.

> **MANDATORY TWO-SIDED REVIEW GATE:** every non-trivial implementation/stabilization slice has two blocking review stages. **PRE:** send the proposed design/contract to at least **2 independent reviewers/workers**, wait for and consume both responses, then Main synthesizes them with its own judgment before implementation. **POST:** after implementation, send the actual current source/diff to at least **2 independent reviewers/workers**, wait for and consume both responses, disposition every material finding, fix as needed, and rerun regression gates. USER FIRST / user-final testing starts only after the POST gate is closed. A sent-but-unread review is still open work and blocks progression.
>
> **STRICT REVIEW COUNT:** each non-trivial slice has exactly **1 PRE round** and exactly **1 POST round**. Never repeat reviewer rounds until they report zero findings, and never add a same-slice closure/re-review round. Main owns all finding disposition and closes fixes with regression/runtime evidence after the single POST round.

> **REVIEW PROVIDER POLICY:** PRE/POST review and verification use **DeepSeek / Qwen / Gemini as the first-tier External Web Bridge pool**, with at least 2 independent providers per round. Rotate pairings across successive gates to avoid provider-perspective bias; do not hard-code one provider as the permanent first choice. **GLM is the slower fallback** and is used when the first-tier pool is unavailable/occupied or an extra independent view is materially useful. Luna is not the default reviewer/verifier; use Luna for implementation/coding tasks where its throughput is useful. Superseded Luna review requests do not satisfy or block the replacement web-review gate after an explicit provider-policy change; clean those lanes when available.
>
> **MANDATORY PROGRESSIVE-SKILL GATE:** every substantive Luna **implementation** dispatch must carry the relevant AWB progressive skills explicitly. Main first inspects the skill catalog metadata, selects the smallest relevant skill set, and passes it through `awb_skills`. Empty skill injection requires an explicit no-relevant-skill rationale. Worker-learning APPLY decisions are not archival notes: an APPLYed lesson must be exercised by including its target skill on the next relevant Luna implementation task. External Web reviewers use their bridge-selected skills, but Main must verify the resulting skill/evidence packet is appropriate before the review can satisfy a PRE/POST gate.

## 1. Core rule

Parallel-worker quality is not judged only by raw model capability. Main owns the control problem:

```text
useful worker output = model capability
                     × context sufficiency
                     × task fit
                     × worker-specific prompt compensation
                     × Main validation
```

Do not send the same generic request to every worker. Before dispatch, Main must read the relevant worker profile and shape the request around that worker's observed failure modes.

Profiles are evidence-driven operating notes, not personality stereotypes. Update them only from actual AWB tasks or clearly comparable project work. Mark uncertain traits as provisional.

## 2. Mandatory dispatch pipeline

For every non-trivial worker task, Main performs these gates in order.

### A0. Progressive-skill gate

Before every substantive Luna launch:
1. inspect `list_awb_skills` metadata;
2. choose the smallest relevant skill set;
3. pass those ids explicitly through `awb_skills`;
4. after the response, process every worker-learning candidate as APPLY / DEFER / REJECT;
5. if APPLY targets a skill, the next relevant Luna request must include that skill;
6. only after response consumption + learning review may the lane be cleaned and reused; use the registered Named Task `awb-luna-lanes-clean` as the canonical cleanup path rather than direct per-lane cleanup calls.

Default mapping:
- implementation / semantic architecture / bounded source changes → `worker-implementation`;
- Blender runtime, GUI, state, Undo, replay, acceptance → add `blender-runtime-verification`;
- PRE/POST reviewer evidence, challenger packets, source-backed verdicts → add `review-evidence`;
- worker-owned branch/worktree mutation or Git integration → add `git-worktree-safety`.

Do not attach every skill mechanically. Use the minimum set that materially applies. `awb_skills=[]` on a substantive Luna task requires an explicit reason in the dispatch record.

For External Web challenger requests, the bridge selects skills automatically for the packet. Main still verifies that the selected skill/evidence packet fits the task; a malformed or skill/evidence-deficient request does not count toward the mandatory reviewer minimum until corrected.

### A. Task-fit gate

Choose the worker for the job rather than filling idle capacity.

- implementation vs review vs GUI verification vs architecture
- repository access needed or self-contained packet sufficient
- expected file overlap with other workers
- quota/cost sensitivity
- runtime/GUI evidence requirements
- risk if the worker over-expands scope or misses a frozen contract

If the task does not benefit from parallelism, Main does it directly.

**Repeated-debug reset invariant:** once Main triggers the three-attempt rollback-first reset for a bug family, no worker may continue reasoning from the contaminated failed-patch stack. New implementation/analysis packets must name the last-known-good checkpoint and the replay boundary immediately before the defect. Prior failed diffs may be supplied only as rejected-hypothesis evidence. The worker must propose a different root-cause path and must not ask the user to rebuild already-captured setup/keying.

Worker availability/quota is a live runtime condition, not a durable quality grade. Check current availability before dispatch; if a provider or Luna is quota-limited/unavailable, skip it without rewriting its quality profile.

Current routing priority:

```text
1. Main FAST PATH for tiny low-ambiguity work.
2. Luna High pool: implementation/coding only by default; 2 lanes normally, optional third High burst lane only for real low-overlap implementation work.
3. External Web review pool: **DeepSeek / Qwen / Gemini are co-equal first-tier reviewers**; rotate pairs across PRE/POST gates for diversity. Use at least 2 independent providers per round. GLM is fallback because of latency.
4. Luna Max: one exclusive deep-reasoning/implementation lane for genuinely difficult work. Drain High first; High and Max never overlap. Do not use it for routine PRE/POST review.
5. Antigravity: dormant capacity only; preserve integration, do not assign normal work.
6. Astra: manual high-value architecture escalation.
```

**External Web challenger invariant:** for every non-trivial AWB implementation/stabilization batch, Main sends at least one small decision-boundary review through the provider-agnostic External Web Bridge using the current frozen contract plus exact relevant source/diff; that reviewer may occupy one slot in the mandatory PRE/POST minimum of two independent reviewers. Select review providers from the first-tier pool **DeepSeek / Qwen / Gemini** with deliberate rotation across successive gates (for example D+Q, Q+G, G+D) so evidence is not repeatedly filtered through the same model pair. Use GLM only as fallback or when a fourth independent perspective is materially justified. Record concrete hits, false alarms, context failures, and Main repair burden in the worker ledger so routing can become evidence-driven later. Launcher auto-open/import behavior remains the dispatch mechanism. Reviewer conclusions are advisory, but once a request is sent **Main must consume and disposition it before that gate closes**. A consumed review remains authoritative closed work even though the active Launcher lane hides it as clean; paired REQUEST/RESPONSE evidence is retained until session handoff/worker-clean and must not be mistaken for a lost response. If a bridge/provider is unavailable, record that explicitly and fill the missing reviewer slot with another independent reviewer rather than lowering the minimum.

**Luna concurrency is a fixed orchestration invariant:** normal operation may use **two High workers in parallel**; when a third genuinely independent/low-overlap slice exists, Main may use **three High workers in parallel** via burst. **Max is one worker only and must run alone with respect to Luna**: all High lanes must be drained before Max starts, and no High lane may launch while Max is active.

**Luna reasoning/service-tier invariant:** keep Luna reasoning at **`xhigh`**. When supported by the Luna sidecar runtime, enable Codex **Fast mode as a separate service tier** (`service_tier = "fast"`, `[features].fast_mode = true`) rather than lowering reasoning effort. Current Harness status/launch schemas must be checked before claiming Fast is active; `xhigh` alone does not imply Fast.

**Luna parallel diversity invariant:** parallel Luna lanes for one Main blocker must receive **complementary subproblems by default, not duplicated generic prompts**. Split work across orthogonal responsibilities such as source/control-flow diagnosis, contract/math/solver analysis, bounded implementation/test design, and adversarial regression/failure-path review. Duplicate independent prompts are reserved for an explicit high-risk A/B decision where replication itself is the goal. Every dispatch must state its unique lane deliverable and forbidden overlap; only one lane owns coupled production mutation unless Main intentionally requests competing implementations.

**Luna High granularity is also a fixed orchestration invariant:** High lanes receive micro-slices, not an entire implementation slice. Target one concrete deliverable per lane, usually 2-5 minutes of worker wall time, with explicit `OWN / DO / PROVE / DO NOT / STOP` instructions. If a request combines investigation + production patch + multiple regressions + Blender verification + documentation/commit, Main must split it first. When three independent low-overlap micro-slices exist, use all three High lanes rather than serializing them behind one oversized request. A 10+ minute High task or repeated timeout is a decomposition signal; do not solve it by repeatedly extending the same timeout. See `WORKER_PROFILES/LUNA.md` for the mandatory detail.

**Launcher worker-state invariant:** the Launcher `Workers` panel is the human-visible authority for the pinned project's Luna lane cleanliness (`작업중 / 클린 / 변경있음 / 사용불가`). Before handoff/session close, Main checks this state directly; the user does not need to separately tell Main “워커클린”.

### B. Context-sufficiency gate

Before sending, ask: **Could a competent worker answer correctly from only this packet/worktree?**

For worktree workers, `git status clean` is not a context check. Before launch Main must verify that every task-critical current file actually exists in that checkout and that any required uncommitted Main snapshot has been mirrored. Record the source snapshot/HEAD or exact mirrored-file set. If context parity is missing, do not launch the worker.

A production request normally supplies or points to:

1. current task/milestone and exact goal,
2. canonical SOT documents required for this slice,
3. exact affected source/diff or permission to inspect it,
4. frozen product/architecture contracts that must not change,
5. authoritative runtime/state assumptions,
6. explicit non-goals and forbidden files/areas,
7. dependency/API contracts with neighboring code,
8. required tests/runtime/GUI evidence,
9. expected output/report format,
10. what to do when context is missing or contradictory.

For web-only workers, package the actual relevant source excerpts; do not assume filesystem visibility.

**External Web Bridge source-visibility invariant (2026-09-21):**
- A GitHub/pinned-commit URL in a challenger packet is only a **locator/provenance hint**, not proof that the provider can fetch or inspect that source.
- Unless the provider response explicitly proves repository access by citing concrete fetched source content, Main must assume the linked GitHub files were **not read**.
- PRE/POST review packets for web-only providers must therefore include the exact decision-critical source/diff **inline in the request body** (or in another directly consumable inline evidence block). Do not rely on `context_files` being converted to readable source.
- Before a review can count toward the mandatory reviewer minimum, Main verifies the generated REQUEST artifact itself and confirms that the required function/class/diff text is actually present inline.
- A response that reports link-only evidence, no fetch channel, or source-unavailable status is an **evidence-packaging failure**, not a valid source review. Correct the packet and resend the same review gate; do not grade the provider for missing source it was never given.
- Record this as a Main orchestration defect in the worker ledger when it happens. Distinguish it from providers/modes that have separately demonstrated real repository fetch capability (for example, a proven Agent-mode fetch in a prior task).

### C. Worker-profile compensation gate

Read `docs/AGENT/WORKER_PROFILES/<worker>.md` and add targeted guardrails.

Examples:

- a worker that infers missing behavior too quickly must be told to **prove absence by exact source search before claiming a gap**;
- a worker that over-expands patches gets a strict file allowlist, line-of-responsibility boundary, and `STOP AND REPORT` rule for adjacent defects;
- a worker that follows green tests too readily gets the frozen semantic contract repeated before the test command;
- a worker strong at review but without repository access gets a self-contained evidence packet and explicit `MISSING PROJECT CONTEXT` escape hatch.

Do not blindly copy old compensation text. Use only traits still supported by the performance ledger.

### D. Deliverable gate

State exactly what Main wants back.

Implementation tasks should request:

- files changed,
- behavior implemented,
- focused tests/runtime checks,
- remaining uncertainty,
- no unrelated cleanup,
- diff/commit or worktree status as appropriate.

Review tasks should separate:

- proven contract violations,
- likely risks requiring proof,
- optional alternatives,
- minimal patch suggestions,
- assumptions not established by evidence.

### E. Evidence gate

Never accept a worker claim merely because it sounds plausible or its own tests are green.

Main validates against this order:

```text
actual Blender 5.2.1 runtime
> frozen/current AWB contracts
> current Main source/diff
> official documentation when relevant
> worker reasoning/self-tests
```

GUI acceptance additionally requires the project's real INPUT / STATE / SCREEN / UNDO evidence when specified.

### F. Speed / dispatch-latency gate

Do not pay worker startup, context packaging, and result-integration cost for work Main can close faster with authoritative evidence already in hand.

Use two paths:

```text
FAST PATH
- frozen contract already known,
- current Main source already loaded,
- narrow reproducer/runtime task already available,
- expected fix/test is local and low-ambiguity.
=> Main executes directly, then optionally asks one cheap challenger for review.

FULL WORKER PATH
- independent source exploration is valuable,
- implementation can proceed in parallel with low overlap,
- the bug/contract ownership is genuinely uncertain,
- GUI/runtime work can be isolated productively,
- a second implementation or challenger materially lowers risk.
=> dispatch the best-fit worker with the full context gate.
```

Before any worktree worker launch, perform a short preflight and do not launch until all three are true:

1. task-critical snapshot parity is proven,
2. the exact frozen contract for the assigned semantic boundary is present,
3. the expected evidence/deliverable is explicit.

If those checks fail, fixing the packet is cheaper than recovering from a long stale or mis-scoped run.

Prefer one bounded worker task with a clear stop condition over a broad "review everything" request. Prefer Main runtime validation immediately after a useful worker finding instead of sending the same finding through multiple workers unless the decision remains ambiguous.

## 3. Request construction template

Main should mentally or explicitly build every substantial worker request from these blocks:

```text
ROLE
- why this worker was chosen
- implementation/review/GUI/challenger role

CURRENT GOAL
- one bounded outcome
- current milestone/slice

AUTHORITATIVE CONTEXT
- SOT docs
- exact source/diff
- runtime facts already proven

FROZEN CONTRACTS
- behavior that may not be redesigned
- exact state/value/API semantics where relevant

ASSIGNED OWNERSHIP
- files/responsibilities allowed
- overlap rules with Main/other workers

NON-GOALS / STOP RULES
- adjacent areas not to redesign
- when to stop and report rather than repair

WORKER-SPECIFIC GUARDRAILS
- generated from that worker's current profile

VERIFICATION
- focused tests
- Ruff/static
- Blender runtime
- GUI proof if required

OUTPUT CONTRACT
- exact report/diff/evidence expected
- distinguish proven facts from suspicions
```

If one of these blocks is materially missing, Main should fix the packet before dispatch rather than blaming the worker later for predictable ambiguity.

## 4. Parallel overlap control

Main assigns ownership by responsibility and preferably by file.

Before dispatching two workers simultaneously, record:

- primary owned files,
- allowed read-only neighboring files,
- forbidden overlap,
- shared contracts both must obey,
- which worker is implementation authority for that slice,
- which worker is challenger/verifier only.

Two workers may inspect the same source for independent review, but should not both mutate the same production file unless Main deliberately wants competing implementations.

## 5. Worker learning loop

After every meaningful worker result, Main appends a compact evidence entry to `WORKER_PERFORMANCE_LEDGER.md`.

Record:

- worker/model,
- task type and difficulty,
- result accepted/rejected/partial,
- useful strengths observed,
- concrete mistakes or wasted effort,
- Main repair burden,
- prompt/context defect attributable to Main,
- next-request compensation rule.

Then update the worker profile only when the evidence changes a durable routing/prompting rule.

Important: distinguish **worker failure** from **Main dispatch failure**. If the packet omitted a frozen contract, source file, test requirement, or ownership boundary, record that as a Main orchestration defect and fix the template/profile instead of simply demoting the model.

## 6. Main repair-budget policy

A worker is valuable when it reduces total Main effort.

Classify each substantial result:

```text
A — near-direct integration; only mechanical packaging/review needed
B — useful result; small local semantic correction needed
C — partial; significant Main repair/rewrite needed
D — misleading/wrong direction; cheaper for Main to redo
```

Routing guidance:

- repeated A/B => expand suitable task scope gradually;
- repeated C => narrow task size, strengthen context/guardrails, or demote route;
- D on a well-specified task => reject result and materially reduce similar assignments;
- D caused by missing Main context => fix orchestration first, then re-evaluate the worker.

Do not keep a worker busy merely because quota is available.

## 7. Active profiles

Read only the profile needed for the current dispatch:

- `WORKER_PROFILES/LUNA.md`
- `WORKER_PROFILES/DEEPSEEK_WEB.md`
- `WORKER_PROFILES/QWEN_WEB.md`
- `WORKER_PROFILES/GEMINI_WEB.md`
- `WORKER_PROFILES/GLM_WEB.md`
- `WORKER_PROFILES/ANTIGRAVITY.md`
- `WORKER_PROFILES/ASTRA.md`

## 8. Update discipline

Worker profiles are living operational documents.

- use dated observations,
- separate `Observed` from `Provisional`,
- remove stale compensation rules when later evidence disproves them,
- avoid vague labels such as "smart" or "bad at code",
- describe actionable behavior: scope drift, evidence discipline, test bias, patch size, context sensitivity, runtime/tool reliability, etc.

The goal is not to characterize models socially. The goal is to make the **next request measurably better**.
