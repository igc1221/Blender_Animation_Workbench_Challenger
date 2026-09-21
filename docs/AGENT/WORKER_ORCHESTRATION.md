# Blender Animation Workbench — Worker Orchestration

> Updated: **2026-09-21 15:55 KST**
> Authority: **Main / Sol**
> Purpose: turn model-specific strengths/weaknesses into better task packets and lower Main repair cost.

> **MANDATORY TWO-SIDED REVIEW GATE:** every non-trivial implementation/stabilization slice has two blocking review stages. **PRE:** send the proposed design/contract to at least **2 independent reviewers/workers**, wait for and consume both responses, then Main synthesizes them with its own judgment before implementation. **POST:** after implementation, send the actual current source/diff to at least **2 independent reviewers/workers**, wait for and consume both responses, disposition every material finding, fix as needed, and rerun regression gates. USER FIRST / user-final testing starts only after the POST gate is closed. A sent-but-unread review is still open work and blocks progression.

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

### A. Task-fit gate

Choose the worker for the job rather than filling idle capacity.

- implementation vs review vs GUI verification vs architecture
- repository access needed or self-contained packet sufficient
- expected file overlap with other workers
- quota/cost sensitivity
- runtime/GUI evidence requirements
- risk if the worker over-expands scope or misses a frozen contract

If the task does not benefit from parallelism, Main does it directly.

Current routing priority:

```text
1. Main FAST PATH for tiny low-ambiguity work.
2. Luna High pool: 2 lanes normally; optional third High burst lane only for real low-overlap parallelism.
3. External Web challenger: DeepSeek / Qwen / GLM are the active comparison set. Until AWB-specific characteristics are established from enough real tasks, rotate them deliberately across PRE/POST/adjacent decision boundaries instead of favoring one provider.
4. Luna Max: one exclusive deep-reasoning lane for genuinely difficult reasoning. Drain High first; High and Max never overlap.
5. Gemini API: MANUAL OPTIONAL ONLY. Never auto-route or depend on it; use only when Main explicitly chooses expendable free capacity.
6. Antigravity: dormant capacity only; preserve integration, do not assign normal work.
7. Astra: manual high-value architecture escalation.
```

**External Web challenger invariant:** for every non-trivial AWB implementation/stabilization batch, Main sends at least one small decision-boundary review through the provider-agnostic External Web Bridge using the current frozen contract plus exact relevant source/diff; that reviewer may occupy one slot in the mandatory PRE/POST minimum of two independent reviewers. While provider traits remain provisional, distribute requests across DeepSeek / Qwen / GLM over successive gates; do not send all three for every item unless the decision is genuinely ambiguous enough to justify it. Record concrete hits, false alarms, context failures, and Main repair burden in the worker ledger so routing can become evidence-driven later. Launcher auto-open/import behavior remains the dispatch mechanism. Reviewer conclusions are advisory, but once a request is sent **Main must consume and disposition it before that gate closes**. If a bridge/provider is unavailable, record that explicitly and fill the missing reviewer slot with another independent reviewer rather than lowering the minimum.

**Luna concurrency is a fixed orchestration invariant:** normal operation may use **two High workers in parallel**; when a third genuinely independent/low-overlap slice exists, Main may use **three High workers in parallel** via burst. **Max is one worker only and must run alone with respect to Luna**: all High lanes must be drained before Max starts, and no High lane may launch while Max is active.

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
- `WORKER_PROFILES/ANTIGRAVITY.md`
- `WORKER_PROFILES/GEMINI.md`
- `WORKER_PROFILES/DEEPSEEK_WEB.md`
- `WORKER_PROFILES/ASTRA.md`

## 8. Update discipline

Worker profiles are living operational documents.

- use dated observations,
- separate `Observed` from `Provisional`,
- remove stale compensation rules when later evidence disproves them,
- avoid vague labels such as "smart" or "bad at code",
- describe actionable behavior: scope drift, evidence discipline, test bias, patch size, context sensitivity, runtime/tool reliability, etc.

The goal is not to characterize models socially. The goal is to make the **next request measurably better**.
