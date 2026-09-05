---
name: runtime-trace
description: Trace whether a module, symbol, package, or responsibility is actually active in the Agent-first-browse runtime before moving, rewriting, extracting, or deleting it. Use during base_refac when code appears legacy or duplicated; when ownership is unclear between root modules, workers, skills, orchestrator, advanced_agent.py, or python-orchestrator/app; when identifying entry points, callers, state flow, side effects, tests, and compatibility requirements; or before declaring code safe to retire. Perform read-only repository analysis and do not infer dead code from filenames or architecture age alone.
---

# Runtime Trace

Produce repository evidence about a target before structural edits. This skill is intentionally read-only.

## 1. Anchor to the known runtime

Read `AGENTS.md` and the relevant section of `ARCHITECTURE.md`.

Unless current repository evidence proves otherwise, begin from the canonical browser-agent path:

```text
agent.sh
  -> run_v16.py run
  -> brain_graph.run_brain(...)
  -> graph nodes/workers
  -> Overwatch / verification
  -> browser side effects
```

Remember the known transitional boundaries:

- `advanced_agent.py` is legacy but still has live consumers.
- `orchestrator/` is an older architecture but `orchestrator/critic_v12.py` has active references.
- `python-orchestrator/app/` still supplies active infrastructure and a separate browser-promoter graph.
- root `skills/` means runtime browser actions, not Codex skills.

Do not classify any of these as dead solely from architectural intent.

## 2. Define the trace target

Identify the exact target:

```text
module/package:
symbol(s):
responsibility:
proposed destination or deletion, if known:
```

Avoid tracing an entire 2,000-line module when the task concerns one function or responsibility.

## 3. Find definitions and references

Prefer targeted repository searches:

```bash
rg -n "<symbol-or-import>" .
rg -n "from <module>|import <module>" .
```

Exclude generated/cache/vendor paths when necessary.

Separate references into:

- production runtime;
- CLI/startup;
- tests;
- scripts/smoke tools;
- documentation/plans;
- compatibility shims.

A test or historical document is not proof that a production path is active.

## 4. Trace from entry point to target

When the target is claimed to be active, identify the shortest credible call/import chain from an active entry point to it.

Example form:

```text
run_v16.py
  -> brain_graph.py
  -> workers/base_worker.py
  -> target_symbol()
```

If the path is conditional, record the condition:

- feature flag;
- environment variable;
- worker/route selection;
- fallback/recovery rung;
- site/domain specialization;
- manual-login mode;
- browser-promoter mode.

Do not call a path inactive just because the default happy path does not execute it.

## 5. Trace state and side effects

For active behavior, record only relevant state flow:

- `BrainState` fields read;
- `BrainState` fields written;
- other persisted/global state used;
- environment/config dependencies;
- browser/model/network/file/database side effects;
- verification or safety gates crossed.

Pay special attention to responsibilities that bypass the expected proposal -> verification -> side-effect path.

## 6. Identify tests and migration constraints

Find the nearest tests that exercise the target or its public contract.

Record:

- deterministic tests;
- integration/live tests;
- missing regression seams;
- public import paths relied on by tests or runtime;
- compatibility aliases/shims that would be needed during a move.

Do not run paid/live tests during this trace.

## 7. Classify conservatively

Use exactly one status:

- **ACTIVE_DIRECT** — reached directly by the canonical runtime/startup path.
- **ACTIVE_CONDITIONAL** — production behavior reached under a flag, route, fallback, domain, recovery path, or alternate supported mode.
- **LEGACY_WITH_LIVE_DEPENDENCY** — old architecture/file still provides behavior to active code.
- **TEST_OR_SCRIPT_ONLY** — no production path found, but tests/scripts still depend on it.
- **NO_ACTIVE_EVIDENCE** — no active caller found after targeted tracing; this is not yet proof of safe deletion.
- **UNKNOWN** — evidence is conflicting or incomplete.

Never report `SAFE_TO_DELETE=yes` merely from `NO_ACTIVE_EVIDENCE`.

## 8. Apply a strict deletion bar

A target is a deletion candidate only when all relevant conditions are demonstrated:

1. no production import/call path remains;
2. no supported CLI/alternate mode requires it;
3. replacement behavior exists where needed;
4. callers have migrated;
5. relevant deterministic regressions cover the replacement contract;
6. documentation no longer identifies it as required;
7. removing it does not remove a compatibility surface still in use.

Otherwise prefer extraction, migration, or a compatibility shim.

## 9. Return a concise trace report

Use this form:

```text
TARGET:
<module/symbol/responsibility>

STATUS:
<one classification>

ACTIVE PATH:
<entry -> ... -> target, or none found>

CONDITIONS:
<flags/routes/modes, or none>

CALLERS / IMPORTERS:
- <production>
- <tests/scripts>

STATE / SIDE EFFECTS:
- <relevant items>

NEAREST TESTS:
- <paths or none found>

MIGRATION CONSTRAINTS:
- <public imports, shims, invariants>

SAFE TO MOVE:
<yes/no/conditional + reason>

SAFE TO DELETE:
<yes/no + evidence still required>
```

Do not modify files while performing this skill unless the user separately asks for the subsequent migration.
