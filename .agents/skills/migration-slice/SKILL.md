---
name: migration-slice
description: Execute one small, reversible, behavior-preserving migration slice in the Agent-first-browse base_refac branch. Use when moving one module, symbol group, or responsibility toward the target src/agent_first_browse package; replacing a root import with a packaged import; extracting a still-live responsibility from advanced_agent.py, orchestrator, workers, skills, or python-orchestrator/app; or removing a temporary compatibility shim after callers have migrated. Trace active dependencies first, move before decomposing, preserve public behavior/defaults/invariants, validate with the cheapest sufficient deterministic checks, and update architecture ownership when it changes.
---

# Migration Slice

Perform one coherent structural migration while keeping the repository usable at every meaningful checkpoint.

## 1. Read the governing context

Read:

- `AGENTS.md`;
- the relevant current-ownership and target-layout sections of `ARCHITECTURE.md`.

Inspect:

```bash
git status --short
```

Preserve unrelated user changes. Do not switch branches, commit, push, reset, clean, rebase, or rewrite history unless explicitly requested.

## 2. Define one migration boundary

State the slice before editing:

```text
SOURCE:
<current module/symbol/responsibility>

DESTINATION:
<target package/module>

INTENDED BEHAVIOR CHANGE:
none

PUBLIC/COMPATIBILITY SURFACE TO PRESERVE:
<imports, CLI, state schema, function signatures, config defaults>

OUT OF SCOPE:
<decomposition, cleanup, prompt tuning, dependency upgrades, unrelated renames, etc.>
```

If the task requires multiple unrelated ownership changes, split it into separate slices.

## 3. Trace before editing

Use the `runtime-trace` skill when available, or perform the equivalent read-only analysis.

Identify:

- active entry/call path;
- importers and callers;
- `BrainState` fields read/written;
- configuration/environment dependencies;
- browser/model/network side effects;
- verification/safety boundaries;
- nearest deterministic tests;
- legacy/alternate supported modes.

Do not move or delete code whose active status remains unknown without explicitly reporting that uncertainty.

## 4. Establish a focused baseline

Use the `repo-check` skill when available, or follow its validation strategy.

Before the move, run the smallest deterministic check that exercises the public contract when feasible. Record pre-existing failures.

Do not invoke paid model providers or live browser/site tests merely to establish a baseline for a structural move.

## 5. Move before decomposing

For large modules, first relocate behavior with minimal internal changes.

Prefer this sequence:

```text
existing implementation
    -> create target package/module
    -> move/copy implementation with behavior preserved
    -> preserve old import path with a temporary shim if useful
    -> migrate active callers incrementally
    -> verify
    -> decompose responsibilities in a later slice
```

Do not combine the initial move with broad renaming, style cleanup, algorithm changes, prompt rewriting, retry tuning, provider changes, or performance optimization.

## 6. Preserve migration invariants

Unless the user explicitly requests behavioral change, preserve:

- canonical runtime startup behavior;
- `BrainState` semantics and field ownership;
- proposal -> verification -> browser-side-effect boundaries;
- Overwatch/action-verification behavior;
- idempotency/intent-journal protections;
- DOM/a11y-first perception and conditional vision escalation;
- bounded retry/recovery/failover behavior;
- provider ordering and same-model repair semantics;
- environment-variable and feature-flag defaults;
- CLI/public import compatibility where still consumed;
- model-call fan-out and cost-sensitive escalation behavior.

A refactor is not allowed to “simplify” these away incidentally.

## 7. Use compatibility shims deliberately

A temporary shim is preferred when it allows callers to migrate independently.

A shim should:

- be tiny;
- contain no new business logic;
- re-export/delegate to the new implementation;
- be clearly marked temporary;
- preserve the old import contract only as long as needed.

Do not maintain duplicate implementations.

Example pattern:

```python
"""Temporary compatibility shim during base_refac migration."""

from agent_first_browse.agent.state import *  # noqa: F401,F403
```

Use narrower explicit exports when practical.

## 8. Update imports incrementally

Migrate active production callers first or in a controlled batch justified by the slice.

Check for remaining references after edits:

```bash
rg -n "<old-module-or-symbol>" .
```

Classify remaining matches rather than blindly replacing documentation, historical plans, generated data, or compatibility tests.

## 9. Validate after the move

Use `repo-check` when available.

At minimum compare the focused pre-move and post-move evidence when possible. Then run the affected subsystem or repository deterministic check if justified by the dependency surface.

If a regression appears, stop broad cleanup and use `systematic-debugging` rather than layering speculative fixes onto the migration.

Do not claim completion without fresh verification; use `verification-before-completion` when available.

## 10. Update architecture ownership

If ownership or the authoritative import path changed, update `ARCHITECTURE.md` in the same logical slice.

Document current truth, not just the target aspiration. If a compatibility shim remains, say so.

Do not update README/user-facing docs unless the user-visible invocation/setup actually changed.

## 11. Remove old code only after proof

Delete the old implementation or shim only when:

- active callers are migrated;
- alternate supported modes have been checked;
- replacement tests pass;
- repository search finds no required old import path;
- architecture docs identify the new owner;
- deletion does not mix in another refactor concern.

For `advanced_agent.py`, old `orchestrator/`, and `python-orchestrator/`, default to **extract first, retire later**.

## 12. Completion report

Return:

```text
MIGRATION SLICE:
<source -> destination>

BEHAVIOR INTENTION:
unchanged

FILES CHANGED:
- ...

COMPATIBILITY:
<shim/public imports retained or removed, with reason>

VALIDATION:
- <command> -> PASS/FAIL

REMAINING OLD REFERENCES:
- <matches that still matter>

ARCHITECTURE DOC:
<updated / no ownership change>

DEFERRED WORK:
- <decomposition/cleanup intentionally not mixed into this slice>
```

Keep deferred improvements separate rather than opportunistically implementing them.
