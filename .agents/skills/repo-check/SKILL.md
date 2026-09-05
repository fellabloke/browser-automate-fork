---
name: repo-check
description: Select and run the cheapest sufficient validation for changes in the Agent-first-browse base_refac repository. Use when checking a patch, refactor, migration, bug fix, config change, or documentation change before completion; when deciding which tests/lint/import checks are necessary; or when comparing behavior before and after a structural move. Prefer deterministic local checks and avoid credentialed, browser-network, or paid model/provider tests unless the task explicitly requires them.
---

# Repo Check

Validate changes with the smallest reliable evidence set. Do not equate “more tests” with “better verification”; choose checks based on the affected behavior and risk.

## 1. Establish repository state

Read `AGENTS.md` testing rules first. Read the relevant `ARCHITECTURE.md` section when the change affects ownership, imports, runtime flow, or package boundaries.

Inspect the working tree without modifying it:

```bash
git status --short
git diff --stat
git diff --name-only
```

Distinguish pre-existing changes from changes made for the current task. Never discard unrelated user work.

## 2. Classify the change

Classify the patch into the narrowest applicable category:

- **Docs-only:** Markdown/comments with no executable/configuration effect.
- **Structural:** file/package moves, import changes, compatibility shims, symbol relocation, packaging changes intended to preserve behavior.
- **Behavioral:** changes to runtime logic, state transitions, routing, verification, browser behavior, prompts, retries, defaults, or output.
- **Bug fix:** behavior intentionally corrected; should normally have a regression test.
- **Configuration/dependency:** `pyproject.toml`, environment defaults, CI, packaging, feature flags, dependencies.
- **High-risk integration:** browser launch/input/click mechanics, provider/model routing, paid APIs, live sites, session persistence, CAPTCHA or irreversible-side-effect safeguards.

If multiple categories apply, validate the highest-risk behavior actually changed. Do not broaden scope merely because unrelated files are dirty.

## 3. Select the cheapest sufficient checks

Use this order and stop when the evidence is sufficient for the task:

```text
syntax/import/static check
    -> focused unit or regression test
    -> affected subsystem suite
    -> repository deterministic check
    -> integration/live validation only when justified
```

### Docs-only

- Verify changed paths/references exist when practical.
- Do not run the full Python test suite solely for prose changes.

### Structural move or import migration

Prefer:

1. import/collection checks for the moved symbols;
2. the nearest existing deterministic tests;
3. the same focused checks before and after when a baseline can still be obtained;
4. the repository deterministic check if one exists.

Do not use a live provider or browser session to prove a pure file move unless no deterministic seam exists and the user explicitly authorizes it.

### Behavioral change or bug fix

Prefer:

1. reproduce or add a focused regression;
2. run that regression;
3. run tests for the affected subsystem;
4. run the repository deterministic check.

Do not weaken assertions merely to make a changed implementation pass.

### Configuration/dependency change

Validate the exact surface affected: parsing, installation metadata, imports, CLI startup, lint/type configuration, or CI command. Avoid unrelated live tests.

### High-risk integration change

First exhaust mocks, deterministic regression tests, recorded fixtures, and local import/startup checks. Run live/browser/provider validation only when the task explicitly requires it and the environment is intentionally configured for it.

## 4. Prefer repository-owned commands

If `./scripts/check.sh` exists and is documented as deterministic, use it as the default broad validation command after focused checks.

If it does not exist yet, do **not** pretend bare `pytest` is authoritative. This repository has historically mixed deterministic tests with script-style/live tests. Inspect test scope before invoking broad collection.

Prefer targeted commands such as:

```bash
pytest path/to/test_file.py -q
pytest path/to/test_file.py::test_name -q
python -m compileall <affected-package-or-files>
ruff check <affected-paths>
```

Use only tools actually configured/available in the repository.

## 5. Handle baseline failures correctly

When a check fails:

1. determine whether the failure existed before the current patch when feasible;
2. separate collection/environment failures from behavioral failures;
3. do not claim the patch introduced a pre-existing failure without evidence;
4. do not fix unrelated failures as incidental cleanup;
5. use `systematic-debugging` for a new unexplained regression.

## 6. Protect cost and side effects

Never invoke paid model/provider calls, real-site automation, destructive browser actions, or credentialed integration tests merely to increase confidence in a structural refactor.

For model-routing, prompt, vision, consensus, retry, or failover changes, verification must distinguish structural correctness from cost/behavior evaluation. Do not claim a cost improvement without measurement.

## 7. Report evidence, not confidence language

At completion, report:

```text
CHANGE CLASS:
<docs / structural / behavioral / bug fix / config / high-risk>

CHECKS RUN:
- <command> -> PASS/FAIL
- <command> -> PASS/FAIL

NOT RUN:
- <check> -> <why it was unnecessary, unavailable, or requires live credentials>

PRE-EXISTING FAILURES:
- <failure or none observed>

VERDICT:
<what the evidence proves, and any remaining validation gap>
```

Do not say “all good”, “fully tested”, or “safe” unless the executed checks actually support that scope.
