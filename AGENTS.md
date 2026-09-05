Purpose

This repository is being refactored on the base_refac branch so it can be maintained safely by humans and Codex.

The immediate goal is architectural cleanup without accidental behavior changes. Prefer small, reversible, test-backed changes that make the active runtime easier to understand, package, test, document, and evolve.

Read ARCHITECTURE.md before making structural changes.

Current production path

Treat this as the canonical browser-agent execution path unless repository evidence proves otherwise:

agent.sh
  -> run_v16.py run
  -> brain_graph.run_brain(...)
  -> LangGraph nodes/workers
  -> Overwatch verification
  -> browser side effects

Important distinctions:

brain_graph.py is the current LangGraph orchestration spine.

BrainState in `agent_first_browse.agent.state` is the primary typed runtime state; root `brain_state.py` is a compatibility shim during migration.

workers/base_worker.py contains the current specialist worker decision path.

advanced_agent.py is the legacy monolith, but it is not dead code. The current CLI and other modules still import selected utilities from it.

orchestrator/ is an older CEO/Spawner/Executor architecture. Do not assume it is active as a whole. orchestrator/critic_v12.py is still referenced by active code.

python-orchestrator/app/ is a separately packaged application tree that still supplies active logging/browser utilities and contains the browser-promoter graph.

Root skills/ contains runtime browser action classes. It is unrelated to Codex skills under .agents/skills/.

Before editing a module that appears legacy, search its active importers and call sites.

Refactor objective

Converge the repository toward one installable package under src/agent_first_browse/, one deterministic test hierarchy, and one authoritative documentation hierarchy.

The target structure and migration order are defined in ARCHITECTURE.md.

Do not attempt the target layout in one change.

Working rules

Trace before editing. Identify the active entry point, callers, state ownership, and nearest relevant tests before changing behavior or moving code.

One concern per change. Do not mix file moves, large decompositions, behavior changes, prompt changes, dependency upgrades, and cost optimizations in the same patch.

Move before decomposing. When migrating a large module, first move it with behavior preserved and compatibility maintained. Split responsibilities only in later changes.

Preserve public imports during migration. Use temporary compatibility shims when they materially reduce breakage. Remove a shim only after all active callers have migrated and tests pass.

Prefer deletion after proof, not suspicion. A file is removable only after active imports/callers are gone and relevant regression coverage exists.

Keep diffs narrow. Do not reformat or rename unrelated code while solving another problem.

Do not silently change defaults. Environment-variable defaults, model order, retry budgets, timeouts, feature flags, browser behavior, and safety thresholds are behavior.

Document ownership changes. When responsibility moves between packages, update ARCHITECTURE.md in the same logical change.

Behavioral invariants

Preserve these unless the task explicitly requests a behavior change and includes tests/evaluation for it.

Decision and execution

One grounded browser action is proposed at a time.

Workers propose actions; browser side effects are executed through the established verification/execution path.

A proposed action is not treated as successful merely because a model produced it.

Verification distinguishes action failure, delayed/ambiguous verification, and provider failure before action production.

Retry/recovery loops remain bounded.

State

BrainState remains the orchestration source of truth during the migration.

Preserve the existing single-writer discipline for state fields.

Do not create a second competing state/guidance channel to work around an existing field.

Verified completed sub-goals must not be silently demoted by weaker later observations.

Perception

DOM/accessibility evidence remains the default perception source.

Resolve targets against fresh live page state before side effects.

Vision remains an escalation for genuine visual ambiguity or visual-only content, not an automatic replacement for available DOM/a11y grounding.

Verification and side-effect safety

Preserve Overwatch/action-verification semantics while moving code.

Preserve intent-journal/idempotency protection around uncertain side effects.

Do not weaken irreversible/cautious-action checks to make tests or benchmarks faster.

CAPTCHA verification logic, where present, must retain independent read/compare behavior; generic loop suppression must not collapse required checks.

Browser/session behavior

Until covered by replacement tests and explicitly migrated, treat these areas as high-risk:

cdp_click.py click fallback/verification behavior

cdp_input.py typing and reversion checks

ghost_input.py humanized input behavior

stealth/browser launch and warm-up code under python-orchestrator/app/browser_promoter/

session persistence/manual-login behavior still reached through advanced_agent.py

Refactoring these is allowed, but not as incidental cleanup.

Model and cost invariants

Do not increase model-call fan-out as a side effect of a refactor.

Unless a task explicitly targets routing/cost behavior:

preserve run-scoped model affinity and provider ordering semantics;

preserve same-model structured-output repair before unnecessary sibling failover where implemented;

preserve bounded retry/failover budgets;

do not introduce an LLM call for logic that can remain deterministic;

do not make vision, consensus, probing, or auxiliary judges unconditional;

do not run paid/live provider calls merely to validate a structural refactor.

When changing model routing or prompts, measure behavior separately from structural cleanup.

Git safety

Assume the working tree may contain valuable user work.

Never run git reset --hard, git clean -fd, destructive checkout/restore commands, or force-push commands unless the user explicitly asks for that exact operation.

Never rewrite history or rebase interactively unless explicitly requested.

Do not switch branches, merge branches, create/delete branches, commit, tag, or push unless the task explicitly asks for it.

Never discard unrelated working-tree changes.

Before broad edits, inspect git status and distinguish pre-existing changes from your own.

If the working tree is unexpectedly dirty in files you need to edit, preserve the changes and work around them; do not overwrite them.

Testing and validation

Tests are separated under `tests/unit/`, `tests/regression/`, `tests/integration/`, and `tests/live/`. Executable provider/browser/manual checks live under `scripts/smoke/` and are opt-in.

Therefore:

Ordinary `pytest` is the credential-free deterministic validation boundary; integration, live-provider, browser, and smoke checks must be invoked explicitly.

Before modifying behavior, run the smallest deterministic tests that cover the affected subsystem and record any pre-existing failure.

Do not run credentialed/network/live-provider tests unless the task requires them and the environment is intentionally configured for them.

When a deterministic scripts/check.sh (or equivalent) exists, treat it as the default repository validation command.

A structural move should ideally produce the same focused test results before and after the move.

New bug fixes should add or strengthen a regression test whenever practical.

Do not “fix” a test by weakening a behavioral invariant.

Preferred validation order:

focused unit/regression test
  -> affected subsystem suite
  -> repository deterministic check
  -> integration/live tests only when justified

Dependencies and packaging

pyproject.toml is the intended long-term dependency and packaging source of truth.

The current package layout is transitional and still depends on python-orchestrator/ plus root modules.

Do not add a new production dependency without a concrete need.

Prefer the standard library or an already-installed dependency for refactor tooling.

Remove sys.path hacks only as the corresponding imports become valid through the package layout; do not delete them early and break runtime startup.

Target naming

As code is migrated:

application/runtime browser action modules should move toward agent_first_browse.actions, not a top-level skills package;

reserve .agents/skills/ for Codex development skills;

orchestration code belongs under agent_first_browse.agent;

model/provider logic belongs under agent_first_browse.models;

perception, browser mechanics, verification, cognition, memory, survey, and promotion should have explicit package boundaries described in ARCHITECTURE.md.

Documentation rules

Use documentation according to authority:

AGENTS.md — concise working rules for coding agents.

ARCHITECTURE.md — current architecture, ownership, migration target, and invariants.

README.md — user-facing setup and usage.

docs/ — detailed architecture, development guides, decisions, audits, and plans.

Historical/versioned planning documents are evidence of past decisions, not automatically current truth.

When code contradicts an old plan, verify the live code path and tests before implementing something the plan says is “missing.”

Efficient Codex workflow

Use repository evidence efficiently:

prefer rg/targeted symbol searches before reading whole large modules;

inspect only the relevant slices of 1,000+ line files initially;

reuse findings instead of repeatedly rescanning the same files;

use read-only exploration/review subagents for parallel investigation when useful, but keep overlapping write ownership narrow;

avoid launching browsers, provider probes, or network calls for a source-only refactor;

put repeatable mechanical checks in scripts/CI rather than expanding this file with formatting rules.

Code review rules

When reviewing a refactor, prioritize:

changed runtime behavior that was supposed to remain unchanged;

broken or stale imports and entry points;

duplicate sources of truth/state writers;

changed retry, verification, or safety semantics;

increased model/vision/provider call fan-out;

missing regression coverage for moved responsibilities;

dead compatibility shims that can now be removed;

documentation that names the wrong canonical path.

Style-only comments are secondary unless they obscure ownership or correctness.

Definition of done for a refactor task

A refactor task is complete when:

the intended responsibility is clearer than before;

active callers use the intended interface;

behavior-changing edits are absent or explicitly identified;

focused deterministic tests pass, or pre-existing failures are clearly reported;

no unrelated files were changed;

temporary compatibility code is clearly marked;

ARCHITECTURE.md is updated if ownership or the migration state changed;

the final report lists changed files, validation commands/results, and remaining risks/follow-ups.

Codex configuration references

Current Codex supports repository instructions in root AGENTS.md, with narrower AGENTS.md/AGENTS.override.md files close to specialized subtrees when needed. Repository skills can live under .agents/skills/<skill>/SKILL.md, and custom project subagents under .codex/agents/*.toml.

Keep this root file compact. Add nested instructions only when a subtree genuinely requires different rules; do not duplicate repository-wide guidance.

## Repository-specific skills

Use the project skills when applicable:

- `runtime-trace` — before moving, rewriting, or deleting code whose active runtime status or ownership is unclear.
- `migration-slice` — for one behavior-preserving package/module/responsibility migration at a time.
- `repo-check` — to choose and run the cheapest sufficient deterministic validation for a change.

For structural migrations, prefer:

`runtime-trace -> migration-slice -> repo-check`

Use `systematic-debugging` if a new regression appears and
`verification-before-completion` before claiming the task is complete.
