# Agent First Browse development rules

Structural package migration is complete. Preserve canonical ownership and
focus current work on deterministic validation, measurement, and evaluation.
Do not recreate root implementations or historical orchestration trees.

## Canonical runtime

```
agent.sh / Start-Agent.ps1 / agent-browse
  -> agent_first_browse.cli
  -> agent_first_browse.agent.graph.run_brain(...)
  -> workers and cognition
  -> models and failover
  -> verification.Overwatch
  -> actions/browser/perception
```

The import package is `agent_first_browse`; the distribution is
`agent-first-browse`; the installed CLI is `agent-browse`.

## Current priorities

1. Preserve canonical package ownership and dependency direction.
2. Keep ordinary validation deterministic and credential-free.
3. Measure behavior before optimizing it.
4. Establish telemetry and evaluation baselines before behavioral changes.
5. Preserve Overwatch as the action commit/execution authority.
6. Preserve model provider ordering, bounded failover, and same-model
   structured-output repair before failover.

Telemetry, inference-cost optimization, worker-contract redesign, capability
routing, prompt reduction, and subagents are separate future work.

## Non-negotiable runtime invariants

- Workers propose one grounded browser action; they do not execute effects.
- Overwatch verifies proposals, commits trusted outcomes, and protects
  idempotency, retry, rollback, and completion semantics.
- BrainState is the orchestration source of truth with single-writer discipline.
- DOM/accessibility evidence is the default; vision is an ambiguity escalation.
- Verified progress is sticky and recovery loops are bounded.
- Intent-journal protection remains around uncertain side effects.
- Model failover preserves provider/model order, retry budgets, timeouts, health,
  cooldowns, and call count.
- Malformed structured output is repaired on the same model when supported;
  provider/model failover happens only after repair failure.

## Package boundaries

Application code belongs under `src/agent_first_browse/`:

- graph/state/persistence under `agent` and `persistence`;
- models under `models`;
- workers under `workers`;
- verification/cognition under their named packages;
- browser mechanics and shared warm-up/stealth under `browser`;
- browser actions under `actions`;
- perception, memory, survey, and promotion under their named packages.

Canonical packages must not import root modules, `app.*`,
`python-orchestrator`, the retired runtime `skills` package, or promotion
internals for shared functionality. Compatibility surfaces, if intentionally
retained, point inward to canonical code.

## Testing and validation

Tests are separated into `tests/unit/`, `tests/regression/`,
`tests/integration/`, and `tests/live/`. Ordinary pytest and
`./scripts/check.sh` must not require credentials, network access, a browser,
or provider services. Browser/provider/manual scripts are opt-in under
`scripts/smoke/` and related tool directories.

Use the cheapest sufficient deterministic checks first:

```
focused unit/regression test
  -> affected subsystem suite
  -> ./scripts/check.sh
  -> live/integration checks only when explicitly justified
```

Never weaken an invariant to make a test pass. Record pre-existing failures
separately from regressions.

## Safe changes

Read `ARCHITECTURE.md` and inspect active callers before changing ownership,
state, side effects, prompts, retries, browser behavior, or persistence.
Preserve public contracts and defaults. Do not mix behavior changes with
unrelated cleanup.

Use standard library or existing dependencies for tooling. Do not add
production dependencies without a concrete need. Keep documentation truthful
about the current runtime.

## Git safety

Preserve user work. Before editing, inspect `git status --short` and do not
overwrite unrelated changes. Never use destructive reset, clean, checkout, or
history-rewrite commands unless explicitly requested. Do not switch branches,
commit, tag, merge, or push unless explicitly requested.

## Development skills

Repository development skills live under `.agents/skills/`. They are not
application runtime actions. Use repository-owned checks and do not invoke live
providers, browsers, websites, or credentials for source-only changes.
