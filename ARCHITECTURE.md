# Architecture

This document describes the current Agent First Browse runtime. Structural
package migration is complete; future changes preserve these ownership
boundaries and invariants rather than recreating historical architectures.

## Product identity

- Distribution: `agent-first-browse`
- Import package: `agent_first_browse`
- CLI: `agent-browse`

## Runtime path

```
agent.sh / Start-Agent.ps1 / agent-browse
    -> agent_first_browse.cli
    -> agent_first_browse.agent.graph.run_brain
    -> LangGraph nodes and workers
    -> verification.Overwatch
    -> actions and browser runtime
    -> fresh perception and state update
```

## Ownership

### Agent, state, and persistence

`agent_first_browse.agent.graph` owns graph construction, node orchestration,
conditional edges, bounded recovery, and completion handling.
`agent_first_browse.agent.state` owns typed orchestration state.
`agent_first_browse.persistence` owns checkpoint retention and state
persistence.

### Workers

`agent_first_browse.workers` owns worker contracts, prompt construction,
deterministic paths, model-backed decisions, escalation, and the worker
façade. Workers propose one grounded action at a time; they do not execute
browser effects.

### Models

The public model subsystem is `agent_first_browse.models`:

- `schemas.py` — model contracts, including `ModelClient`;
- `providers.py` — provider adapters and client/pipeline construction;
- `health.py` — health, cooldown, quota, probe freshness, and persistence;
- `routing.py` — deterministic role/tier selection and ordering;
- `probes.py` — startup capability probing and pruning;
- `failover.py` — ordinary invocation, budgets, retry/failover, and recovery;
- `registry.py` — stable public façade and coordination.

The structured-output invariant is:

```
malformed structured output
    -> same-model repair when supported
    -> provider/model failover only if repair fails
```

Provider ordering, model-call count, retry budgets, timeouts, health state, and
cooldown semantics are part of this subsystem's behavior.

### Perception, browser, actions

`agent_first_browse.perception` owns DOM, accessibility, diff, engine, and
vision evidence. `agent_first_browse.browser` owns browser launch/session
lifecycle, CDP and Playwright integration, stealth, warm-up, input, overlays,
platform helpers, and site customization. `agent_first_browse.actions` owns
the browser-action façade. Warm-up and stealth are shared browser concerns;
promotion consumes them and does not own them.

### Verification and cognition

`agent_first_browse.verification` owns action feedback and safety, action and
outcome verification, progress criticism, and Overwatch. Overwatch is the final
action commit authority and protects intent-journal, idempotency, verification,
retry, rollback, and completion semantics.

`agent_first_browse.cognition` owns deterministic locks/clarity/stagnation
and model-backed reasoning, consensus, reality reconciliation, PRM, WebDreamer,
and content critique. Cognition proposes or evaluates; it does not execute
browser actions.

### Domain packages and infrastructure

`agent_first_browse.survey` owns survey context, profiles, recipes, audio,
outcomes, quirks, benchmarks, and package data. `agent_first_browse.memory`
owns campaign, skill, intent-journal, and content memory. Shared logging is
owned by `agent_first_browse.logging`. `agent_first_browse.promotion`
owns the separate promotion/browser-promoter graph, state, integrations,
database, and promotion-only observability.

## Promotion boundary

Promotion is a separate domain graph. It uses canonical browser helpers and
the model registry where that is already an established contract, but its
promotion-specific inference paths remain isolated:

- `promotion/browser_promoter/worker_planner.py` invokes registry-provided
  clients with promotion-specific prompts, structured parsing, and retry
  behavior for reasoning and vision agents;
- `promotion/browser_promoter/supervisor_subgraph.py` constructs its own
  `ChatOpenAI` instances and invokes its supervisor chain;
- promotion graph `.ainvoke()` calls execute the promotion graph, not the
  primary browser-agent graph.

These paths are not imported by the primary runtime. Unifying them would be a
behavioral inference change and is deferred until it has its own evaluation.

## Dependency direction

```
cli -> agent graph/state
          |       \\
          v        v
       workers  cognition/models/domain
          |        |
          +--> proposal
                    |
                    v
           verification / Overwatch
                    |
                    v
             actions / browser
                    |
                    v
              perception -> state
```

Canonical core packages must not import root modules, retired namespaces,
`app.*`, `python-orchestrator`, the historical runtime `skills` package,
or promotion internals for shared functionality. Intentional compatibility
surfaces point inward to canonical packages.

## Invariants

- BrainState remains the orchestration source of truth with single-writer
  discipline.
- Workers propose actions; Overwatch verifies and commits them; actions/browser
  modules perform effects; fresh perception records outcomes.
- DOM/accessibility evidence is the default; vision is an ambiguity escalation.
- Verified progress is sticky and recovery loops remain bounded.
- Intent-journal and duplicate-action protection remain around uncertain effects.
- Same-model structured repair precedes unnecessary sibling/provider failover.
- Structural changes do not increase model/provider call fan-out or weaken safety.

## Validation and layout

The canonical local and CI command is `./scripts/check.sh`. It runs Ruff,
shell syntax validation, repository naming/architecture checks, and the
credential-free unit/regression suite. Browser, provider, network, and
credentialed checks are opt-in under `tests/integration`, `tests/live`, and
`scripts/smoke`.

`pyproject.toml` is the packaging source of truth; source discovery is rooted
at `src/`, and the installed entry point is
`agent-browse = agent_first_browse.cli:main`.

```
src/agent_first_browse/  canonical runtime
tests/unit/              deterministic isolated tests
tests/regression/        deterministic invariants
tests/integration/       local multi-module or browser-boundary tests
tests/live/              explicit provider/browser/network tests
scripts/                 validation, diagnostics, manual, and smoke tools
docs/                    development, audits, plans, and decisions
examples/                example objectives, prompts, and configuration
```

The repository root contains project metadata, canonical documentation,
supported shell/PowerShell launchers, and external-tool files only. No second
application package or retired orchestration tree is supported.
