Architecture and Refactor Map

Status

This document is the architectural source of truth for the base_refac migration.

It describes:

what the repository actually runs today;

which apparently legacy components are still dependencies;

the invariants that must survive the refactor;

the target package structure;

the safe migration order from the current hybrid layout to that target.

This is a migration architecture, not an instruction to rewrite the repository at once.

When code and an older versioned plan disagree, trace the live entry point, imports, and tests. Update this document when verified ownership changes.

1. Architectural goals

The refactor should make the system:

easier for Codex and humans to map correctly;

safer to change through deterministic regression tests;

installable as one coherent Python package;

explicit about state ownership and side-effect boundaries;

capable of evolving workers/models without duplicating orchestration logic;

measurable for latency, provider calls, retries, and cost;

less dependent on root-level modules, sys.path mutation, and historical package boundaries.

The migration should not change browser-agent behavior simply to obtain a cleaner directory tree.

1. Current canonical runtime

The primary user-facing runtime is:

agent.sh
    |
    v
agent_first_browse.cli run <objective>
    |
    v
agent_first_browse.agent.graph.run_brain(...)
    |
    v
LangGraph StateGraph

The installed `agent-browse` command and root launchers use the canonical CLI;
the historical `run_v16.py` launcher is retired.

The current graph is approximately:

START
  |
  v
goal_compiler
  |
  v
planner
  |
  v
perceive
  |
  v
router
  |
  +-----------------+-----------------+
  |                 |                 |
  v                 v                 v
navigator        interactor        extractor
  |                 |                 |
  +-----------------+-----------------+
                    |
                    v
                 overwatch
                    |
         +----------+-----------+
         |          |           |
         v          v           v
      commit      retry      rollback/recovery
         |          |           |
         |          +-----+-----+
         |                |
         +------------> perceive
         |
         v
      done_check
         |
         v
      finalize
         |
         v
        END

Exact edges are defined in `agent_first_browse.agent.graph`; this diagram is conceptual and should remain high level.

Current major runtime components

agent_first_browse.cli     CLI / startup
agent_first_browse.agent.graph orchestration graph
agent_first_browse.agent.state typed global graph state
agent_first_browse.agent.routing worker/verdict routing
agent_first_browse.config.feature_flags runtime feature switches
agent_first_browse.persistence.checkpoint_retention checkpoint pruning
agent_first_browse.perception.* DOM, accessibility, diff, engine, and vision evidence
agent_first_browse.browser.* CDP, humanized input, overlays, display, and platform helpers
agent_first_browse.survey.* survey context, profile, recipes, audio, outcomes, quirks, and benchmarks
agent_first_browse.memory.* campaign, skill, intent, and content memory
agent_first_browse.promotion.* browser-promoter graph, state, nodes, supervisor, integrations, database, and observability
agent_first_browse.logging    shared runtime logger
agent_first_browse.verification.* action safety, verification, outcomes, progress, and Overwatch
agent_first_browse.cognition.* deterministic and model-backed cognition
agent_first_browse.actions.tools canonical browser-action façade
agent_first_browse.workers.base specialist worker façade
agent_first_browse.models.registry public model façade and coordination
agent_first_browse.models.health health/cooldown/probe state and persistence used by the registry façade
agent_first_browse.models.providers provider adapters and model/pipeline construction
agent_first_browse.models.routing deterministic model/role selection and ordering
agent_first_browse.models.probes startup/capability probing and probe-result pruning
agent_first_browse.models.failover ordinary inference, retry/failover, and structured recovery
agent_first_browse.workers.base specialist worker decision path

1. Retired legacy architectures

Wave 4 retired the old `orchestrator/`, root runtime `skills/`, and
`python-orchestrator/` trees after active callers were migrated to canonical
package owners. The autonomous implementation in `advanced_agent.py` was
drained and the historical façade was removed after repository callers migrated.

The repository root contains no application Python modules. Canonical
production code imports only `agent_first_browse.*`; optional diagnostics and
smoke tools live under `scripts/`.

1. Runtime invariants

The refactor must preserve the following architectural properties unless a dedicated behavior-changing task explicitly supersedes them.

4.1 Typed orchestration state

BrainState is the current orchestration source of truth.

Rules:

each state field should have one authoritative writer;

readers should consume the state rather than create competing hidden state channels;

bounded history/checkpoint data must remain bounded;

state-schema migrations require explicit regression tests and checkpoint compatibility consideration.

4.2 Proposal -> verification -> side effect/result

The architecture must retain a clear boundary between model reasoning and trusted browser effects.

Conceptually:

fresh evidence
    -> worker proposes ActionProposal/ProposedAction
    -> verifier / Overwatch checks the proposal and context
    -> executor performs the browser effect
    -> live result is observed
    -> state/history is committed, retried, rolled back, or recovered

Workers should not gain ad-hoc direct browser side effects during cleanup.

4.3 DOM/a11y first, vision on ambiguity

The normal evidence path is structural page evidence first.

Vision is appropriate for visual-only or genuinely ambiguous state; it should not become the default because a refactor made DOM interfaces inconvenient.

4.4 Sticky verified progress

Verified completion/progress should not be silently invalidated by a later weaker observation.

4.5 Bounded recovery

All retry, failover, perception-escalation, and recovery cycles need explicit limits and state identity. No refactor should introduce an unbounded “try again” path.

4.6 Side-effect idempotency / hesitation

Intent-journal and duplicate-action protection exist because a browser effect can occur even when transport/verification reports uncertainty.

This protection is architectural, not optional cleanup.

4.7 Cost-aware inference

Model intelligence should be spent on uncertainty rather than deterministic bookkeeping.

Preserve or improve this ordering:

deterministic resolution
    -> primary semantic model
    -> same-model repair where appropriate
    -> provider/model fallback on classified failure
    -> vision/consensus/simulation only when their trigger is justified

Do not make multiple escalation mechanisms independently fire for the same uncertainty without an explicit reason.

1. Current ownership map

This table describes current ownership, not final filenames.

Concern

Current primary location

Migration destination

CLI/startup

agent.sh, Start-Agent.ps1, and the installed `agent-browse` command

agent_first_browse.cli + wrappers

Graph orchestration

`agent_first_browse.agent.graph`

agent_first_browse.agent.graph

Typed graph state

`agent_first_browse.agent.state`

agent_first_browse.agent.state

Graph routing

`agent_first_browse.agent.routing.py`

agent_first_browse.agent.routing

Checkpoint retention

`agent_first_browse.persistence.checkpoint_retention.py`

agent_first_browse.persistence.checkpoint_retention

Model/provider layer

src/agent_first_browse/models/registry.py

agent_first_browse.models

Worker decisions

`agent_first_browse.workers.base.py`

agent_first_browse.workers

Browser action facade

`agent_first_browse.actions.tools`

agent_first_browse.browser / actions / perception

DOM parsing

dom_parser.py, a11y_parser.py, dom_diff.py

agent_first_browse.perception

Adaptive perception

perception_engine.py, vision_consult.py

agent_first_browse.perception

Click/type mechanics

cdp_click.py, cdp_input.py, ghost_input.py

agent_first_browse.browser

Verification

agent_first_browse.verification.action, feedback, safety, engine, outcome,
progress, and overwatch

agent_first_browse.verification

Progress critic

agent_first_browse.verification.progress (legacy orchestrator critic retired)

agent_first_browse.verification.progress

Cognition/guidance

agent_first_browse.cognition.core, reasoning, consensus, reality, prm, dreamer,
content_critic, and deterministic lock/clarity/stagnation/action primitives

agent_first_browse.cognition

Memory

campaign_memory.py, skill_memory.py, intent_journal.py, content_store.py

agent_first_browse.memory

Survey domain

survey_*.py, survey JSON/config

agent_first_browse.survey

Browser promoter

src/agent_first_browse/promotion/browser_promoter/ (compatibility shims remain under app/)

agent_first_browse.promotion + shared browser modules

Shared app infrastructure

src/agent_first_browse/logging.py; promotion config, observability, and pacing

package root/shared infrastructure

Runtime action classes

historical root skills/ (retired; Codex skills remain under .agents/skills/)

agent_first_browse.actions

The destination column is directional. Exact splits should be justified by dependency boundaries and tests rather than by this table alone.

1. Target repository structure

The intended end state is approximately:

Agent-first-browse/
|
|-- AGENTS.md
|-- ARCHITECTURE.md
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- .env.example
|-- .gitignore
|
|-- agent.sh
|-- Start-Agent.ps1
|
|-- .github/
|   `-- workflows/
|
|-- .agents/
|`-- skills/                  # Codex skills only
|
|-- .codex/
|   |-- config.toml
|   `-- agents/                  # Codex subagent definitions
|
|-- docs/
|   |-- architecture/
|   |-- development/
|   |-- decisions/
|   |-- plans/
|`-- audits/
|
|-- examples/
|   |-- objectives/
|   |-- prompts/
|   `-- config/
|
|-- scripts/
|   |-- check.sh
|   |-- diagnostics/
|`-- smoke/
|
|-- src/
|   `-- agent_first_browse/
|       |-- __init__.py
|       |-- cli.py
|       |-- config/
|       |   |-- __init__.py
|       |   `-- feature_flags.py
|       |-- logging.py
|       |
|       |-- agent/
|       |   |-- graph.py
|       |   |-- state.py
|       |   |-- routing.py
|       |`-- lifecycle.py
|       |
|       |-- persistence/
|       |   |-- __init__.py
|       |   `-- checkpoint_retention.py
|       |
|       |-- models/
|       |   |-- registry.py
|       |   |-- routing.py
|       |   |-- health.py
|       |   |-- probes.py
|       |   |-- failover.py
|       |   |-- providers.py
|       |   `-- schemas.py
|       |
|       |-- browser/
|       |   |-- runtime.py
|       |   |-- warmup.py
|       |   |-- tabs.py
|       |   |-- overlays.py
|       |   |-- platform.py
|       |   |-- proxy.py
|       |   |-- virtual_display.py
|       |   |-- cdp/
|       |`-- input/
|       |
|       |-- perception/
|       |-- actions/              # application runtime actions, not Codex skills
|       |-- verification/
|       |-- cognition/
|       |-- workers/
|       |-- memory/
|       |-- survey/
|       `-- promotion/
|
`-- tests/
    |-- conftest.py
    |-- unit/
    |-- regression/
    |-- integration/
    `-- live/

This structure is deliberately conventional: source code has one import root, tests have explicit cost/environment classes, and repository-agent configuration is visually distinct from application runtime code.

1. Target dependency direction

The target should reduce cross-package cycles and make the costly/side-effecting parts obvious.

A useful dependency direction is:

cli
 |
 v
agent graph/state/routing
 |
 +------------------+------------------+------------------+
 |                  |                  |                  |
 v                  v                  v                  v
workers          cognition          models           domain logic
 |                                                     survey/promotion
 |
 v
action proposal
 |
 v
verification
 |
 v
browser/actions
 |
 v
perception / observed result
 |
 +---------------------------> state

Boundary rules

models should not know about Playwright pages.

perception should return structured evidence rather than mutate orchestration policy ad hoc.

workers should consume context and return proposals, not own unverified browser effects.

verification is the trust boundary before/after effects.

browser owns low-level browser/session/input mechanics.

domain packages such as survey should compose core capabilities rather than fork a second generic agent architecture.

promotion should reuse shared browser/model infrastructure where appropriate without becoming the owner of the primary v16 graph.

1. Migration strategy

Rule: move first, decompose second

For large modules, use two separate phases.

Example:

model_registry.py
    -> src/agent_first_browse/models/registry.py   # behavior-preserving move
    -> compatibility shim if required
    -> migrate callers/tests
    -> validate

later:

models/registry.py
    -> registry.py + health.py + failover.py + routing.py + providers.py

The first responsibility extraction is complete: `models/health.py` owns the
existing `ProviderHealthTracker` implementation, including identity aliases,
cooldowns, probe freshness, quota/accounting state, schema blacklist state, and
JSON persistence. `models/registry.py` remains the public façade and coordinates
provider construction, probing, and invocation through the extracted owners.
The façade continues to re-export `ProviderHealthTracker` for compatibility.

The provider construction extraction is complete: `models/providers.py` owns
provider SDK adapters, credential discovery/fingerprints, and free/premium
text, vision, and audio pipeline builders. `models/registry.py` continues to
re-export those construction symbols for compatibility, while retaining
routing and façade coordination.
The Cloudflare adapter resolves response parsing and error-compaction helpers
from `models.failover` through a narrow lazy contract so structured-output
repair behavior remains unchanged without making providers depend on the
registry façade.

The deterministic routing extraction is complete: `models/routing.py` owns
model-tier resolution, worker and auxiliary chain shaping, worker priority, and
health-aware candidate ordering. `models/registry.py` re-exports the routing
symbols for compatibility while retaining façade coordination and inference
failover/recovery. The failover implementation now lives in
`models/failover.py`.

The startup probe extraction is complete: `models/probes.py` owns representative
selection, capability probing, JSON-mode probe rescue, probe-specific failure
classification, latency/health seeding, and dead/incapable candidate pruning.
Capability gating is probe-result interpretation rather than general routing
policy, so its safety-floor logic lives with `models/probes.py`. The registry
continues to own the public façade and probe idempotence; probe JSON rescue and
normal inference share the narrow executor owned by `models/failover.py`.

The ordinary inference extraction is complete: `models/failover.py` owns
single-client invocation, timeout and budget enforcement, retry/failover
sequencing, error classification, health bookkeeping, circuit-breaker updates,
and structured-output recovery. `models/registry.py` re-exports the failover
surface and retains façade coordination and startup probe delegation for
compatibility.

The model package now exposes an explicit package-level API from
`models/__init__.py` for `ModelRegistry`, `ModelClient`, health tracking,
provider adapters, deterministic routing, circuit breaking, and ordinary
invocation. Active runtime callers use the package API directly.

Shared runtime logging is owned by `agent_first_browse.logging`; promotion-only
observability and call pacing remain under `agent_first_browse.promotion`.

The canonical browser-promoter implementation is owned by
`agent_first_browse.promotion.browser_promoter` and uses relative package
imports plus the canonical logger. The historical `python-orchestrator` tree
has been retired; callers must use the canonical package.

The worker implementation is owned by `src/agent_first_browse/workers/base.py`.
Further worker decomposition is deferred until its prompt, deterministic fast
paths, and escalation contracts have dedicated boundaries.

Do the same for brain_graph.py, mcp_tools.py, and other large files.

Compatibility shims

A temporary root file may re-export the new implementation:

"""Temporary compatibility shim; remove after callers migrate."""
from agent_first_browse.agent.state import *

The former root compatibility surfaces were removed after repository
callers and tests migrated. New compatibility files require an explicit
external support contract.

A shim is a migration tool, not a permanent second API.

Track it and remove it once:

active callers are migrated;

imports are package-correct;

deterministic tests are green.

Recommended migration phases

Phase 0 — Refactor harness

Before broad source movement:

add/maintain AGENTS.md and this document;

classify tests into deterministic vs integration/live;

establish a reliable deterministic validation command;

align CI with the repository's declared tooling;

record pre-existing failures rather than attributing them to the refactor.

Phase 1 — Establish the package root

Create:

src/agent_first_browse/

Move low-risk shared infrastructure first where dependencies allow it, such as configuration/logging/observability helpers.

Update pyproject.toml only when the package can actually be imported and tested through the new layout.

Phase 2 — Low-level, relatively independent modules

Move foundational modules with minimal behavior change:

feature flags;

typed state helpers;

platform/virtual-display/proxy helpers;

small cognition/perception primitives with clear callers.

Phase 3 — Browser and perception boundaries

Migrate browser runtime/input/perception modules while preserving their existing behavioral semantics.

Avoid rewriting click/type/stealth behavior during a directory move.

Phase 4 — Verification and memory

Migrate verification, progress critic, intent journal, and memory modules.

This phase should make the side-effect trust boundary clearer, not weaker.

Phase 5 — Survey and promotion domains

Move survey modules and browser-promoter modules into explicit domain packages while extracting genuinely shared browser/model infrastructure.

Keep domain-specific behavior out of the generic core unless more than one domain actually uses it.

Phase 6 — Model layer

The behavior-preserving move is complete: the authoritative implementation is
`src/agent_first_browse/models/registry.py`; the historical root alias was
removed after legacy callers and tests migrated.

The first decomposition boundary is complete: health/cooldown/probe state now
lives in `src/agent_first_browse/models/health.py`, while the registry façade
continues to own coordination. Provider/model construction lives in
`src/agent_first_browse/models/providers.py`, and deterministic model and role
routing lives in `src/agent_first_browse/models/routing.py`. Startup and
capability probing now live in `src/agent_first_browse/models/probes.py`; the
ordinary inference and structured recovery now live in
`src/agent_first_browse/models/failover.py`, while the registry keeps the façade.
Further extraction should remain incremental and test-backed.

Integration Wave 2 is complete: verification is canonically owned by
`agent_first_browse.verification`, including action safety, feedback, engine and
outcome checks, the progress critic, and intact Overwatch coordination. The
historical root verification aliases and `orchestrator/critic_v12.py` were
removed after callers migrated.

Cognition is canonically owned by `agent_first_browse.cognition`. Deterministic
target/subgoal locks, clarity, stagnation, and action classification live beside
the intact strategic/core, consensus, reality, PRM, WebDreamer, and content
critique modules. The historical root cognition aliases were removed after
callers migrated.

The former `mcp_tools.py` implementation is now owned intact by
`agent_first_browse.actions.tools`; its root alias was removed after callers
migrated.
Overwatch imports canonical actions, cognition, verification, memory, survey,
browser, perception, and model dependencies. Workers still propose actions and
Overwatch remains the sole commit/execution authority.

Phase 7 — Workers

Move workers/base_worker.py intact first.

Later extract prompt composition, deterministic resolvers, planning/escalation, and shared worker contracts where tests justify the split.

Phase 8 — Graph and CLI

Move:

brain_graph.py -> agent/graph.py
agent-browse  -> cli.py

At this point launchers should converge on one installed CLI entry point while retaining shell/PowerShell convenience wrappers.

Phase 9 — Legacy retirement (complete)

The final active responsibilities were drained from:

advanced_agent.py;

old orchestrator/;

historical python-orchestrator/ package boundaries;

temporary root compatibility modules.

The old orchestrator and `python-orchestrator` trees are removed. The only
root launchers are the documented shell and PowerShell convenience wrappers.

Phase 10 — Decomposition and optimization

Only after the package migration is stable:

split oversized modules;

reduce prompt/context duplication;

optimize model-call and vision escalation costs;

introduce stronger worker capability contracts;

add Codex subagents/skills for repeatable development workflows.

Architecture cleanup and model-quality/cost experiments should normally be separate commits/evaluations.

1. Testing architecture

The final test hierarchy should communicate environment and cost clearly:

tests/
|-- unit/          # isolated deterministic logic
|-- regression/    # previously broken behaviors/invariants
|-- integration/   # multi-module/local integration
`-- live/          # browser/network/provider credentials required

Default developer/Codex loop

The default repository check should eventually be credential-free and deterministic:

format/lint check
    -> unit tests
    -> regression tests
    -> selected local integration tests

Live/provider/browser tests must be opt-in.

Regression priorities

The refactor harness should preserve coverage around:

canonical graph startup and routing;

model failover and structured-output repair;

action idempotency / duplicate click prevention;

navigation/recovery budgets;

context lifecycle and bounded memory;

perception and vision escalation;

verification/Overwatch outcomes;

survey handoff/provider lifecycle;

session/browser handoff behavior.

A file move should not require weakening these expectations.

 1. Packaging and CLI target

The target pyproject.toml should discover packages from src/ and expose a console entry point, conceptually:

[project.scripts]
agent-browse = "agent_first_browse.cli:main"

Then all launch forms converge on the same code path:

agent.sh -----------+
                    |
Start-Agent.ps1 ----+--> agent_first_browse.cli
                    |
agent-browse -------+

The shell scripts remain convenience wrappers; they should not implement an alternate orchestration architecture.

 1. Future worker architecture

After the behavior-preserving refactor, worker flexibility should move toward explicit contracts rather than model-specific classes.

Conceptually:

TaskContext
    -> WorkerPolicy
    -> ActionProposal
    -> Verification
    -> Execution
    -> ObservedResult

A future worker manifest may describe:

name
capabilities
supported action types
required modalities
cost class
latency class
risk class
preferred model class

The router can then choose capabilities based on task requirements instead of hard-coding model/provider identity into worker ownership.

This is a post-refactor direction, not a requirement to introduce a new abstraction during package migration.

 1. Model/cost architecture direction

The long-term efficiency principle is:

Spend model intelligence only on genuine uncertainty. Keep routing mechanics, state bookkeeping, retries, validation, and deterministic page operations in code where possible.

Measure optimizations by successful outcomes, not only by raw call count.

Useful metrics include:

successful task rate
calls per successful task
input/output tokens per successful task
provider retries and failovers
vision calls
consensus/judge/planner calls
p50/p95 inference latency
wall-clock runtime
estimated model cost per successful task
unchanged-state repeated requests

Do not optimize these during structural moves unless a task explicitly combines the work and supplies an evaluation plan.

 1. Documentation authority

Use the following hierarchy:

AGENTS.md
    operational rules for coding agents

ARCHITECTURE.md
    current architecture + target + migration status

README.md
    user-facing project setup, configuration, and usage

docs/architecture/
    deeper subsystem designs

docs/decisions/
    durable architectural decisions / ADRs

docs/plans/active/
    current execution plans

docs/plans/completed/
    historical implementation plans

Versioned files such as V29_OVERHAUL.md are valuable history but should not permanently compete with current architecture documentation.

 1. Known migration hazards

These are known sources of ambiguity that Codex should check before making broad changes:

The active v16 runtime is packaged under `src/agent_first_browse/`; root
launchers are convenience entrypoints and no root compatibility modules remain.

`advanced_agent.py` and the old orchestrator and `python-orchestrator` package
trees have been removed.

The browser-promoter graph is a separate graph and should not be mistaken for the v16 orchestration spine.

Root skills/ means application actions, while future .agents/skills/ means Codex workflows.

The test tree now follows the target `tests/unit/`, `tests/regression/`, `tests/integration/`, and `tests/live/` boundaries. `./scripts/check.sh` is the canonical local and CI validation entrypoint: it runs Ruff and the deterministic unit/regression suite only. Browser integration checks and executable provider/browser/manual scripts remain opt-in under `tests/integration/` and `scripts/smoke/`, keeping the default check credential-free and side-effect bounded.

Historical planning documents may describe work as missing even when current code has partially or fully implemented it.

Large modules have intertwined behavior; splitting them before regression boundaries are stable increases risk substantially.

Packaging/import cleanup can appear successful from the repository root while failing in an installed environment; eventually test installed-package imports explicitly.

 1. Non-goals of the base refactor

Unless explicitly requested in a dedicated task, this migration does not aim to:

redesign model prompts;

change provider ranking or model strategy;

weaken or remove verification layers;

make vision/consensus more aggressive;

rewrite stealth/input mechanics;

add large new frameworks;

perform a single “clean architecture” rewrite;

preserve obsolete APIs forever;

optimize every large file merely because it is large.

The purpose is to create a structure in which those later changes can be evaluated safely.

 1. Refactor exit criteria

 1. Current Wave 3 ownership

 The top-level runtime migration now has these canonical ownership boundaries:

 `agent_first_browse.workers` owns the active worker façade and its preserved
 contracts (`schemas.py`), prompt construction (`prompt_builder.py`),
 deterministic/model-free paths (`deterministic.py`), and model-backed decision
 orchestration (`decision.py`). `workers.base` is the stable worker façade.

 `agent_first_browse.browser.runtime` owns v16 browser launch, LOCAL_CDP versus
 local Playwright selection, persistent profile handling, `SessionGuard`, manual
 login, and shutdown. The historical `advanced_agent.py` façade has been
 removed and owns no runtime implementation.

 `agent_first_browse.agent.graph` owns the LangGraph orchestration spine.
 The historical `brain_graph.py` façade was removed and does not retain a
 second graph implementation. Repository-level checkpoint persistence remains at the same
 historical `persistence/` location.

 `agent_first_browse.cli` is the canonical v16 CLI and is exposed as the
 `agent-browse` project entry point. `agent.sh` and `Start-Agent.ps1` invoke
 that module; no root Python launcher remains.

 These moves preserve the worker proposal boundary, Overwatch execution
 authority, graph topology/state/retry behavior, browser session semantics, and
 CLI argument behavior. Root launchers and compatibility wrappers remain only
 where legacy imports or supported convenience entrypoints justify them; the
 old orchestrator and python-orchestrator implementations are removed.

The base structural refactor is substantially complete when:

one installable agent_first_browse package owns active application code;

launchers route through one canonical CLI and graph;

sys.path mutation for python-orchestrator is gone;

obsolete legacy orchestration code has no active imports and is removed;

application runtime skills/ has become an unambiguous action/capability package;

tests are separated by deterministic/integration/live requirements;

a single default validation command is reliable for Codex and humans;

CI runs the same deterministic checks used locally;

active architecture is described here rather than scattered across historical plans;

 state ownership and side-effect boundaries remain explicit;

## Final repository organization

The canonical runtime and all substantive application implementations live
under `src/agent_first_browse/`. The root contains only project metadata,
canonical documentation, the two supported shell/PowerShell launchers, and the
skills lock file required by the development tooling. Historical migration
documents are categorized under `docs/`; examples are under `examples/`; and
diagnostic, maintenance, manual, and opt-in smoke tools are under `scripts/`.

No root Python compatibility modules remain. Tests and scripts use canonical
package imports directly, and package-relative survey example data is shipped
under `agent_first_browse.survey`.

model/vision/provider call behavior has not accidentally regressed during structural migration.

After those conditions are met, deeper module decomposition and worker/subagent optimization can proceed with much lower risk.
