# Project Goal: Autonomous Human-Like Survey Agent

## Objective

Build a fully autonomous browser agent that completes legitimate survey sessions
with near-perfect task accuracy and near-perfect human imitation, while remaining
safe, observable, recoverable, and respectful of site rules and user-controlled
identity data.

The agent should answer each survey question from authoritative current-page
evidence, execute exactly one grounded action at a time, verify the resulting
state, and continue without unnecessary model or vision calls.

## Definition of success

- At least 99% successful grounding of actionable controls in a representative
  survey benchmark, without unsafe or duplicate submissions.
- CAPTCHA input is never guessed or submitted without an independent,
  character-for-character comparison between the displayed code and the filled
  field.
- A healthy primary model normally completes many consecutive steps without
  failover; failover occurs only for a classified provider failure, not for a
  semantic disagreement or ordinary model output variation.
- No repeated request is made against an unchanged page state without a clear
  state-machine reason and a bounded attempt budget.
- Every provider attempt records model, provider, role, request mode, latency,
  status, sanitized provider response/error, and whether the response was used.
- A run can stop safely with a useful diagnosis instead of hanging, blindly
  repeating an action, or silently abandoning the task.
- Human-like timing, pointer movement, browser identity, and interaction style
  remain subordinate to correctness and must not weaken execution safety.

## Workstreams

### 1. Perception and grounding

- Keep accessibility/DOM perception as the default.
- Resolve element IDs and selectors against the fresh live DOM immediately before
  execution.
- Use vision only for genuine visual ambiguity, custom controls, canvas content,
  or CAPTCHA reading.
- Separate semantic confirmation, spatial grounding, and execution verification
  in state and telemetry.
- Never convert a missing coordinate into an automatic vision retry when a live
  DOM or accessibility target is available.

### 2. Model routing and failover

- Maintain a stable primary model for a run when it succeeds; do not rotate for
  ordinary successful steps merely because other keys exist.
- Classify failures into timeout, rate limit/quota, dead endpoint, schema
  incompatibility, malformed output, transport error, and safety refusal.
- Repair malformed structured output on the same model when possible before
  trying a sibling model.
- Use per-request budgets and provider-specific cooldowns without walking a
  large chain unnecessarily.
- Treat capability probing as optional, cached, quota-aware startup work.
- Pre-initialize reusable clients and overlap safe model initialization with
  browser startup/warm-up where that reduces wall-clock latency.

### 3. CAPTCHA accuracy state machine

- Preserve the deliberate two-check flow:
  1. independently read a fresh CAPTCHA and type a plausible code;
  2. independently reread the displayed CAPTCHA and compare it with the filled
     value character-for-character, including case.
- Submit only on exact agreement.
- On mismatch, replace the value once and verify again.
- On uncertainty or implausible text, refresh rather than guess.
- Generic loop protection must never suppress these required independent checks.
- Track CAPTCHA attempts separately from ordinary vision and action retries.

### 4. Verification and recovery

- Preserve the order: fresh DOM evidence, visual evidence when required, then
  path/completion proof for genuinely ambiguous outcomes.
- Keep the verification ledger sticky: a verified completed sub-goal cannot be
  silently demoted by a later weak observation.
- Distinguish “action failed,” “action succeeded but verification lagged,” and
  “provider failed before an action was produced.”
- Bound every retry/escalation loop by state identity, page evidence hash, and
  action-specific attempt count.

### 5. Observability and evaluation

- Emit structured per-run and per-provider metrics rather than relying only on
  free-form log text.
- Preserve bounded, credential-redacted provider response bodies and exception
  metadata for diagnosis.
- Add dashboards/reports for first-model success rate, failover rate, p50/p95
  inference latency, vision-call rate, CAPTCHA verification outcomes, duplicate
  actions, and terminal causes.
- Build a deterministic mock-provider/browser benchmark covering navigation,
  survey controls, delayed React state, overlays, shadow DOM, malformed JSON,
  timeouts, quota responses, and CAPTCHA mismatch/correction.

## Current findings to address first

1. Survey worker calls use a 15-second per-model timeout and a 45-second total
   failover budget. A single timeout therefore immediately invokes another model.
2. Runtime malformed structured output such as `Expecting value` is currently
   treated as generic failure, causing avoidable sibling failover instead of
   same-model JSON repair.
3. Startup capability probing runs after browser warm-up and can add substantial
   latency when providers time out; it probes representative combinations but
   does not fully pre-warm every credential.
4. Runtime logs usually retain only a truncated stringified exception. Provider
   HTTP status/body and response metadata are not consistently preserved.
5. The main survey path and the separate coordinate-only orchestrator have
   different routing and safety semantics; fixes must be applied to the active
   path rather than assumed to be shared.
6. CAPTCHA handling already has an explicit independent comparison state machine
   and must remain exempt from generic retry suppression.

## Near-term implementation sequence

1. Add structured provider-attempt records with safe response extraction and
   tests for timeout, HTTP error, quota, malformed JSON, and schema failures.
2. Repair malformed structured output on the same provider/model where safe;
   only then invoke a sibling fallback.
3. Add a run-scoped primary-model lease/affinity with health-based escape when
   the primary is actually unhealthy.
4. Cache and overlap startup preparation; measure whether probing improves live
   first-call latency before enabling broader pre-warming.
5. Add explicit CAPTCHA retry counters and regression tests proving the two-check
   comparison cannot be skipped or collapsed into a generic loop guard.
6. Add end-to-end benchmark scenarios and use their metrics as release gates.

## Guardrails

- Do not weaken domain, action, CAPTCHA, or completion safety checks to improve
  throughput.
- Do not log API keys, cookies, full screenshots, passwords, or personal survey
  answers unless explicitly configured for a secure local diagnostic run.
- Do not invent or persist respondent identity facts; use only configured,
  user-authorized profile data.
- Any autonomous stop must state the exact bounded reason and the last useful
  provider/browser evidence.
