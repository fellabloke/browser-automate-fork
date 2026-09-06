# Project Goal: Measurable, Safe Browser Automation

Agent First Browse should complete legitimate browser tasks safely and
reliably. It should ground each action in current page evidence, execute one
proposal at a time, verify the resulting state, and stop with a bounded,
useful diagnosis when progress is not possible.

## Current runtime commitments

- Accessibility and DOM evidence remain the default perception path.
- Vision is used for genuine visual ambiguity, custom controls, and other
  evidence gaps.
- Workers propose actions; Overwatch remains the verification and commit
  authority.
- Intent journaling and idempotency protection prevent uncertain effects from
  becoming duplicate actions.
- Malformed structured output is repaired on the same model when supported;
  provider/model failover occurs only after repair fails.
- CAPTCHA handling retains independent read/compare checks and is not collapsed
  into generic retry suppression.
- Provider ordering, health/cooldown state, retry budgets, and model-call
  behavior remain explicit and bounded.

## Future goals

### Telemetry and evaluation

Add structured, credential-redacted records for provider attempts, browser
actions, verification outcomes, retries, latency, and terminal causes. Establish
deterministic mock-provider and mock-browser evaluations before changing runtime
policy.

Measure:

- successful task rate and grounded-control accuracy;
- first-model success, failover, and same-model repair rates;
- p50/p95 inference and browser latency;
- model and vision calls per successful task;
- duplicate actions, verification ambiguity, CAPTCHA outcomes, and stop causes.

### Measured cost and reliability reduction

Use evaluation evidence to reduce unnecessary model, vision, retry, and startup
work without weakening safety or verification. Any cost change must preserve
provider ordering, bounded budgets, structured repair ordering, and action
semantics.

### Worker capability architecture

Define explicit worker capabilities, supported action types, modality needs,
risk/cost classes, and model requirements. Introduce this only after the
current worker behavior has a measured baseline; do not infer capability
routing from provider identity alone.

### Selective subagents

Evaluate narrowly scoped subagents only where they reduce genuine uncertainty
or improve bounded recovery. They must use the existing proposal,
verification, and execution boundaries rather than creating a second runtime
orchestration path.

## Safety constraints

- Never weaken domain, action, CAPTCHA, or completion checks for throughput.
- Never log API keys, cookies, passwords, full screenshots, or personal survey
  answers outside an explicitly secured local diagnostic run.
- Never invent or persist respondent identity facts; use configured,
  user-authorized profile data only.
- Keep all autonomous retries and escalation loops bounded by state, evidence,
  and action-specific attempt budgets.
