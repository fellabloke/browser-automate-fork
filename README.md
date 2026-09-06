# 🌐 Agent First Browse

> **An autonomous, vision-capable browser agent that thinks before it acts, verifies before it moves on, and behaves like a real human user — not a script.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/langgraph-1.x-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Playwright](https://img.shields.io/badge/playwright-1.55%2B-green.svg)](https://playwright.dev/python/)
[![Status](https://img.shields.io/badge/status-active%20preview-orange.svg)](#project-status)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## Overview

**Agent First Browse** turns a plain-English instruction — *“search Flipkart for a water bottle under ₹300 and add it to the cart”*, *“star this GitHub repo”*, *“log in and create an API key”* — into a fully autonomous session in a **real, headed browser**. The primary Windows flow controls a dedicated native Chrome through CDP while the Python agent runs in WSL.

Unlike screen-scrapers or brittle click-bots, it runs on a **LangGraph cognitive “brain”**: a graph of specialized reasoning nodes that perceives the page through the **accessibility tree first** (fast and cheap), escalates to a **vision model only when genuinely confused**, and **verifies every outcome with evidence** before declaring success. It plans, predicts the consequence of each action, recovers from failure with a non-repeating tactical ladder, and presents a human-grade fingerprint to evade bot detection.

In short: it is engineered to be a **critical thinker**, not an obedient parrot — it reasons about *what* to do, *how little* it needs to confirm success, and *when* it is truly done.

---

## Table of Contents

- [Overview](#overview)
- [Why It’s Different](#why-its-different)
- [Key Features & Architecture](#key-features--architecture)
  - [1. The Cognitive Brain (LangGraph)](#1-the-cognitive-brain-langgraph)
  - [2. Accessibility-DOM-First Perception](#2-accessibility-dom-first-perception)
  - [3. Vision-on-Demand (Thinker, not Parrot)](#3-vision-on-demand-thinker-not-parrot)
  - [4. Critical-Thinker Verification](#4-critical-thinker-verification)
  - [5. Self-Curating Model Layer](#5-self-curating-model-layer)
  - [6. Human-Grade Anti-Bot Realism](#6-human-grade-anti-bot-realism)
  - [7. Resilient Execution & Self-Healing](#7-resilient-execution--self-healing)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration & API Keys](#configuration--api-keys)
  - [Operating Modes](#operating-modes)
  - [Text Models](#text-models)
  - [Vision Models (Optional)](#vision-models-optional)
  - [Graceful Fallback](#graceful-fallback)
  - [Self-Expanding Survey Profiles](#self-expanding-survey-profiles)
  - [Continuous Survey Sessions](#continuous-survey-sessions)
  - [Survey Audio Questions](#survey-audio-questions)
- [Usage](#usage)
- [Project Status](#project-status)
- [License](#license)

---

## Why It’s Different

Most browser agents do one of two things: drive everything through an expensive vision model (slow, imprecise clicking), or follow rigid DOM selectors (brittle, breaks on any UI change). Agent First Browse was engineered to avoid both traps — and to solve the failure modes those approaches ignore:

| Common failure in browser agents | How Agent First Browse solves it |
| --- | --- |
| Clicks the *wrong* look-alike element on dense pages | **Stable element registry** resolves the exact node the LLM chose, with fresh, drift-proof coordinates |
| “Completes” a task but can’t confirm it, then **loops re-doing it** | **Sticky Verification Ledger** — once a sub-goal is verified, it can never be silently un-completed |
| Misses a button that’s off-screen or rendered as a styled `<div>` | **Primary-action recall** + real-scroll fix surface the goal button even when the DOM hides it |
| Gets flagged as a bot and has its clicks ignored | **Native headed Windows Chrome** (or Xvfb for Linux fallback) + fingerprint stealth + trusted CDP input |
| Burns tokens/latency sending every frame to a vision model | **A11y-DOM by default**, vision **only** when the text view is ambiguous |
| Declares success without checking, or never stops | **Evidence-grounded outcome judge** that cites on-page proof |

---

## Key Features & Architecture

### 1. The Cognitive Brain (LangGraph)

The agent is not a simple `observe → act` loop. It is a **stateful graph of cognitive nodes**, each with a single responsibility:

```
Goal Compiler → Planner → Perceive → Router ─► [ Navigator | Interactor | Extractor ]
                                                          │
                                          Overwatch (multi-layer verification)
                                                          │
                          ┌───────────────┬───────────────┼───────────────┐
                        commit          retry           rollback        finalize
```

- **Goal Compiler / Planner** decompose the objective into a strategy, explicit *success criteria* (“done when…”), and an ordered checklist of sub-goals.
- **Mixture-of-Experts Router** dispatches each step to the right specialist worker (navigation, interaction, or data extraction).
- **Overwatch** is the only node allowed to commit state — every proposed action passes through layered verification first.
- A deterministic **escalation ladder** guarantees that a stuck step never repeats a tactic that already failed, and terminates cleanly instead of looping.

### 2. Accessibility-DOM-First Perception

Perception runs in a **single zero-mutation `page.evaluate()` pass (~80 ms)** that produces a compact, LLM-friendly semantic map of the page:

- **Shadow-DOM piercer** captures elements inside open *and* closed shadow roots.
- **Semantic Markdown compression** cuts perception tokens by 60–80% versus raw DOM/HTML.
- **Stable element registry** stamps every interactive element with a durable handle, so an action resolves to the *exact* node the model chose — re-read at click time, immune to layout drift and “snap to the wrong neighbor” errors.
- **Primary-action recall** guarantees commerce/checkout “goal” buttons (Add to Cart, Buy Now, Star, …) are surfaced even when they’re scrolled off-screen or rendered as non-semantic `<div>`s.

### 3. Vision-on-Demand (Thinker, not Parrot)

The agent works from the accessibility DOM by default and **“opens its eyes” only when it genuinely cannot resolve the page from text** — for example, two visually identical buttons where one is disabled, or a canvas-rendered control with no DOM. A single screenshot is sent to a vision model, the answer is mapped back to a **stable element id** (not fuzzy pixels), and the agent **immediately reverts to the a11y DOM**. Vision is strictly **viewport-bounded** — it never sees or maps coordinates onto the surrounding desktop.

### 4. Critical-Thinker Verification

This is the heart of the system — the mechanism that makes the agent trustworthy on multi-step tasks:

- **Forward Modeling (pre-click anticipation).** Before acting, the worker explicitly predicts the exact observable change (*“the button flips to ‘Starred’ and the count increments”*). The verifier then checks reality against that prediction.
- **Adaptive 3-Tier Verification.** The agent uses the *cheapest sufficient* proof, escalating only when needed:
  1. **A11y DOM** — did the predicted structural change happen (state switched, element appeared/vanished, redirect)?
  2. **Vision** — if the DOM is ambiguous for a small visual change, confirm it visually (e.g., “did the star fill?”).
  3. **Path Proof** — only if still unsure, navigate to where the result *undeniably* lives and confirm there.
- **Sticky Verification Ledger.** Once a sub-goal is verified, it is **permanently locked** — a later glance that can’t re-confirm it can never demote it back to “pending.” This is what eliminates the classic *“did it, couldn’t confirm it, did it again”* loop.
- **Task Serialization.** Given multiple tasks, the agent finishes and verifies **one** before it even considers the next.
- **Evidence-Grounded Outcome Judge.** The final “done” gate is a skeptical, independent verdict over the *fresh* page that must **cite concrete on-page proof**; if the proof isn’t there, it returns actionable feedback instead of blindly stopping or blindly looping.

### 5. Self-Curating Model Layer

- **Model-first failover** orders the chain by capability tier and expected cost, exhausting every instance of the best model (across all keys/providers) before falling to a weaker one.
- **Agentic Capability Gate.** At startup, each model is probed with a real structured-reasoning task; any model that can’t reliably produce agentic structured output is **excluded** — so weak models never derail a run. The pipeline is never left empty (a safety floor always remains).
- **Role Separation.** The *worker* (the critical decision-maker) draws only from top-tier proven models; cheaper models handle low-stakes auxiliary calls.
- **Dual-Mode design.** Runs equally well for the **free-tier user** (juggling several free keys with deep fallback) and the **premium user** (one paid, top-tier multimodal key that bypasses all the juggling — no probing, no gates).

### 6. Human-Grade Anti-Bot Realism

- **Native Windows Chrome via CDP (primary).** A dedicated Chrome profile runs on Windows with a real window/compositor; Playwright in WSL attaches to that exact browser instead of launching WSL Chromium. Linux-only runs retain the headed-under-Xvfb fallback.
- **12-layer fingerprint stealth:** `navigator.webdriver` proxy masking, canvas/WebGL/audio noise seeding, **platform-consistent GPU** spoofing, plugin/`chrome` stubs, WebRTC IP sanitization, and font hardening.
- **Trusted, human-like input:** Bézier-curve mouse paths and OS-level CDP clicks (`isTrusted = true`), a **realistic native arrow cursor** with accurate, continuous coordinate awareness, and automatic suppression of the Chrome “didn’t shut down correctly” crash bubble.

### 7. Resilient Execution & Self-Healing

- **Multi-strategy click waterfall** (CDP native event → JS click → direct navigation) with post-action verification.
- **Overlay penetration** to click through cookie banners and modals.
- **Circuit breaker + provider health tracker** quarantine failing models with exponential backoff and recover automatically.
- **Skill memory** records successful workflows for reuse, and a **per-run log file** captures every session for later analysis.

---

## How It Works

```
                ┌─────────────────────────────────────────────────────────┐
   "star this   │  GOAL COMPILER → strategy + success criteria + checklist │
    repo"  ───► │  PLANNER        → ordered, serialized sub-goals          │
                └───────────────────────────┬─────────────────────────────┘
                                            ▼
        ┌──────────────────────────── PER-STEP LOOP ───────────────────────────────┐
        │  PERCEIVE   a11y DOM (fast)  ──► ambiguous? ──► VISION consult (1 shot)    │
        │  DECIDE     worker predicts the exact expected change (forward modeling)   │
        │  EXECUTE    trusted CDP click / type / scroll on the live browser          │
        │  VERIFY     A11y → Vision → Path proof  (cheapest sufficient tier)         │
        │  LEDGER     verified sub-goal is locked — never re-done                    │
        └───────────────────────────────────────────────────────────────────────────┘
                                            ▼
                 OUTCOME JUDGE → cites on-page proof → ✅ done, or 🔁 actionable retry
```

---

## Installation

### Windows + WSL single-command flow (primary)

Requirements: Windows Chrome, WSL 2, and Python **3.11+** with this project's `.venv` created inside WSL. Run the Python installation steps below in WSL; installing Playwright Chromium and Xvfb is optional unless you want the Linux fallback.

WSL must use mirrored networking so WSL and Windows share `127.0.0.1`. Add this manually to `%UserProfile%\.wslconfig` (the launcher never edits it):

```ini
[wsl2]
networkingMode=mirrored
```

Then apply it once from PowerShell:

```powershell
wsl --shutdown
```

From Windows PowerShell in the project directory, run:

```powershell
.\Start-Agent.ps1 "Go to Reddit and find the top post about X"
```

The launcher finds Chrome, reuses a valid automation CDP endpoint when one already exists, otherwise starts Chrome with `%LOCALAPPDATA%\AgentFirstBrowse\ChromeProfile`, verifies `/json/version` from both Windows and WSL, and invokes the canonical `agent_first_browse.cli` entrypoint itself. No bridge terminal or transient WSL gateway IP is used. If your repository is in a non-default distribution, add `-Distro <name>`.

Launcher transcripts are saved under `logs/windows_launcher_<timestamp>.log`; Python runs are saved under `logs/run_<timestamp>.log`.

See [Microsoft's mirrored networking documentation](https://learn.microsoft.com/windows/wsl/networking#mirrored-mode-networking) for Windows/WSL version requirements.

### Python environment and Linux fallback

**Requirements:** Python **3.11+**, Linux / WSL / macOS, and optionally `xvfb` for a headed local-browser fallback.

```bash
# 1. Clone
git clone https://github.com/SandeepAi369/Agent-first-ide.git
cd Agent-first-ide

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install Python and development dependencies
pip install -e ".[dev]"

# 4. Install the Chromium browser engine (Linux/local fallback only)
python -m playwright install chromium

# 5. Install Xvfb for headed local fallback on a display-less machine (optional)
sudo apt-get install -y xvfb

# 6. Configure your API keys (see next section)
cp .env.example .env      # then edit .env with your keys

# 7. Make the launcher executable
chmod +x agent.sh
```

> 💡 If `xvfb` is not installed, the agent automatically falls back to headless mode (more bot-detectable, but fully functional).

### Deterministic validation

Run the same credential-free validation entrypoint used by CI:

```bash
./scripts/check.sh
```

This runs the repository's non-mutating Ruff checks and unit/regression tests.
Browser, provider, and credentialed smoke checks remain opt-in.

---

## Configuration & API Keys

All configuration lives in a `.env` file at the project root. The system is provider-agnostic and reads keys for any combination of providers you have.

### Survey Profiles

Survey identity is stored in `persistence/survey_profiles.json` (git-ignored) and
selected with `SURVEY_PROFILE_NAME`. Durable growth is disabled by default. It
requires both `SURVEY_PROFILE_AUTO_EXPAND_ENABLED=true` and
`learning.auto_expand: true`; even then, only allowlisted identity fields can be
stored. Survey-specific opinions, brands, purchases, recall answers and temporary
intentions stay cycle-local. Reusing a configured fact never creates another
learned record.

Profile maintenance is deterministic rather than model-authored: after
`SURVEY_PROFILE_SANITIZE_AFTER_WRITES` verified updates or
`SURVEY_PROFILE_SANITIZE_INTERVAL_HOURS`, it removes duplicate/transient DOM
answers, removes non-identity keys, normalizes canonical postcode aliases, and
writes a `.last-good` snapshot. The schema-v5 migration also keeps a one-time
`.pre-v5-backup`. It never invents or semantically rewrites personal facts. Invalid JSON
blocks factual survey input instead of silently falling back to an example
identity.

Attention checks, objective logic answers, and navigation actions are excluded
from character memory. Runtime checkpoints retain cycle-local continuity without
promoting those answers into the permanent respondent profile.

### Continuous Survey Sessions

Survey objectives run continuously by default. The agent treats qualification,
entry into the paid questionnaire, and each credited completion as intermediate
states. After a completed survey it returns to the dashboard, chooses the next
best reward-per-minute offer, and continues until you stop the process with
Ctrl+C. Question-text fingerprints count progress even when a provider keeps the
same URL and reuses the same `Next` element ID across its single-page app.

Set `SURVEY_CONTINUOUS_MODE=false` to restore one-task behavior, or say
"complete exactly one survey" in the objective for a one-run override.
`CONTINUOUS_GRAPH_RECURSION_LIMIT` is a high transition safety ceiling; it does
not control provider usage or API spend.

The ordered `SURVEY_PROVIDER_URLS` loop starts with Qmee by default. A provider
is rotated after `SURVEY_PROVIDER_ENTRY_STEP_LIMIT` committed browser actions or
`SURVEY_PROVIDER_ENTRY_TIMEOUT_SECONDS` elapsed seconds if no survey answer has
passed Overwatch verification. Once inside a survey, the
agent abandons it only when its canonical route, normalized question, and
meaningful form state have all remained identical for
`SURVEY_STUCK_TIMEOUT_SECONDS` (180 by default). Volatile tracking-query values
do not count as progress, while filling another field or matrix row does. Screen-outs, load failures, and this
confirmed timeout close the survey target, restore the provider dashboard, and
reset cycle-local context before selecting the next reward-per-minute offer.

Routine survey perception is event-driven: true navigations retain the generous
load ceiling, while same-page answers use `SURVEY_SAME_PAGE_SETTLE_MS`. Simple
pages may execute a guarded local action transaction (including automatic Next),
re-snapshotting and re-validating before every follow-up. Exact page recipes are
replayed only after two strictly verified question transitions and are stored in the git-ignored
`persistence/survey_recipes.db` database.

Long sessions use a bounded context lifecycle:

- During a survey, the model receives a compact cycle-local ledger of recent
  verified answers plus older answers retrieved when their wording is relevant
  to the current question, along with only the latest browser actions. Stable
  identity and preference answers remain in the durable respondent profile.
- The raw history, loop signatures, reflections, and model-facing history text
  all have hard size limits, so a long questionnaire cannot grow prompts without
  bound.
- Cleanup does not run merely because a survey is long. A stronger cycle reset
  occurs only after high-confidence completion/credit evidence was observed and
  the browser has left that completion page. It preserves the profile and cycle
  count while resetting prior-survey reasoning, retries, PRM state, and vision
  budget for the next offer.
- SQLite keeps a recent crash-recovery window for the active run, two snapshots
  for a bounded number of prior runs, and prunes redundant snapshots periodically.
  Logs and respondent-profile storage are separate and are never pruned by this.

The limits can be tuned with `AGENT_HISTORY_MAX_ENTRIES`,
`SURVEY_ANSWER_LEDGER_MAX`, `AGENT_HISTORY_PROMPT_MAX_CHARS`, and the
`CHECKPOINT_*` variables shown in `.env.example`.

### Survey Audio Questions

For animal-sound questions, the agent detects the listening instruction, captures
short audio/video media from the page (including accessible frames), and asks the
dedicated Gemini audio chain to choose among the visible answers. Media analysis
runs only for a detected challenge. If bytes are initially inaccessible, the agent
clicks Play and retries; if capture or classification still fails, it makes a
constrained non-`none of these` guess instead of looping or skipping the attempt.

Inline media is capped below Gemini's request-size limit. Configure or disable the
audio-only chain with `SURVEY_AUDIO_MODEL` and `SURVEY_AUDIO_ENABLED`.

### Operating Modes

Browser routing is endpoint-driven. With `LOCAL_CDP_ENDPOINT` set, the canonical browser runtime attaches with Playwright `connect_over_cdp()` and does not launch a browser or touch Xvfb. With the endpoint unset, the existing local Playwright persistent-context path is the explicit fallback.

```dotenv
BROWSER_MODE=LOCAL_CDP
BROWSER_HEADLESS=false
LOCAL_CDP_ENDPOINT=http://127.0.0.1:9222
```

Set `AGENT_MODE` to choose how the model layer behaves:

| `AGENT_MODE` | Behavior |
| --- | --- |
| `auto` *(default)* | **Premium** if a `PREMIUM_API_KEY` is set, otherwise **Free**. |
| `free` | Multi-key free-tier juggling + the Agentic Capability Gate. |
| `premium` | One trusted paid model for **both text and vision** — skips probing/gating entirely. |

**Premium (single-key) setup** — one paid, multimodal key does everything:

```dotenv
AGENT_MODE=premium
PREMIUM_API_KEY=sk-...
PREMIUM_MODEL=gpt-5                            # or claude-opus-4, gemini-2.5-pro, openrouter/...
PREMIUM_BASE_URL=https://api.openai.com/v1     # any OpenAI-compatible endpoint (OpenRouter, etc.)
# PREMIUM_VISION_MODEL=                        # optional; defaults to PREMIUM_MODEL
# PREMIUM_PROVIDER=openai                      # use "google" for the Gemini client
```

### Text Models

The reasoning chain leads with Gemini 3.5 Flash-Lite, followed by NVIDIA and
Cloudflare fallbacks. High-volume planner, critic, simulation, and outcome-judge
calls use a separate auxiliary ordering so they prefer Google and Cloudflare.

```dotenv
# Provider keys (any subset; comma-separate multiple keys of one provider)
NVIDIA_NIM_API_KEY=nvapi-...
GEMINI_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...

# Optional model overrides
NVIDIA_TEXT_MODELS=nvidia/nemotron-3.5-lightning-30b-a3b,openai/gpt-oss-120b
GEMINI_TEXT_MODEL=gemini-3.5-flash-lite
SURVEY_AUDIO_ENABLED=true
SURVEY_AUDIO_MODEL=gemini-3.5-flash
CLOUDFLARE_TEXT_MODELS=@cf/meta/llama-3.3-70b-instruct-fp8-fast
CLOUDFLARE_MAX_TOKENS=2048
WORKER_MODEL_ORDER=google:gemini-3.5-flash-lite,nvidia:nemotron-3.5-lightning-30b-a3b,cloudflare:llama-3.3-70b-instruct-fp8-fast,nvidia:gpt-oss-120b
AUXILIARY_PROVIDER_ORDER=google,cloudflare,nvidia
```

#### Cloudflare Workers AI

Workers AI exposes an
[OpenAI-compatible endpoint](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/),
so no extra Python SDK is required. Create an API token with Workers AI access,
copy the account ID from the Cloudflare dashboard, and fill in:

```dotenv
CLOUDFLARE_ENABLED=true
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
CLOUDFLARE_BASE_URL=  # blank selects the account-specific endpoint automatically
CLOUDFLARE_TEXT_MODELS=@cf/meta/llama-3.3-70b-instruct-fp8-fast
CLOUDFLARE_MAX_TOKENS=2048

# Optional screenshot fallback on the same account
CLOUDFLARE_VISION_ENABLED=true
CLOUDFLARE_VISION_MODELS=@cf/meta/llama-3.2-11b-vision-instruct
```

Before enabling that optional vision model, complete Cloudflare's one-time
[Meta license acceptance step](https://developers.cloudflare.com/workers-ai/models/llama-3.2-11b-vision-instruct/).
Text-only Cloudflare routing does not require that step.

Cloudflare currently includes 10,000 neurons per account per day at no charge;
its rate limits are per account/model. The configured text model supports
function calling. Check the live
[Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
and [limits](https://developers.cloudflare.com/workers-ai/platform/limits/), as
free allocations and catalogs can change. `CLOUDFLARE_MAX_TOKENS` is explicitly
set because Workers AI otherwise defaults this model to a 256-token response.

#### Recommended setup — diversify providers, not keys

Use one valid, authorized key for each provider first: Cloudflare + Google
Gemini + NVIDIA provides more real headroom than adding keys that share the same
account, organization, or project. Additional keys are still useful for credential
rotation and genuinely independent projects you are authorized to operate, but do
not assume they create quota:

- Gemini limits apply per project, not per API key. See
  [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
- Cloudflare's allocation is account-level; multiple models or tokens do not
  multiply the daily neuron allocation.

The startup capability gate removes invalid, unavailable, or structurally
incompatible model/provider combinations before normal work begins.

#### Free-tier usage controls

The following defaults prevent routine browser turns from triggering several
extra model calls. Irreversible/high-risk actions still force safety consensus.

```dotenv
CLARITY_CONSENSUS_ENABLED=0
REALITY_LLM_ENABLED=0
WEBDREAMER_ENABLED=0
PRM_AUDIT_EVERY=4
WEB_DREAMER_NUM_CANDIDATES=1
WEB_DREAMER_NUM_SIMULATIONS=1
MODEL_TIMEOUT_FLOOR_SECONDS=15
CONSENSUS_VOTER_TIMEOUT_SECONDS=12
```

Model/key health is stored anonymously in `persistence/model_health.json`, so a
new run reuses the responsive key/model instead of relearning the pool. Gemini
also keeps a local per-project RPM/TPM/RPD ledger. The Gemini API key itself
cannot read exact remaining project quota, so copy the limits shown in AI Studio
into the ordered lists when the six projects have different limits:

```dotenv
MODEL_HEALTH_PERSISTENCE=true
MODEL_FAILOVER_MAX_ATTEMPTS=12
MODEL_ROLE_PRIORITY_PENALTY_SECONDS=6
MODEL_TIER_PENALTY_SECONDS=5
# Illustrative only—replace every number with that project's displayed limit.
GEMINI_PROJECT_RPM_LIMITS=10,15,10,5,20,10
GEMINI_PROJECT_TPM_LIMITS=250000,250000,100000,100000,250000,100000
GEMINI_PROJECT_RPD_LIMITS=500,1000,500,250,1000,500
GEMINI_USAGE_SOFT_LIMIT_PERCENT=90
```

List position zero is `GEMINI_API_KEY`; the remaining positions follow
`GEMINI_API_KEY_FALLBACKS`. Leave a position blank to inherit the corresponding
singular `GEMINI_PROJECT_*_LIMIT` value, or leave all limits at zero when they
are unknown. Provider retry delays and daily-quota failures are learned and
persisted automatically.

### Vision Models (Optional)

Vision is **entirely optional** — used only for on-demand visual confirmation. Defaults to the NVIDIA Llama 4 family.

```dotenv
# NVIDIA Vision — Llama 4 Maverick (+ optional fallbacks, comma-separated)
NVIDIA_VISION_API_KEY=nvapi-...
NVIDIA_VISION_MODELS=meta/llama-4-maverick-17b-128e-instruct
```

### Graceful Fallback

- **No vision keys?** The agent runs **accessibility-DOM only** — it never crashes or requires vision; it simply doesn’t escalate to a screenshot. (NVIDIA vision transparently falls back to your text NVIDIA key if a dedicated vision key isn’t set.)
- **Dead or incapable model?** The startup Capability Gate prunes it automatically; the chain self-curates down to models that actually work.
- **All top models rate-limited?** Failover walks the full chain across every key/provider before giving up.

---

## Usage

Windows PowerShell (primary, starts Chrome and WSL automatically):

```powershell
.\Start-Agent.ps1 "Go to github.com/torvalds/linux and star the repository"
```

Linux/manual WSL fallback:

```bash
./agent.sh "search Flipkart for a water bottle under ₹300 and add it to the cart"
```

Run it with no argument and it will prompt you for the task:

```bash
./agent.sh
```

Or call the brain directly:

```bash
# Stealth headed mode (default — recommended)
.venv/bin/python -m agent_first_browse.cli run "Go to github.com/torvalds/linux and star the repository"

# Force true headless (more bot-detectable)
.venv/bin/python -m agent_first_browse.cli run "your task here" --headless
```

**Persisting a login session** — open a browser to sign in manually once; the session is reused on future runs:

```bash
.venv/bin/python -m agent_first_browse.cli login
```

Every run is saved to `logs/run_<timestamp>.log` for later inspection.

---

## Project Status

Agent First Browse is in **active private preview**. The core architecture — cognitive brain, perception, verification, model layer, and anti-bot stealth — is implemented and covered by an automated test suite. APIs and configuration may still evolve.

---

## Acknowledgements & Credits

Agent First Browse stands on the shoulders of outstanding open-source work and research. Deep thanks to:

- **[LangGraph](https://github.com/langchain-ai/langgraph)** & **[LangChain](https://github.com/langchain-ai/langchain)** — the stateful graph that forms the orchestration spine of the cognitive brain.
- **[Playwright](https://playwright.dev/python/)** — the browser-automation engine (CDP, trusted OS-level input).
- **[Pydantic](https://github.com/pydantic/pydantic)** — typed global state and strict structured-output schemas.
- **Browser-agent projects we studied and learned from** (concepts assimilated and re-implemented cleanly, never copied): **[browser-use](https://github.com/browser-use/browser-use)** (DOM pruning, viewport filtering, CDP event-listener detection), **[Stagehand](https://github.com/browserbase/stagehand)** (act/observe/extract action abstraction), **Skyvern** (multi-signal element identification), **Crawl4AI** (markdown compression), **BrowserGym** (stable element IDs), and **Agent-E** (text-DOM-first navigation).
- **Research that shaped the cognitive layers** (cited inline across the modules): WebDreamer (model-based planning), LATS — Language Agent Tree Search, Reflexion, CISC / self-consistency, Self-Grounded Verification, PABU, Prune4Web, and the MAST / Six-Sigma-Agent reliability analyses.
- **Model providers** for fast, accessible inference: **[Groq](https://groq.com)**, **[NVIDIA NIM](https://build.nvidia.com)**, **[Google Gemini](https://ai.google.dev)**, and **[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)**.

If your project or work is reflected here and you'd like different or additional attribution, please open an issue — credit is gladly given.

## License

Licensed under the **GNU General Public License v3 (GPLv3)** — see [LICENSE](LICENSE) for the full text.

You are free to use, study, share, and modify this software under the terms of the GPLv3; derivative works
and redistributions must remain licensed under GPLv3 and keep this notice. The software is provided **"as is",
without warranty of any kind**, express or implied.

```
Agent First Browse — an autonomous, vision-capable browser agent.
Copyright (C) 2026  SandeepAi369

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed WITHOUT ANY WARRANTY; see the GNU GPL v3 for details.
```
