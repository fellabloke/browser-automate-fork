# Live validation

This directory is reserved for pytest tests that require browser sessions,
network access, provider APIs, or credentials. Such tests must remain opt-in
and must never become part of ordinary `pytest` discovery.

Current executable live/manual checks are kept under `scripts/smoke/`.
