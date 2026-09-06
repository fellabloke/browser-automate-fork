# Smoke and manual scripts

These scripts are intentionally outside the default pytest validation set.
Some require browsers, network access, provider APIs, credentials, or manual
observation. Run them explicitly and inspect each script before execution.

The `run_*` scripts are live-site examples. `test_nvidia.py` requires an NVIDIA
credential. `test_auth_graph.py` was removed because credentialed scripts must
not contain embedded secrets.
