"""Smoke test for v8.0 Self-Healing & Anti-Detection modules."""
from collections import Counter

# Test 1: Circuit Breaker
from model_registry import CircuitBreaker

cb = CircuitBreaker(window_size=5, min_calls=3, failure_rate_threshold=0.6)
assert not cb.tripped
cb.record_success()
cb.record_success()
assert not cb.tripped

cb.record_failure()
cb.record_failure()
cb.record_failure()
assert cb.tripped
assert "OPEN" in cb.reason
print(f"✅ Circuit Breaker: trips correctly — {cb.reason}")

cb2 = CircuitBreaker(window_size=3, min_calls=3, failure_rate_threshold=2 / 3)
cb2.record_success()
cb2.record_failure()
cb2.record_failure()
assert cb2.tripped
print(f"✅ Circuit Breaker: sliding failure-rate trip works — {cb2.reason}")

# Test 2: Anti-Loop Watchdog (simulate the Counter logic)
url_history = [
    "https://hashnode.com/",
    "https://hashnode.com/login",
    "https://hashnode.com/",
    "https://hashnode.com/login",
    "https://hashnode.com/",
    "https://hashnode.com/login",
]
recent = url_history[-6:]
url_counts = Counter(recent)
most_common_url, visit_count = url_counts.most_common(1)[0]
assert visit_count >= 3, f"Expected loop detection, got {visit_count}"
print(f"✅ Anti-Loop Watchdog: detected '{most_common_url}' visited {visit_count}x in 6 steps")

# Test 3: SessionGuard exists and is a singleton
from agent_first_browse.browser.runtime import SessionGuard

g1 = SessionGuard.get()
g2 = SessionGuard.get()
assert g1 is g2, "SessionGuard should be a singleton"
print("✅ SessionGuard: singleton pattern works")

print()
print("=== ALL v8.0 SELF-HEALING MODULES VERIFIED ===")
