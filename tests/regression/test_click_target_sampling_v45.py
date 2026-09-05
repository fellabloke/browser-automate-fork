"""Regression tests for non-centre live target sampling."""

import mcp_tools


def test_sample_click_point_stays_inside_target_and_varies(monkeypatch):
    values = iter([0.30, 0.70, 0.70, 0.30])
    monkeypatch.setattr(mcp_tools.random, "uniform", lambda _low, _high: next(values))
    rect = {"x": 100, "y": 200, "width": 100, "height": 50}

    first = mcp_tools._sample_click_point(rect, 150, 225)
    second = mcp_tools._sample_click_point(rect, 150, 225)

    assert first == (130.0, 235.0)
    assert second == (170.0, 215.0)
    assert first != second


def test_sample_click_point_uses_center_for_tiny_target():
    assert mcp_tools._sample_click_point(
        {"x": 1, "y": 2, "width": 7, "height": 20}, 4, 12
    ) == (4.0, 12.0)
