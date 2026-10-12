"""Headless smoke tests for the Streamlit dashboard (Streamlit's AppTest, no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


def run_app(results_dir, monkeypatch):
    monkeypatch.setenv("VOLVAR_RESULTS_DIR", str(results_dir))
    return AppTest.from_file(str(APP), default_timeout=90).run()


def test_dashboard_renders_with_simulated_banner(pipeline_run, monkeypatch):
    at = run_app(pipeline_run[1], monkeypatch)
    assert not at.exception
    assert any("SIMULATED DATA" in e.value for e in at.error)
    assert [t.label for t in at.tabs] == [
        "Model comparison",
        "Next-day forecast",
        "VaR backtest",
        "Ask the assistant",
    ]
    labels = {m.label for m in at.metric}
    assert {"Volatility (daily)", "Violations", "Kupiec p-value"} <= labels


def test_dashboard_selection_and_chat(pipeline_run, monkeypatch):
    at = run_app(pipeline_run[1], monkeypatch)
    second = at.sidebar.selectbox[0].options[1]
    at.sidebar.selectbox[0].set_value(second).run()
    at.sidebar.radio[0].set_value(0.95).run()
    assert not at.exception
    assert any(m.label == "95% VaR" for m in at.metric)
    at.chat_input[0].set_value("Which model has the lowest QLIKE?").run()
    assert not at.exception
    assert any("QLIKE" in c.markdown[0].value for c in at.chat_message)


def test_dashboard_explains_missing_results(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert any("run_pipeline" in e.value for e in at.error)
