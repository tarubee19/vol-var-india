"""API tests: run against the small simulated pipeline output via FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from volvar.api import create_app


@pytest.fixture(scope="module")
def client(pipeline_run):
    with TestClient(create_app(results_dir=pipeline_run[1])) as c:
        yield c


@pytest.fixture(scope="module")
def ticker(pipeline_run):
    import json

    return json.loads((pipeline_run[1] / "meta.json").read_text())["tickers"][0]


def test_health_reports_dataset_and_labels_every_response(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["dataset"] == "simulated"
    assert r.headers["X-Data-Label"] == "simulated"


def test_forecast_endpoint(client, ticker):
    r = client.get(f"/forecast/{ticker}", params={"model": "ewma"})
    body = r.json()
    assert r.status_code == 200 and body["model"] == "ewma"
    assert body["vol_daily_pct"] > 0
    assert body["vol_annual_pct"] == pytest.approx(body["vol_daily_pct"] * 252**0.5, rel=0.01)


def test_var_endpoint_defaults_and_ordering(client, ticker):
    body = client.get(f"/var/{ticker}").json()
    assert body["level"] == 0.99 and body["method"] == "fhs"
    assert body["es_pct"] >= body["var_pct"] > 0
    v95 = client.get(f"/var/{ticker}", params={"level": 0.95, "model": body["model"]}).json()
    assert v95["var_pct"] < body["var_pct"]  # higher confidence means a larger VaR


def test_backtest_endpoints(client, ticker):
    bt = client.get(f"/backtest/{ticker}", params={"model": "garch", "level": 0.95}).json()
    assert bt["violations"] <= bt["n_days"] and bt["expected_rate_pct"] == 5.0
    summ = client.get("/backtest-summary", params={"model": "ewma"}).json()
    assert 0 <= summ["pass_rate_pct"] <= 100


def test_models_endpoint_sorted_by_metric(client):
    for metric in ("qlike", "mse"):
        ranking = client.get("/models", params={"metric": metric}).json()["ranking"]
        values = [r[metric] for r in ranking]
        assert values == sorted(values)


def test_error_statuses(client, ticker):
    assert client.get("/forecast/NOPE").status_code == 404
    assert client.get(f"/var/{ticker}", params={"level": 0.5}).status_code == 422
    assert client.get(f"/var/{ticker}", params={"model": "magic"}).status_code == 422
    assert client.get("/models", params={"metric": "sharpe"}).status_code == 422
    assert client.post("/assistant", json={"question": ""}).status_code == 422
    assert client.post("/assistant", json={"question": "x" * 2000}).status_code == 422


def test_assistant_endpoint(client, ticker):
    r = client.post("/assistant", json={"question": f"What is {ticker}'s 99% VaR?"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] and body["dataset"] == "simulated"
    assert body["tool_calls"][0]["name"] == "get_var_es" and "SIMULATED" in body["answer"]


def test_service_degrades_to_503_without_results(tmp_path):
    with TestClient(create_app(results_dir=tmp_path)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and "run_pipeline" in health.json()["detail"]
        assert c.get("/meta").status_code == 503
        assert "X-Data-Label" not in health.headers


def test_live_backend_without_key_returns_503(pipeline_run, monkeypatch):
    from volvar.assistant import AnthropicBackend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(results_dir=pipeline_run[1], backend=AnthropicBackend())
    with TestClient(app) as c:
        r = c.post("/assistant", json={"question": "What is VaR?"})
        assert r.status_code == 503
