"""Shared fixtures: small simulated universes so tests stay fast."""

from __future__ import annotations

import pytest

from volvar.data import make_panels
from volvar.synthetic import simulate_universe


@pytest.fixture(scope="session")
def small_universe():
    return simulate_universe(n_days=1500, n_tickers=3, seed=5)


@pytest.fixture(scope="session")
def small_panels(small_universe):
    return make_panels(small_universe)


@pytest.fixture(scope="session")
def one_panel(small_panels):
    return small_panels["SIM01"]
