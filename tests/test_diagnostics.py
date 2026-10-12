import numpy as np
import pandas as pd

from volvar.diagnostics import (
    adf_pvalue,
    arch_lm_pvalue,
    describe_returns,
    diagnostics_table,
    ljung_box_pvalue,
)


def test_iid_noise_has_no_arch_effects():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(3000))
    assert arch_lm_pvalue(x) > 0.01
    assert ljung_box_pvalue(x**2) > 0.01


def test_garch_like_data_shows_arch_effects(one_panel):
    r = one_panel["ret"]
    assert arch_lm_pvalue(r) < 0.05
    assert ljung_box_pvalue(r**2) < 0.05


def test_returns_are_stationary_random_walk_is_not():
    rng = np.random.default_rng(1)
    walk = pd.Series(np.cumsum(rng.standard_normal(2000)))
    assert adf_pvalue(walk.diff().dropna()) < 0.01
    assert adf_pvalue(walk) > 0.05


def test_describe_and_table_shapes(small_panels):
    row = describe_returns(small_panels["SIM01"]["ret"])
    assert row["n"] > 1000
    assert row["excess_kurtosis"] > 0  # fat tails from Student-t innovations
    table = diagnostics_table(small_panels)
    assert list(table.index) == list(small_panels)
    assert {"arch_lm_p", "ljungbox_p_squared", "adf_p_returns"} <= set(table.columns)
