import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from cricket_ba.evaluate import (
    difference_interval,
    interval_coverage,
    match_bootstrap,
    per_ball_losses,
)


def _toy(seed=0, n_matches=40, balls=50):
    r = np.random.default_rng(seed)
    n = n_matches * balls
    p = r.uniform(0.02, 0.1, n)
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n_matches), balls),
            "wicket": r.binomial(1, p),
            "good": p,
            "flat": np.full(n, 0.06),
        }
    )


def test_per_ball_log_loss_matches_sklearn():
    df = _toy()
    ll = per_ball_losses(df["wicket"].to_numpy(), df["good"].to_numpy())["log_loss"].mean()
    assert np.isclose(ll, log_loss(df["wicket"], df["good"]))


def test_bootstrap_interval_contains_the_point_estimate():
    df = _toy()
    boot = match_bootstrap(df, ["good", "flat"], n_boot=300, seed=1)
    ci = difference_interval(boot, "good", "flat")
    point = log_loss(df["wicket"], df["good"]) - log_loss(df["wicket"], df["flat"])
    assert ci.loc["log_loss", "2.5%"] <= point <= ci.loc["log_loss", "97.5%"]


def test_coverage_is_nominal_when_the_model_is_true():
    r = np.random.default_rng(2)
    n_players, draws = 2000, 1000
    n_balls = r.integers(60, 400, n_players)
    p = r.uniform(0.02, 0.08, n_players)
    sims = r.binomial(n_balls[:, None], p[:, None], size=(n_players, draws))
    obs = r.binomial(n_balls, p)
    cov = interval_coverage(sims, obs, seed=3)
    for lev in (50, 80, 95):
        assert abs(cov[f"{lev}% PIT"] - lev / 100) < 0.03
        assert cov[f"{lev}% interval"] >= lev / 100 - 0.03  # discrete: at or above nominal
