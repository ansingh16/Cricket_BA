"""Model tests without MCMC."""

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logit
from scipy.stats import bernoulli

from cricket_ba.hierarchical import (
    LEVELS,
    TERMS,
    Draws,
    aggregate,
    build_model,
    make_index,
    predict_draws,
)


def _balls(seed=0, n=3000):
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "batter_id": r.choice([f"b{i}" for i in range(15)], n),
            "bowler_id": r.choice([f"w{i}" for i in range(10)], n),
            "phase": r.choice(["powerplay", "middle", "death"], n),
            "season": r.choice([2010, 2016, 2023], n),
            "role": r.choice(list(LEVELS["role"]), n),
            "experience": r.choice(list(LEVELS["experience"]), n),
            "wicket": r.binomial(1, 0.05, n),
        }
    )


def _point(idx, seed):
    r = np.random.default_rng(seed)
    return {
        "alpha": r.normal(-3, 0.3, 3),
        "sigma_bat_log__": np.log(r.uniform(0.1, 0.6)),
        "sigma_bowl_log__": np.log(r.uniform(0.1, 0.6)),
        "z_bat": r.normal(size=len(idx.batters)),
        "z_bowl": r.normal(size=len(idx.bowlers)),
        "beta": r.normal(0, 0.3),
        **{f"{t}_free": r.normal(0, 0.3, len(LEVELS[t]) - 1) for t in TERMS},
    }


def test_binomial_on_cells_equals_bernoulli_on_balls_up_to_a_constant():
    balls = _balls()
    idx = make_index(balls)
    cells = aggregate(balls, idx)
    assert cells["n"].sum() == len(balls) and cells["k"].sum() == balls["wicket"].sum()

    model = build_model(cells, idx)
    loglik = model.compile_fn(model.observedlogp)
    b = balls.assign(
        bat_idx=idx.batters.get_indexer(balls["batter_id"]),
        bowl_idx=idx.bowlers.get_indexer(balls["bowler_id"]),
        phase_idx=idx.phases.get_indexer(balls["phase"]),
    )

    def bernoulli_loglik(pt):
        eta = (
            pt["alpha"][b["phase_idx"]]
            + np.exp(pt["sigma_bat_log__"]) * pt["z_bat"][b["bat_idx"]]
            + np.exp(pt["sigma_bowl_log__"]) * pt["z_bowl"][b["bowl_idx"]]
            + pt["beta"] * (b["season"] - 2016) / 10
        )
        for t in TERMS:
            eff = np.concatenate([[0.0], pt[f"{t}_free"]])
            eta = eta + eff[pd.Index(LEVELS[t]).get_indexer(b[t])]
        return bernoulli.logpmf(b["wicket"], 1 / (1 + np.exp(-eta))).sum()

    gaps = []
    for seed in range(3):
        pt = _point(idx, seed)
        gaps.append(loglik(pt) - bernoulli_loglik(pt))
    # the gap is the sum of log binomial coefficients, the same at every parameter value
    assert np.allclose(gaps, gaps[0], atol=1e-6)


def test_prior_predictive_runs():
    balls = _balls()
    idx = make_index(balls)
    with build_model(aggregate(balls, idx), idx, observed=False):
        prior = pm.sample_prior_predictive(draws=50, random_seed=1)
    # with observed=False, k is a free variable and lands in the prior group
    assert prior.prior["k"].shape[-1] == len(aggregate(balls))


def test_predictions_cover_unseen_players():
    balls = _balls()
    idx = make_index(balls)
    S = 200
    r = np.random.default_rng(0)
    draws = Draws(
        alpha=np.full((3, S), -3.0),
        bat=r.normal(0, 0.3, (len(idx.batters), S)),
        bowl=r.normal(0, 0.3, (len(idx.bowlers), S)),
        sigma_bat=np.full(S, 0.3),
        sigma_bowl=np.full(S, 0.3),
        beta=np.full(S, 0.2),
    )
    test = pd.DataFrame(
        {
            "batter_id": ["b0", "new_batter", "b1"],
            "bowler_id": ["w0", "w1", "new_bowler"],
            "phase": ["death", "middle", "powerplay"],
            "season": [2024, 2024, 2025],
            "role": ["top", "tail", "middle"],
            "experience": ["500+", "<100", "100-499"],
            "wicket": [0, 1, 0],
        }
    )
    p = predict_draws(aggregate(test), idx, draws, seed=0)
    assert p.shape == (3, S)
    assert np.all((p > 0) & (p < 1))
    # an unseen batter's draws vary, because his effect is drawn fresh in each draw
    row = aggregate(test).reset_index(drop=True)
    assert p[row.index[row["batter_id"] == "new_batter"][0]].std() > 0


def test_season_trend_shifts_predictions_by_beta_per_decade():
    balls = _balls()
    idx = make_index(balls)
    S = 4
    draws = Draws(
        alpha=np.full((3, S), -3.0),
        bat=np.zeros((len(idx.batters), S)),
        bowl=np.zeros((len(idx.bowlers), S)),
        sigma_bat=np.full(S, 0.3),
        sigma_bowl=np.full(S, 0.3),
        beta=np.full(S, 0.5),
    )
    cells = pd.DataFrame(
        {
            "batter_id": ["b0", "b0"],
            "bowler_id": ["w0", "w0"],
            "phase": ["middle"] * 2,
            "season": [2016, 2026],
        }
    )
    lp = logit(predict_draws(cells, idx, draws, seed=0))
    assert np.allclose(lp[1] - lp[0], 0.5)


def test_fixed_effects_shift_the_logit():
    balls = _balls()
    idx = make_index(balls)
    S = 3
    draws = Draws(
        alpha=np.full((3, S), -3.0),
        bat=np.zeros((len(idx.batters), S)),
        bowl=np.zeros((len(idx.bowlers), S)),
        sigma_bat=np.full(S, 0.3),
        sigma_bowl=np.full(S, 0.3),
        beta=np.zeros(S),
        effects={"role": np.array([[0.0] * S, [0.1] * S, [0.2] * S, [0.4] * S])},
    )
    cells = pd.DataFrame(
        {
            "batter_id": ["b0", "b0"],
            "bowler_id": ["w0", "w0"],
            "phase": ["middle"] * 2,
            "season": [2016] * 2,
            "role": ["top", "tail"],
            "experience": ["500+"] * 2,
        }
    )
    lp = logit(predict_draws(cells, idx, draws, seed=0))
    assert np.allclose(lp[1] - lp[0], 0.4)
