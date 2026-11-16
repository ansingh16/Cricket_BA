"""Hierarchical logistic model with partially pooled batter and bowler effects.

    logit p = alpha[phase] + sigma_bat * z_bat[batter] + sigma_bowl * z_bowl[bowler]
              + beta * (season - 2016) / 10 + role[role] + experience[experience]

Non-centred: z ~ N(0, 1). The linear season term was chosen on 2023 as a validation season
in ``notebooks/04_season_drift.ipynb``. Role (from batting position) and experience (IPL
balls faced in earlier matches) are fixed effects with the first level as reference,
tested in ``notebooks/05_batting_role.ipynb``; ``TERMS`` lists those in use. The likelihood
is binomial on cells of balls that share every covariate,
which gives the same posterior as a Bernoulli likelihood on the individual balls, since
the two differ by a constant that does not involve the parameters
(``tests/test_hierarchical.py`` checks this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit, logit

from cricket_ba.features import EXPERIENCE, PHASES, ROLES

LEVELS = {"role": ROLES, "experience": EXPERIENCE}
TERMS: tuple[str, ...] = ("role", "experience")
KEYS = ["batter_id", "bowler_id", "phase", "season", *TERMS]
SAMPLER = dict(draws=1000, tune=1000, chains=4, cores=4, target_accept=0.95)
SEASON_CENTRE = 2016


def season_x(season) -> np.ndarray:
    """Season on the trend's scale: decades from 2016."""
    return (np.asarray(season, dtype=float) - SEASON_CENTRE) / 10


@dataclass
class Index:
    batters: pd.Index
    bowlers: pd.Index
    phases: pd.Index = field(default_factory=lambda: pd.Index(PHASES))


def make_index(train: pd.DataFrame) -> Index:
    return Index(pd.Index(train["batter_id"].unique()), pd.Index(train["bowler_id"].unique()))


def aggregate(balls: pd.DataFrame, idx: Index | None = None) -> pd.DataFrame:
    """Balls -> cells with n (balls) and k (wickets); add integer indices if given."""
    cells = balls.groupby(KEYS, observed=True)["wicket"].agg(n="size", k="sum").reset_index()
    if idx is not None:
        cells["bat_idx"] = idx.batters.get_indexer(cells["batter_id"])
        cells["bowl_idx"] = idx.bowlers.get_indexer(cells["bowler_id"])
        cells["phase_idx"] = idx.phases.get_indexer(cells["phase"])
        for t in TERMS:
            cells[f"{t}_idx"] = pd.Index(LEVELS[t]).get_indexer(cells[t])
    return cells


def build_model(cells: pd.DataFrame, idx: Index, observed: bool = True) -> pm.Model:
    coords = {"phase": list(idx.phases), "batter": list(idx.batters), "bowler": list(idx.bowlers)}
    for t in TERMS:
        coords[t] = list(LEVELS[t])
        coords[f"{t}_vs_ref"] = list(LEVELS[t][1:])
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=logit(0.05), sigma=0.5, dims="phase")
        sigma_bat = pm.HalfNormal("sigma_bat", 0.5)
        sigma_bowl = pm.HalfNormal("sigma_bowl", 0.5)
        z_bat = pm.Normal("z_bat", 0, 1, dims="batter")
        z_bowl = pm.Normal("z_bowl", 0, 1, dims="bowler")
        bat = pm.Deterministic("bat", sigma_bat * z_bat, dims="batter")
        bowl = pm.Deterministic("bowl", sigma_bowl * z_bowl, dims="bowler")
        beta = pm.Normal("beta", 0, 0.5)
        eta = (
            alpha[cells["phase_idx"].to_numpy()]
            + bat[cells["bat_idx"].to_numpy()]
            + bowl[cells["bowl_idx"].to_numpy()]
            + beta * season_x(cells["season"])
        )
        for t in TERMS:
            free = pm.Normal(f"{t}_free", 0, 0.5, dims=f"{t}_vs_ref")
            eff = pm.Deterministic(f"{t}_eff", pt.concatenate([pt.zeros(1), free]), dims=t)
            eta = eta + eff[cells[f"{t}_idx"].to_numpy()]
        pm.Binomial(
            "k",
            n=cells["n"].to_numpy(),
            logit_p=eta,
            observed=cells["k"].to_numpy() if observed else None,
        )
    return model


def fit(cells: pd.DataFrame, idx: Index, seed: int, **sampler):
    settings = {**SAMPLER, **sampler}
    with build_model(cells, idx):
        return pm.sample(**settings, random_seed=seed, progressbar=False)


@dataclass
class Draws:
    """Posterior draws as arrays, with draws on the last axis."""

    alpha: np.ndarray  # phases x S
    bat: np.ndarray  # batters x S
    bowl: np.ndarray  # bowlers x S
    sigma_bat: np.ndarray  # S
    sigma_bowl: np.ndarray  # S
    beta: np.ndarray  # S
    effects: dict[str, np.ndarray] = field(default_factory=dict)  # term -> levels x S

    @classmethod
    def from_idata(cls, idata) -> Draws:
        post = idata.posterior.stack(s=("chain", "draw"))
        return cls(
            alpha=post["alpha"].transpose("phase", "s").to_numpy(),
            bat=post["bat"].transpose("batter", "s").to_numpy(),
            bowl=post["bowl"].transpose("bowler", "s").to_numpy(),
            sigma_bat=post["sigma_bat"].to_numpy(),
            sigma_bowl=post["sigma_bowl"].to_numpy(),
            beta=post["beta"].to_numpy(),
            effects={
                t: post[f"{t}_eff"].transpose(t, "s").to_numpy()
                for t in TERMS
                if f"{t}_eff" in post
            },
        )


def predict_draws(cells: pd.DataFrame, idx: Index, draws: Draws, seed: int) -> np.ndarray:
    """Wicket probability per cell and posterior draw (cells x S).

    A player unseen in training gets, in each draw, a new effect from N(0, sigma) for
    that draw, so the uncertainty includes not knowing who the player is. The season trend is
    extrapolated linearly to seasons after training.
    """
    rng = np.random.default_rng(seed)
    S = draws.alpha.shape[1]
    new_bat = cells["batter_id"][~cells["batter_id"].isin(idx.batters)].unique()
    new_bowl = cells["bowler_id"][~cells["bowler_id"].isin(idx.bowlers)].unique()
    bat_all = np.vstack([draws.bat, rng.standard_normal((len(new_bat), S)) * draws.sigma_bat])
    bowl_all = np.vstack([draws.bowl, rng.standard_normal((len(new_bowl), S)) * draws.sigma_bowl])
    bat_index = idx.batters.append(pd.Index(new_bat))
    bowl_index = idx.bowlers.append(pd.Index(new_bowl))
    eta = draws.alpha[idx.phases.get_indexer(cells["phase"])]
    eta += bat_all[bat_index.get_indexer(cells["batter_id"])]
    eta += bowl_all[bowl_index.get_indexer(cells["bowler_id"])]
    eta += season_x(cells["season"])[:, None] * draws.beta
    for t, eff in draws.effects.items():
        eta += eff[pd.Index(LEVELS[t]).get_indexer(cells[t])]
    return expit(eta)
