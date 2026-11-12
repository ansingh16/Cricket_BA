"""Scores, a match-level bootstrap, calibration bins and interval coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def scores(y, p) -> dict[str, float]:
    return {"log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p)}


def per_ball_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }


def match_bootstrap(
    df: pd.DataFrame, cols: list[str], n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap the mean per-ball log loss and Brier score of each prediction column.

    Whole matches are resampled, because balls in the same match are not independent.
    Returns one row per bootstrap replicate, columns ``(col, metric)``.
    """
    r = np.random.default_rng(seed)
    y = df["wicket"].to_numpy()
    losses = {}
    for c in cols:
        for metric, v in per_ball_losses(y, df[c].to_numpy()).items():
            losses[(c, metric)] = v
    L = pd.DataFrame(losses)
    match = df["match_id"].to_numpy()
    per_match_sum = L.groupby(match).sum()
    per_match_n = L.groupby(match).size().to_numpy()
    sums = per_match_sum.to_numpy()
    ids = np.arange(len(per_match_sum))
    out = []
    for _ in range(n_boot):
        pick = r.choice(ids, size=len(ids), replace=True)
        out.append(sums[pick].sum(0) / per_match_n[pick].sum())
    return pd.DataFrame(out, columns=per_match_sum.columns)


def difference_interval(boot: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """2.5%, 50% and 97.5% quantiles of metric(a) - metric(b) across replicates."""
    diff = pd.DataFrame({m: boot[(a, m)] - boot[(b, m)] for m in ["log_loss", "brier"]})
    q = diff.quantile([0.025, 0.5, 0.975]).T
    q.columns = ["2.5%", "median", "97.5%"]
    return q


def calibration_points(y, p, n_bins: int = 10):
    """Mean prediction, observed rate and 95% half-width per quantile bin of ``p``."""
    q = pd.qcut(p, n_bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    obs, n = g["y"].mean(), g["y"].size()
    return g["p"].mean(), obs, 1.96 * np.sqrt(obs * (1 - obs) / n)


def interval_coverage(
    sim_totals: np.ndarray,
    observed: np.ndarray,
    levels=(0.5, 0.8, 0.95),
    seed: int = 0,
) -> dict[str, float]:
    """How often observed totals fall inside central predictive intervals.

    ``sim_totals`` is players x draws. Two versions per level: the share of players
    whose observed total lies inside the inclusive central interval, and a randomised
    PIT version, which corrects for whole-number counts covering a little more than the
    nominal level.
    """
    r = np.random.default_rng(seed)
    below = (sim_totals < observed[:, None]).mean(1)
    at = (sim_totals == observed[:, None]).mean(1)
    pit = below + r.uniform(size=len(observed)) * at
    out = {}
    for lev in levels:
        lo = np.quantile(sim_totals, (1 - lev) / 2, axis=1)
        hi = np.quantile(sim_totals, 1 - (1 - lev) / 2, axis=1)
        out[f"{int(lev * 100)}% interval"] = float(((observed >= lo) & (observed <= hi)).mean())
        out[f"{int(lev * 100)}% PIT"] = float((np.abs(pit - 0.5) <= lev / 2).mean())
    return out
