"""L2-regularised logistic regression on one-hot player and phase features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from cricket_ba.evaluate import scores

FEATURE_SETS = {
    "batter+bowler+phase": ["batter_id", "bowler_id", "phase"],
    "batter+bowler+phase+venue": ["batter_id", "bowler_id", "phase", "venue_clean"],
}
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def logistic(features: list[str], C: float) -> Pipeline:
    """Players unseen in training get a zero coefficient, i.e. are treated as average."""
    enc = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), features)], sparse_threshold=1.0
    )
    return make_pipeline(enc, LogisticRegression(C=C, max_iter=2000))


def time_cv(
    train: pd.DataFrame,
    val_seasons: list[int],
    feature_sets: dict[str, list[str]] = FEATURE_SETS,
    c_grid=C_GRID,
) -> pd.DataFrame:
    """Expanding-window CV: to validate on season s, fit on every season before s."""
    rows = []
    for s in val_seasons:
        tr, va = train[train["season"] < s], train[train["season"] == s]
        const = np.full(len(va), tr["wicket"].mean())
        rows.append(
            {"features": "overall rate", "C": np.nan, "season": s, **scores(va["wicket"], const)}
        )
        for name, feats in feature_sets.items():
            for C in c_grid:
                m = logistic(feats, C).fit(tr[feats], tr["wicket"])
                p = m.predict_proba(va[feats])[:, 1]
                rows.append({"features": name, "C": C, "season": s, **scores(va["wicket"], p)})
    return pd.DataFrame(rows)


def choose(cv: pd.DataFrame) -> tuple[str, float]:
    """Feature set and C with the lowest mean validation log loss."""
    mean = cv[cv["features"] != "overall rate"].groupby(["features", "C"])["log_loss"].mean()
    name, C = mean.idxmin()
    return name, float(C)
