"""Rebuild every result table from the raw Cricsheet JSON.

    python -m cricket_ba.pipeline            # or: make results

Writes to ``results/`` and saves the posterior to ``models/hierarchical.nc``. MCMC takes
about 15-20 minutes on 4 cores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from cricket_ba import baseline, hierarchical
from cricket_ba.evaluate import (
    difference_interval,
    interval_coverage,
    match_bootstrap,
    scores,
)
from cricket_ba.features import legal_balls, split_by_season
from cricket_ba.parse import parse_dir

SEED = 20261112
SEASONS = range(2008, 2026)
TEST_SEASONS = [2024, 2025]
VAL_SEASONS = [2019, 2020, 2021, 2022, 2023]
BANDS = [-1, 0, 99, 499, np.inf]
BAND_LABELS = ["unseen", "1-99", "100-499", "500+"]


def run(json_dir: Path, out: Path, models: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    matches, deliveries = parse_dir(json_dir, SEASONS)
    balls = legal_balls(matches, deliveries)
    train, test = split_by_season(balls, TEST_SEASONS)
    print(f"train {len(train):,} balls, test {len(test):,} balls")

    # ---- baseline -------------------------------------------------------------------
    cv = baseline.time_cv(train, VAL_SEASONS)
    name, C = baseline.choose(cv)
    feats = baseline.FEATURE_SETS[name]
    lr = baseline.logistic(feats, C).fit(train[feats], train["wicket"])
    pred = test[["match_id", "batter_id", "bowler_id", "phase", "wicket"]].copy()
    pred["p_overall"] = train["wicket"].mean()
    pred["p_logistic"] = lr.predict_proba(test[feats])[:, 1]
    cv.to_csv(out / "baseline_cv.csv", index=False)
    print(f"baseline: {name}, C = {C}")
    if "venue_clean" in feats:
        raise NotImplementedError("cross-validation chose venue; the package model omits it")

    # ---- hierarchical model -----------------------------------------------------------
    idx = hierarchical.make_index(train)
    cells = hierarchical.aggregate(train, idx)
    t0 = time.time()
    idata = hierarchical.fit(cells, idx, SEED)
    minutes = (time.time() - t0) / 60
    summ = az.summary(
        idata, var_names=["alpha", "sigma_bat", "sigma_bowl", "z_bat", "z_bowl"], round_to=None
    )
    diag = {
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "sampling_minutes": round(minutes, 2),
    }
    print("diagnostics:", diag)
    params = az.summary(idata, var_names=["alpha", "sigma_bat", "sigma_bowl"], round_to=None)
    params.to_csv(out / "hierarchical_parameters.csv")

    draws = hierarchical.Draws.from_idata(idata)
    tcells = hierarchical.aggregate(test)
    p_draws = hierarchical.predict_draws(tcells, idx, draws, SEED)
    tcells["p_hier"] = p_draws.mean(1)
    keys = hierarchical.KEYS
    pred = pred.merge(tcells[keys + ["p_hier"]], on=keys, how="left")

    # ---- test metrics -----------------------------------------------------------------
    cols = {"p_overall": "overall rate", "p_logistic": "logistic L2", "p_hier": "hierarchical"}
    metrics = pd.DataFrame({v: scores(pred["wicket"], pred[k]) for k, v in cols.items()}).T
    metrics["mean predicted"] = [pred[k].mean() for k in cols]
    metrics["observed rate"] = pred["wicket"].mean()
    metrics.to_csv(out / "test_metrics.csv")

    boot = match_bootstrap(pred, list(cols), n_boot=1000, seed=SEED)
    difference_interval(boot, "p_hier", "p_logistic").to_csv(out / "test_metric_differences.csv")

    # ---- by how much training data each player had -------------------------------------
    bat_n, bowl_n = train["batter_id"].value_counts(), train["bowler_id"].value_counts()
    rows = []
    for who, col, counts in [("batter", "batter_id", bat_n), ("bowler", "bowler_id", bowl_n)]:
        band = pd.cut(pred[col].map(counts).fillna(0), BANDS, labels=BAND_LABELS)
        for lab in BAND_LABELS:
            sub = pred[band == lab]
            b = match_bootstrap(sub, ["p_logistic", "p_hier"], n_boot=500, seed=SEED)
            dd = b[("p_hier", "log_loss")] - b[("p_logistic", "log_loss")]
            rows.append(
                {
                    "player": who,
                    "training balls": lab,
                    "test balls": len(sub),
                    "log loss logistic": scores(sub["wicket"], sub["p_logistic"])["log_loss"],
                    "log loss hierarchical": scores(sub["wicket"], sub["p_hier"])["log_loss"],
                    "difference": dd.median(),
                    "diff 2.5%": dd.quantile(0.025),
                    "diff 97.5%": dd.quantile(0.975),
                }
            )
    pd.DataFrame(rows).to_csv(out / "test_by_training_balls.csv", index=False)

    # ---- interval coverage on the test seasons -----------------------------------------
    r = np.random.default_rng(SEED)
    pick = r.choice(p_draws.shape[1], size=2000, replace=False)
    lr_cell = pred.groupby(keys, observed=True)["p_logistic"].mean().reset_index()
    tc = tcells.merge(lr_cell, on=keys, how="left")
    n = tc["n"].to_numpy()[:, None]
    sims = {
        "hierarchical": r.binomial(n, p_draws[:, pick]),
        "logistic plug-in": r.binomial(
            n, tc["p_logistic"].to_numpy()[:, None], size=(len(tc), len(pick))
        ),
    }
    rows = []
    for who, col in [("batters", "batter_id"), ("bowlers", "bowler_id")]:
        nb = tc.groupby(col)["n"].sum()
        players = pd.Index(nb[nb >= 60].index)
        codes = players.get_indexer(tc[col])
        m = codes >= 0
        obs = np.bincount(codes[m], weights=tc["k"].to_numpy()[m], minlength=len(players))
        for model_name, sim in sims.items():
            tot = np.zeros((len(players), sim.shape[1]))
            np.add.at(tot, codes[m], sim[m])
            rows.append(
                {
                    "players": who,
                    "model": model_name,
                    "n players": len(players),
                    **interval_coverage(tot, obs, seed=SEED),
                }
            )
    pd.DataFrame(rows).to_csv(out / "interval_coverage.csv", index=False)

    # ---- example batters ----------------------------------------------------------------
    oh = lr[0].named_transformers_["onehot"]
    coef = pd.Series(lr[-1].coef_[0], index=oh.get_feature_names_out(feats))
    lr_bat = expit(
        lr[-1].intercept_[0]
        + coef["phase_middle"]
        + coef[[f"batter_id_{b}" for b in idx.batters]].to_numpy()
    )
    hb = expit(draws.alpha[1][None, :] + draws.bat)
    est = pd.DataFrame(
        {
            "batter_id": idx.batters,
            "train_balls": bat_n.reindex(idx.batters).to_numpy(),
            "train_wickets": train.groupby("batter_id")["wicket"].sum().reindex(idx.batters),
            "logistic": lr_bat,
            "hier_mean": hb.mean(1),
            "hier_10": np.quantile(hb, 0.1, axis=1),
            "hier_90": np.quantile(hb, 0.9, axis=1),
        }
    ).reset_index(drop=True)
    est["raw_rate"] = est["train_wickets"] / est["train_balls"]
    names = train.drop_duplicates("batter_id").set_index("batter_id")["batter"]
    est["name"] = est["batter_id"].map(names)
    est.to_csv(out / "batter_estimates.csv", index=False)

    with open(out / "hierarchical_run.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "sampler": hierarchical.SAMPLER,
                "diagnostics": diag,
                "baseline": {"features": feats, "C": C},
                "n_cells": len(cells),
                "n_train_balls": len(train),
                "n_test_balls": len(test),
                "pymc": pm.__version__,
                "arviz": az.__version__,
            },
            f,
            indent=2,
        )
    idata.to_netcdf(models / "hierarchical.nc")
    print(metrics.round(5).to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json-dir", type=Path, default=Path("cricket_data/ipl_json"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--models", type=Path, default=Path("models"))
    a = p.parse_args(argv)
    run(a.json_dir, a.out, a.models)


if __name__ == "__main__":
    main()
