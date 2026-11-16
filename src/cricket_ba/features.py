"""Turn parsed deliveries into the modelling table: one row per legal ball."""

from __future__ import annotations

import pandas as pd

PHASES = ("powerplay", "middle", "death")
ROLES = ("top", "middle", "lower", "tail")
# IPL balls faced in earlier matches; the first label is the reference level in models
EXPERIENCE = ("500+", "100-499", "<100")

# Grounds renamed during the IPL's history. Motera (Sardar Patel Stadium) and the
# Narendra Modi Stadium are left apart because the ground was rebuilt.
VENUE_RENAMES = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
}

COLUMNS = [
    "match_id",
    "date",
    "season",
    "innings",
    "over",
    "phase",
    "venue_clean",
    "batter",
    "batter_id",
    "bat_position",
    "role",
    "prior_balls",
    "experience",
    "bowler",
    "bowler_id",
    "runs_total",
    "wicket",
]


def clean_venue(venue: pd.Series) -> pd.Series:
    """Drop the city suffix, tidy punctuation and map renamed grounds."""
    out = venue.str.split(",").str[0].str.replace(".", " ", regex=False)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.replace(VENUE_RENAMES)


def phase(over: pd.Series) -> pd.Series:
    """Powerplay is overs 1-6, middle 7-15, death 16-20 (``over`` is 0-based)."""
    return pd.cut(over, [-1, 5, 14, 19], labels=list(PHASES)).astype(str)


def role(position: pd.Series) -> pd.Series:
    """Batting role from batting position: top 1-3, middle 4-5, lower 6-7, tail 8-11."""
    return pd.cut(position, [0, 3, 5, 7, 11], labels=list(ROLES)).astype(str)


def add_experience(balls: pd.DataFrame) -> pd.DataFrame:
    """Add ``prior_balls`` (legal balls the batter faced in earlier matches) and its band.

    Only matches in ``balls`` count, so in the first IPL season (2008) every batter starts
    at zero.
    """
    per = (
        balls.groupby(["batter_id", "date", "match_id"]).size().rename("n").reset_index()
    ).sort_values(["batter_id", "date", "match_id"])
    per["prior_balls"] = per.groupby("batter_id")["n"].cumsum() - per["n"]
    out = balls.merge(per[["batter_id", "match_id", "prior_balls"]], on=["batter_id", "match_id"])
    band = pd.cut(
        out["prior_balls"], [-1, 99, 499, float("inf")], labels=["<100", "100-499", "500+"]
    )
    out["experience"] = band.astype(str)
    return out


def legal_balls(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Legal balls outside super overs, with ``wicket`` = bowler took the striker's wicket."""
    d = deliveries[deliveries["legal_ball"] & ~deliveries["is_super_over"]].copy()
    d["phase"] = phase(d["over"])
    d["role"] = role(d["bat_position"])
    venues = matches[["match_id"]].assign(venue_clean=clean_venue(matches["venue"]))
    d = d.merge(venues, on="match_id", how="left")
    d["wicket"] = d["bowler_wicket"].astype(int)
    d = add_experience(d)
    return d[COLUMNS].reset_index(drop=True)


def split_by_season(
    balls: pd.DataFrame, test_seasons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = balls["season"].isin(test_seasons)
    return balls[~test].reset_index(drop=True), balls[test].reset_index(drop=True)
