"""Turn parsed deliveries into the modelling table: one row per legal ball."""

from __future__ import annotations

import pandas as pd

PHASES = ("powerplay", "middle", "death")

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


def legal_balls(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Legal balls outside super overs, with ``wicket`` = bowler took the striker's wicket."""
    d = deliveries[deliveries["legal_ball"] & ~deliveries["is_super_over"]].copy()
    d["phase"] = phase(d["over"])
    venues = matches[["match_id"]].assign(venue_clean=clean_venue(matches["venue"]))
    d = d.merge(venues, on="match_id", how="left")
    d["wicket"] = d["bowler_wicket"].astype(int)
    return d[COLUMNS].reset_index(drop=True)


def split_by_season(
    balls: pd.DataFrame, test_seasons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = balls["season"].isin(test_seasons)
    return balls[~test].reset_index(drop=True), balls[test].reset_index(drop=True)
