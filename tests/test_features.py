from pathlib import Path

import pandas as pd

from cricket_ba.features import (
    add_experience,
    clean_venue,
    legal_balls,
    phase,
    role,
    split_by_season,
)
from cricket_ba.parse import parse_files

FILES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def test_clean_venue():
    raw = pd.Series(
        [
            "Wankhede Stadium, Mumbai",
            "Wankhede Stadium",
            "M.Chinnaswamy Stadium",
            "M Chinnaswamy Stadium, Bengaluru",
            "Feroz Shah Kotla",
            "MA Chidambaram Stadium, Chepauk, Chennai",
        ]
    )
    assert list(clean_venue(raw)) == [
        "Wankhede Stadium",
        "Wankhede Stadium",
        "M Chinnaswamy Stadium",
        "M Chinnaswamy Stadium",
        "Arun Jaitley Stadium",
        "MA Chidambaram Stadium",
    ]


def test_phase_boundaries():
    # over is 0-based: overs 0-5 powerplay, 6-14 middle, 15-19 death
    got = phase(pd.Series([0, 5, 6, 14, 15, 19]))
    assert list(got) == ["powerplay", "powerplay", "middle", "middle", "death", "death"]


def test_role_boundaries():
    got = role(pd.Series([1, 3, 4, 5, 6, 7, 8, 11]))
    assert list(got) == ["top", "top", "middle", "middle", "lower", "lower", "tail", "tail"]


def test_legal_balls_table():
    m, d = parse_files(FILES)
    b = legal_balls(m, d)
    expected = d["legal_ball"] & ~d["is_super_over"]
    assert len(b) == expected.sum()
    assert b["wicket"].sum() == d.loc[expected, "bowler_wicket"].sum()
    assert b["phase"].isin(["powerplay", "middle", "death"]).all()
    assert b["venue_clean"].notna().all()
    assert b["role"].isin(["top", "middle", "lower", "tail"]).all()


def test_split_by_season():
    balls = pd.DataFrame({"season": [2022, 2023, 2024, 2025], "x": range(4)})
    tr, te = split_by_season(balls, [2024, 2025])
    assert list(tr["season"]) == [2022, 2023] and list(te["season"]) == [2024, 2025]


def test_experience_counts_only_earlier_matches():
    balls = pd.DataFrame(
        {
            "batter_id": ["a", "a", "a", "a", "b"],
            "date": pd.to_datetime(["2020-04-01"] * 2 + ["2020-04-05"] * 2 + ["2020-04-05"]),
            "match_id": [1, 1, 2, 2, 2],
        }
    )
    out = add_experience(balls).sort_values(["batter_id", "match_id"])
    assert list(out["prior_balls"]) == [0, 0, 2, 2, 0]
    assert set(out["experience"]) == {"<100"}
