from pathlib import Path

import pandas as pd

from cricket_ba.features import clean_venue, legal_balls, phase, split_by_season
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


def test_legal_balls_table():
    m, d = parse_files(FILES)
    b = legal_balls(m, d)
    expected = d["legal_ball"] & ~d["is_super_over"]
    assert len(b) == expected.sum()
    assert b["wicket"].sum() == d.loc[expected, "bowler_wicket"].sum()
    assert b["phase"].isin(["powerplay", "middle", "death"]).all()
    assert b["venue_clean"].notna().all()


def test_split_by_season():
    balls = pd.DataFrame({"season": [2022, 2023, 2024, 2025], "x": range(4)})
    tr, te = split_by_season(balls, [2024, 2025])
    assert list(tr["season"]) == [2022, 2023] and list(te["season"]) == [2024, 2025]
