"""Parser tests on four Cricsheet fixture matches.

Expected values come from fields in the JSON that are independent of the deliveries
(``target``, the result margin, ``miscounted_overs``), so the parser is not checked
against its own output.
"""

from pathlib import Path

import pandas as pd
import pytest

from cricket_ba.parse import innings_totals, load_match, parse_files

FIXTURES = Path(__file__).parent / "fixtures"
FILES = sorted(FIXTURES.glob("*.json"))


@pytest.fixture(scope="module")
def parsed():
    return parse_files(FILES)


@pytest.fixture(scope="module")
def totals(parsed):
    return innings_totals(parsed[1]).set_index(["match_id", "innings"])


def test_identifiers_survive(parsed):
    """Regression test for the October 2025 parser, which lost all three identifiers."""
    matches, d = parsed
    assert set(matches["match_id"]) == {int(f.stem) for f in FILES}
    assert set(d["match_id"]) == set(matches["match_id"])
    assert d["innings"].min() == 1
    assert d["delivery_in_over"].min() == 1
    # each delivery is uniquely identified
    key = ["match_id", "innings", "over", "delivery_in_over"]
    assert not d.duplicated(key).any()


def test_one_row_per_delivery_in_json(parsed):
    _, d = parsed
    for f in FILES:
        data = load_match(f)
        n = sum(len(o["deliveries"]) for inn in data["innings"] for o in inn["overs"])
        assert (d["match_id"] == int(f.stem)).sum() == n


def test_first_innings_total_matches_target(totals):
    """The chasing side's target is the first-innings total plus one (no D/L here)."""
    for f in FILES:
        data = load_match(f)
        mid = int(f.stem)
        for inn_no, inn in enumerate(data["innings"], start=1):
            if "target" in inn and not inn.get("super_over"):
                assert totals.loc[(mid, 1), "runs"] == inn["target"]["runs"] - 1
                # the target says how many overs the chase had
                assert totals.loc[(mid, inn_no), "legal_balls"] <= inn["target"]["overs"] * 6


def test_win_by_wickets_matches_wickets_lost(parsed, totals):
    matches, _ = parsed
    for m in matches.itertuples():
        if pd.notna(m.win_by_wickets):
            assert totals.loc[(m.match_id, 2), "wickets"] == 10 - m.win_by_wickets
            assert totals.loc[(m.match_id, 2), "runs"] >= totals.loc[(m.match_id, 1), "runs"] + 1


def test_six_legal_balls_per_completed_over(parsed):
    """Every over except an innings' last has six legal balls, unless the JSON
    records that the umpire miscounted it."""
    _, d = parsed
    legal = d[d["legal_ball"]].groupby(["match_id", "innings", "over"]).size()
    for f in FILES:
        data = load_match(f)
        mid = int(f.stem)
        for inn_no, inn in enumerate(data["innings"], start=1):
            miscounted = {int(k): v["balls"] for k, v in inn.get("miscounted_overs", {}).items()}
            overs = [o["over"] for o in inn["overs"]]
            for over in overs[:-1]:
                assert legal.loc[(mid, inn_no, over)] == miscounted.get(over, 6)


def test_miscounted_over_has_seven_legal_balls(parsed):
    _, d = parsed
    over = d[(d["match_id"] == 1136564) & (d["innings"] == 2) & (d["over"] == 11)]
    assert over["legal_ball"].sum() == 7
    assert over["legal_ball_in_over"].max() == 7


def test_tie_and_super_over(parsed, totals):
    matches, d = parsed
    m = matches.set_index("match_id").loc[1473469]
    assert m["result"] == "tie" and m["n_innings"] == 4
    assert totals.loc[(1473469, 1), "runs"] == totals.loc[(1473469, 2), "runs"]
    so = d[d["match_id"] == 1473469].groupby("innings")["is_super_over"].first()
    assert list(so) == [False, False, True, True]
    # a super over ends after two wickets
    assert totals.loc[(1473469, 3), "wickets"] == 2


def test_no_result_single_innings(parsed):
    matches, d = parsed
    m = matches.set_index("match_id").loc[1473495]
    assert m["result"] == "no result" and pd.isna(m["winner"])
    assert set(d.loc[d["match_id"] == 1473495, "innings"]) == {1}


def test_penalty_runs_on_a_legal_ball(parsed):
    _, d = parsed
    pen = d[d["extra_penalty"] > 0]
    assert len(pen) == 1
    row = pen.iloc[0]
    assert row["match_id"] == 335988 and row["legal_ball"]
    assert row["runs_total"] == 5 and row["runs_batter"] == 0


def test_wides_and_noballs_are_not_legal(parsed):
    _, d = parsed
    illegal = (d["extra_wides"] > 0) | (d["extra_noballs"] > 0)
    assert (~d.loc[illegal, "legal_ball"]).all()
    assert d.loc[~illegal, "legal_ball"].all()
    assert d.loc[illegal, "legal_ball_in_over"].isna().all()


def test_runs_add_up(parsed):
    _, d = parsed
    assert (d["runs_batter"] + d["runs_extras"] == d["runs_total"]).all()
    extra_cols = [c for c in d.columns if c.startswith("extra_")]
    assert (d[extra_cols].sum(axis=1) == d["runs_extras"]).all()


def test_bowler_wickets(parsed):
    _, d = parsed
    credited = d[d["bowler_wicket"]]
    assert (
        credited["wicket_kind"]
        .isin(["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"])
        .all()
    )
    assert (credited["player_out"] == credited["batter"]).all()
    run_outs = d[d["wicket_kind"] == "run out"]
    assert not run_outs["bowler_wicket"].any()


def test_player_ids_present(parsed):
    _, d = parsed
    assert d[["batter_id", "bowler_id", "non_striker_id"]].notna().all().all()


def test_season_is_calendar_year(parsed):
    matches, _ = parsed
    m = matches.set_index("match_id")
    assert m.loc[335988, "season_label"] == "2007/08"
    assert m.loc[335988, "season"] == 2008


def test_batting_positions(parsed):
    """Openers are 1 and 2; each later batter needs a wicket or retirement to come in."""
    _, d = parsed
    for _, inn in d[~d["is_super_over"]].groupby(["match_id", "innings"]):
        first = inn.iloc[0]
        assert first["bat_position"] == 1
        names = pd.concat([inn["batter"], inn["non_striker"]]).unique()
        assert len(names) <= inn["n_wickets"].sum() + 2
        pos = inn.drop_duplicates("batter").set_index("batter")["bat_position"]
        assert pos.is_unique and pos.between(1, 11).all()
        assert pos.max() <= len(names)
