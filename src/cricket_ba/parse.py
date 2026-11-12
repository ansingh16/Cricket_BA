"""Parse Cricsheet IPL match files (JSON) into tidy tables.

One row per delivery, plus one row per match. The format is documented at
https://cricsheet.org/format/json/ . Facts this parser relies on, checked against the
1,169 IPL files on disk:

* the match id is the file name; ``info.registry`` holds only ``people``;
* innings and ball numbers are not stored, they are the position in the lists;
* a delivery is a legal ball unless its ``extras`` contain ``wides`` or ``noballs``
  (a penalty-only record does count as a ball);
* super-over innings carry ``"super_over": true``;
* ``season`` is a label such as ``"2007/08"`` or ``"2020/21"``; the calendar year of the
  match date is the IPL season.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Dismissals credited to the bowler. Each of these dismisses the striker.
BOWLER_WICKET_KINDS = frozenset(
    {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
)
# Recorded in the wickets list but not a dismissal: the batter may return.
NOT_A_DISMISSAL = frozenset({"retired hurt", "retired not out"})

EXTRA_TYPES = ("wides", "noballs", "byes", "legbyes", "penalty")


def load_match(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_info(data: dict, match_id: int) -> dict:
    info = data["info"]
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    teams = info["teams"]
    return {
        "match_id": match_id,
        "date": pd.Timestamp(info["dates"][0]),
        "season": int(info["dates"][0][:4]),
        "season_label": str(info["season"]),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": info.get("toss", {}).get("winner"),
        "toss_decision": info.get("toss", {}).get("decision"),
        "winner": outcome.get("winner"),
        "result": outcome.get("result", "win" if "winner" in outcome else None),
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "method": outcome.get("method"),
        "eliminator": outcome.get("eliminator"),
        "n_innings": len(data.get("innings", [])),
        "data_version": data["meta"]["data_version"],
    }


def match_deliveries(data: dict, match_id: int) -> list[dict]:
    """Return one dict per delivery, in the order bowled."""
    info = data["info"]
    people = info.get("registry", {}).get("people", {})
    teams = info["teams"]
    date = pd.Timestamp(info["dates"][0])
    rows: list[dict] = []

    for inn_idx, inn in enumerate(data.get("innings", []), start=1):
        batting = inn["team"]
        bowling = teams[1] if batting == teams[0] else teams[0]
        super_over = bool(inn.get("super_over", False))
        legal_in_innings = 0

        for over in inn.get("overs", []):
            over_no = over["over"]  # 0-based in the JSON
            legal_in_over = 0
            for d_idx, d in enumerate(over["deliveries"], start=1):
                extras = d.get("extras", {})
                legal = not ({"wides", "noballs"} & extras.keys())
                if legal:
                    legal_in_over += 1
                    legal_in_innings += 1

                wickets = d.get("wickets", [])
                dismissals = [w for w in wickets if w["kind"] not in NOT_A_DISMISSAL]
                first = wickets[0] if wickets else {}
                bowler_wicket = any(
                    w["kind"] in BOWLER_WICKET_KINDS and w["player_out"] == d["batter"]
                    for w in wickets
                )

                row = {
                    "match_id": match_id,
                    "date": date,
                    "season": date.year,
                    "venue": info.get("venue"),
                    "batting_team": batting,
                    "bowling_team": bowling,
                    "innings": inn_idx,
                    "is_super_over": super_over,
                    "over": over_no,
                    "delivery_in_over": d_idx,
                    "legal_ball": legal,
                    "legal_ball_in_over": legal_in_over if legal else None,
                    "legal_ball_in_innings": legal_in_innings if legal else None,
                    "batter": d["batter"],
                    "bowler": d["bowler"],
                    "non_striker": d["non_striker"],
                    "batter_id": people.get(d["batter"]),
                    "bowler_id": people.get(d["bowler"]),
                    "non_striker_id": people.get(d["non_striker"]),
                    "runs_batter": d["runs"]["batter"],
                    "runs_extras": d["runs"]["extras"],
                    "runs_total": d["runs"]["total"],
                    "extras_type": "+".join(sorted(extras)) or None,
                    **{f"extra_{k}": extras.get(k, 0) for k in EXTRA_TYPES},
                    "n_wickets": len(wickets),
                    "n_dismissals": len(dismissals),
                    "wicket_kind": first.get("kind"),
                    "player_out": first.get("player_out"),
                    "bowler_wicket": bowler_wicket,
                }
                rows.append(row)
    return rows


def parse_files(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse match files into (matches, deliveries) data frames."""
    matches, deliveries = [], []
    for path in sorted(Path(p) for p in paths):
        data = load_match(path)
        match_id = int(path.stem)
        matches.append(match_info(data, match_id))
        deliveries.extend(match_deliveries(data, match_id))

    m = pd.DataFrame(matches).sort_values(["date", "match_id"], ignore_index=True)
    d = pd.DataFrame(deliveries)
    if not d.empty:
        for col in ("legal_ball_in_over", "legal_ball_in_innings"):
            d[col] = d[col].astype("Int64")
        d = d.sort_values(
            ["date", "match_id", "innings", "over", "delivery_in_over"], ignore_index=True
        )
    return m, d


def parse_dir(
    json_dir: str | Path, seasons: Iterable[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every ``*.json`` match file in ``json_dir``, optionally keeping some seasons."""
    m, d = parse_files(Path(json_dir).glob("*.json"))
    if seasons is not None:
        keep = set(seasons)
        m = m[m["season"].isin(keep)].reset_index(drop=True)
        d = d[d["season"].isin(keep)].reset_index(drop=True)
    return m, d


def innings_totals(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Runs, wickets and legal balls per innings, as a scorecard would show them."""
    g = deliveries.groupby(["match_id", "innings"], sort=True)
    return pd.DataFrame(
        {
            "batting_team": g["batting_team"].first(),
            "is_super_over": g["is_super_over"].first(),
            "runs": g["runs_total"].sum(),
            "wickets": g["n_dismissals"].sum(),
            "legal_balls": g["legal_ball"].sum(),
        }
    ).reset_index()


def write_parquet(json_dir: str | Path, out_dir: str | Path, seasons=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, d = parse_dir(json_dir, seasons)
    m.to_parquet(out / "matches.parquet", index=False)
    d.to_parquet(out / "deliveries.parquet", index=False)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Parse Cricsheet IPL JSON into Parquet tables.")
    p.add_argument("--json-dir", default="cricket_data/ipl_json")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--first-season", type=int, default=2008)
    p.add_argument("--last-season", type=int, default=2025)
    a = p.parse_args(argv)
    seasons = range(a.first_season, a.last_season + 1)
    write_parquet(a.json_dir, a.out_dir, seasons)
    print(f"wrote {a.out_dir}/matches.parquet and {a.out_dir}/deliveries.parquet")


if __name__ == "__main__":
    main()
