"""Join the outputs of Phases 1-3 into one table, one row per player:
current team, age, real RAPM, real current BPM, projected BPM (+1y/+2y/+3y
with bands), and contract surplus score. This is the table every Phase 5
view reads from.

Surplus value (surplus_value.py) and the BPM/projection report
(player_report.py) are both keyed by Basketball-Reference player names -
they read the same underlying season file, so their `player` columns are
already the identical string and merge directly. RAPM (rapm.py) and
current rosters (roster_data.py) are both keyed by nba_api's own numeric
player_id instead, with no shared id against the BR-name-keyed side. This
module's only real job is bridging that gap: matching BR names to an
nba_api id via the same name_matching helper surplus_value.py and
player_report.py already use internally to pull in RAPM (re-run here
since neither of those outputs keeps the matched id around), then using
that id to join in the current-roster team."""

from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as nba_teams

from src.name_matching import match_names
from src.player_report import build_report
from src.roster_data import fetch_all_rosters
from src.surplus_value import build_surplus_table

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CURRENT_SEASON = "2025-26"


def team_abbreviations() -> dict[int, str]:
    return {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}


def add_nba_player_ids(table: pd.DataFrame, rapm: pd.DataFrame) -> pd.DataFrame:
    """Matches `table`'s `player` column (Basketball-Reference names)
    against `rapm`'s `name`/`player_id` (nba_api names/ids). Unmatched
    players are reported and dropped, the same convention every other
    module in this project already follows, since nothing downstream can
    join without a real nba_api id."""
    matched, unmatched = match_names(list(table["player"]), list(rapm["name"]))
    if unmatched:
        print(f"WARNING: {len(unmatched)} players couldn't be matched to an nba_api id: {unmatched}")

    table = table.copy()
    table["_rapm_name"] = table["player"].map(matched)
    table = table.dropna(subset=["_rapm_name"])
    id_by_name = rapm.drop_duplicates(subset="name").set_index("name")["player_id"]
    table["nba_player_id"] = table["_rapm_name"].map(id_by_name)
    return table.drop(columns=["_rapm_name"])


def add_current_team(table: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Joins in each player's current team abbreviation. `rosters` is one
    row per player with `PLAYER_ID` and `team` columns. A player not on
    any current roster (e.g. between the contract data snapshot and now)
    keeps a missing team rather than being dropped - he still has real
    surplus/RAPM/BPM numbers worth showing."""
    rosters = rosters.drop_duplicates(subset="PLAYER_ID")
    merged = table.merge(
        rosters[["PLAYER_ID", "team"]], left_on="nba_player_id", right_on="PLAYER_ID", how="left"
    )
    return merged.drop(columns=["PLAYER_ID"])


def build_player_universe(season: str = CURRENT_SEASON) -> pd.DataFrame:
    surplus = build_surplus_table()
    report = build_report()
    rapm = pd.read_csv(PROCESSED_DIR / f"rapm_{season}.csv")

    # age/rapm are already on the surplus table from the same source row -
    # only pull the BPM/projection columns report adds on top
    report_cols = ["player"] + [c for c in report.columns if c not in ("player", "age", "rapm")]
    universe = surplus.merge(report[report_cols], on="player", how="left")

    universe = add_nba_player_ids(universe, rapm)

    rosters = fetch_all_rosters(season)
    rosters = rosters.copy()
    rosters["team"] = rosters["TeamID"].map(team_abbreviations())
    universe = add_current_team(universe, rosters)

    return universe.reset_index(drop=True)


if __name__ == "__main__":
    universe = build_player_universe()
    pd.set_option("display.width", 200)

    print(f"\n{len(universe)} players in the universe\n")
    display_cols = ["player", "team", "age", "rapm", "current_bpm", "surplus_score"]
    print("Top 15 by surplus score:")
    print(universe.sort_values("surplus_score", ascending=False)[display_cols].head(15).to_string(index=False))

    out_path = PROCESSED_DIR / "player_universe.csv"
    universe.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")
