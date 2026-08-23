"""Views onto the joined player_universe table (player_universe.py): each
function answers one ranking question, plus a single-player deep dive.
Surplus score, RAPM, BPM, projected BPM, and fit are never blended into
one number here - each view sorts by its own real number and shows the
others as context, the same way player_report.py keeps its three signals
side by side rather than merging them.

The ranking functions are pure pandas in/out with no I/O of their own, so
a future UI can call them directly against a table it already has.
player_detail follows the project's usual fetch/parse split: the real
assembly logic (player_detail) takes already-fetched data and is
testable with no network call; get_player_detail is the thin wrapper that
does the actual fetching."""

import pandas as pd

from src.roster_fit import build_team_fit_table, player_playtype_diet, player_shot_zone_diet
from src.shot_profile_data import PLAY_TYPES, fetch_all_playtypes, fetch_player_shots

CURRENT_SEASON = "2025-26"
PROJECTION_YEARS = (1, 2, 3)


def _percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


def _ordinal(n: float) -> str:
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rank_by_surplus(universe: pd.DataFrame) -> pd.DataFrame:
    """Default GM view: best contract value first. Surplus score has no
    minutes floor (unlike current_bpm/projected BPM below, which
    player_report.py already hides under 500 minutes rather than showing
    an unreliable number) - RAPM's own regularization keeps a small-sample
    player's number from being a wild outlier, but it's still built on
    less data than a starter's, so minutes played is surfaced in the
    reasoning rather than left invisible."""
    table = universe.sort_values("surplus_score", ascending=False).copy()
    table["reasoning"] = [
        f"surplus score {score:+.0f} ({_ordinal(value_pct)} percentile production vs "
        f"{_ordinal(cost_pct)} percentile cost) - {minutes:.0f} min this season"
        for score, value_pct, cost_pct, minutes in zip(
            table["surplus_score"], table["value_percentile"], table["cost_percentile"], table["minutes"]
        )
    ]
    return table.reset_index(drop=True)


def rank_by_rapm(universe: pd.DataFrame) -> pd.DataFrame:
    """Also has no minutes floor - see rank_by_surplus."""
    table = universe.sort_values("rapm", ascending=False).copy()
    percentile = _percentile(table["rapm"])
    table["reasoning"] = [
        f"{rapm:+.1f} RAPM this season ({_ordinal(pct)} percentile leaguewide) - {minutes:.0f} min"
        for rapm, pct, minutes in zip(table["rapm"], percentile, table["minutes"])
    ]
    return table.reset_index(drop=True)


def rank_by_current_bpm(universe: pd.DataFrame) -> pd.DataFrame:
    table = universe.dropna(subset=["current_bpm"]).sort_values("current_bpm", ascending=False).copy()
    percentile = _percentile(table["current_bpm"])
    table["reasoning"] = [
        f"{bpm:+.1f} BPM this season ({_ordinal(pct)} percentile leaguewide)"
        for bpm, pct in zip(table["current_bpm"], percentile)
    ]
    return table.reset_index(drop=True)


def rank_by_projected_bpm(universe: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    if years not in PROJECTION_YEARS:
        raise ValueError(f"years must be one of {PROJECTION_YEARS}, got {years}")
    column, band_column = f"bpm_+{years}y", f"bpm_+{years}y_band"
    table = universe.dropna(subset=[column]).sort_values(column, ascending=False).copy()
    table["reasoning"] = [
        f"projected {years}y BPM {bpm:+.1f} {band}" for bpm, band in zip(table[column], table[band_column])
    ]
    return table.reset_index(drop=True)


def rank_by_fit(universe: pd.DataFrame, team_abbreviation: str, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Candidates ranked by how well they'd complement `team_abbreviation`'s
    current roster (roster_fit.py) - build_team_fit_table already excludes
    the team's own players and sorts by raw_fit_pct, which groups by
    fit_grade tier for free since the grade is a percentile of that same
    column. Surplus score is joined in as context, not blended into the
    sort - a great fit who's badly overpaid still sorts by fit here, with
    the bad contract visible in the reasoning, not hidden by it."""
    fit = build_team_fit_table(team_abbreviation, season)
    context_cols = ["nba_player_id", "team", "age", "rapm", "surplus_score", "minutes"]
    table = fit.merge(universe[context_cols], left_on="player_id", right_on="nba_player_id", how="left")
    table["reasoning"] = [
        f"fit grade {grade} ({raw:.0f}% raw fit for {team_abbreviation}) - "
        + (f"surplus score {surplus:+.0f}" if pd.notna(surplus) else "no contract/surplus data")
        for grade, raw, surplus in zip(table["fit_grade"], table["raw_fit_pct"], table["surplus_score"])
    ]
    return table.drop(columns=["nba_player_id"])


def player_detail(
    universe_row: pd.Series,
    shots: pd.DataFrame,
    playtype_rows: pd.DataFrame,
    all_play_types: list[str] = PLAY_TYPES,
) -> dict:
    """Everything about one player: the joined-table numbers already
    computed (read straight off universe_row, no recompute), plus his
    real shot-zone and play-type diet computed fresh for just him -
    independent of any team, unlike roster_fit.py's team-blended view of
    the same underlying data."""
    projection = [
        {"years": n, "bpm": universe_row.get(f"bpm_+{n}y"), "band": universe_row.get(f"bpm_+{n}y_band")}
        for n in PROJECTION_YEARS
        if pd.notna(universe_row.get(f"bpm_+{n}y"))
    ]
    return {
        "player": universe_row["player"],
        "team": universe_row.get("team"),
        "age": universe_row["age"],
        "minutes": universe_row.get("minutes"),
        "rapm": universe_row["rapm"],
        "current_bpm": universe_row.get("current_bpm"),
        "surplus_score": universe_row["surplus_score"],
        "avg_value": universe_row["avg_value"],
        "avg_cap_pct": universe_row["avg_cap_pct"],
        "years_left": universe_row["years_left"],
        "projection": projection,
        "shot_diet": player_shot_zone_diet(shots).to_dict(),
        "playtype_diet": player_playtype_diet(playtype_rows, all_play_types).to_dict(),
    }


def get_player_detail(universe: pd.DataFrame, nba_player_id: int, season: str = CURRENT_SEASON) -> dict:
    row = universe.loc[universe["nba_player_id"] == nba_player_id].iloc[0]
    shots = fetch_player_shots(nba_player_id, season)
    playtypes = fetch_all_playtypes(season)
    playtype_rows = playtypes[playtypes["PLAYER_ID"] == nba_player_id]
    return player_detail(row, shots, playtype_rows)
