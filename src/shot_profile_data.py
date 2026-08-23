"""Fetch and cache the two raw inputs to a player's playing-style profile:
shot location (shotchartdetail, one call per player - no bulk mode exists
for this endpoint) and play-type frequency (synergyplaytypes, one call per
play type - this endpoint returns every player at once, so 11 calls
covers the whole league)."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail, synergyplaytypes

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The 11 official Synergy play types (nba_api.stats.library.parameters.PlayType).
PLAY_TYPES = [
    "Transition", "Isolation", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
]


def fetch_player_shots(player_id: int, season: str, sleep: float = 0.6) -> pd.DataFrame:
    """Every shot attempt for one player in one season, with its court
    location and zone. team_id=0 returns the player's shots regardless of
    which team he was on when he took them, which matters for anyone
    traded mid-season."""
    path = RAW_DIR / "shot_charts" / f"{season}_{player_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_json(path, dtype={"PLAYER_ID": "Int64", "TEAM_ID": "Int64"})

    df = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        context_measure_simple="FGA",
        timeout=60,
    ).get_data_frames()[0]
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_playtype_data(play_type: str, season: str, sleep: float = 0.6) -> pd.DataFrame:
    """Every player's usage and efficiency in one play type, league-wide,
    for one season."""
    path = RAW_DIR / "playtypes" / f"{season}_{play_type}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_json(path, dtype={"PLAYER_ID": "Int64", "TEAM_ID": "Int64"})

    df = synergyplaytypes.SynergyPlayTypes(
        player_or_team_abbreviation="P",
        season=season,
        play_type_nullable=play_type,
        type_grouping_nullable="offensive",
        timeout=60,
    ).get_data_frames()[0]
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_all_playtypes(season: str) -> pd.DataFrame:
    """One row per player per play type, league-wide, for one season -
    stack of all PLAY_TYPES calls."""
    frames = [fetch_playtype_data(pt, season) for pt in PLAY_TYPES]
    return pd.concat(frames, ignore_index=True)


def fetch_players_shots(player_ids: list[int], season: str) -> dict[int, pd.DataFrame]:
    """Shot charts for a list of players, keyed by player_id. Resumable:
    each player's file is saved as soon as it's fetched, so a crash partway
    through a full-league pull doesn't lose earlier players."""
    return {player_id: fetch_player_shots(player_id, season) for player_id in player_ids}


if __name__ == "__main__":
    season = "2025-26"
    playtypes = fetch_all_playtypes(season)
    print(f"{len(playtypes)} player-playtype rows across {playtypes['PLAY_TYPE'].nunique()} play types")

    # smoke-test the per-player shot chart fetch on a couple of real players
    # rather than pulling the full league here
    sample = fetch_players_shots([2544, 201939], season)  # LeBron, Curry
    for player_id, shots in sample.items():
        print(f"player {player_id}: {len(shots)} shot attempts")
