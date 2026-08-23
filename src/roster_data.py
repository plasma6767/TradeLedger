"""Fetch and cache current NBA team rosters. Basketball-Reference's season
file (bref_data.py) deliberately keeps only the combined-season row for
traded players, so it has no reliable "who's on this roster right now"
signal - this module is the actual source of truth for that, pulled fresh
from nba_api's own current roster listing."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "rosters"


def all_team_ids() -> list[int]:
    return [t["id"] for t in teams.get_teams()]


def fetch_team_roster(team_id: int, season: str, sleep: float = 0.6) -> pd.DataFrame:
    path = RAW_DIR / f"{season}_{team_id}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_json(path, dtype={"PLAYER_ID": "Int64", "TeamID": "Int64"})

    df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=60).get_data_frames()[0]
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_all_rosters(season: str) -> pd.DataFrame:
    """One row per player currently on a roster league-wide, with the team
    they're actually on right now."""
    frames = [fetch_team_roster(team_id, season) for team_id in all_team_ids()]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_all_rosters("2025-26")
    print(f"{len(df)} rostered players across {df['TeamID'].nunique()} teams")
    print(df[["PLAYER", "PLAYER_ID", "TeamID"]].head(10).to_string(index=False))
