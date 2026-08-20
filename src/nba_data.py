"""Fetch and cache raw NBA data (play-by-play, boxscores, rotations, game lists)."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    boxscoretraditionalv3,
    gamerotation,
    leaguegamefinder,
    playbyplayv3,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _cache_path(kind: str, key: str) -> Path:
    d = RAW_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _fetch_with_retry(endpoint_cls, retries=3, timeout=60, **kwargs):
    for attempt in range(retries):
        try:
            return endpoint_cls(timeout=timeout, **kwargs).get_data_frames()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_season_games(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    path = _cache_path("game_lists", f"{season}_{season_type}".replace(" ", "_"))
    if path.exists():
        return pd.read_json(path, dtype={"GAME_ID": str})
    df = _fetch_with_retry(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        season_type_nullable=season_type,
    )[0]
    df.to_json(path, orient="records")
    return df


def fetch_playbyplay(game_id: str, sleep: float = 0.6) -> pd.DataFrame:
    path = _cache_path("pbp", game_id)
    if path.exists():
        return pd.read_json(path, dtype={"gameId": str, "personId": "Int64", "teamId": "Int64"})
    df = _fetch_with_retry(playbyplayv3.PlayByPlayV3, game_id=game_id)[0]
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_boxscore_players(game_id: str, sleep: float = 0.6) -> pd.DataFrame:
    path = _cache_path("boxscore", game_id)
    if path.exists():
        return pd.read_json(path, dtype={"gameId": str, "personId": "Int64", "teamId": "Int64"})
    df = _fetch_with_retry(boxscoretraditionalv3.BoxScoreTraditionalV3, game_id=game_id)[0]
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_game_rotation(game_id: str, sleep: float = 0.6) -> pd.DataFrame:
    """One row per player per stint, with the NBA's own official
    IN_TIME_REAL / OUT_TIME_REAL (tenths of a second of elapsed game time).
    Two team frames get concatenated into one DataFrame here."""
    path = _cache_path("rotation", game_id)
    if path.exists():
        return pd.read_json(path, dtype={"GAME_ID": str, "PERSON_ID": "Int64", "TEAM_ID": "Int64"})
    dfs = _fetch_with_retry(gamerotation.GameRotation, game_id=game_id)
    df = pd.concat(dfs, ignore_index=True)
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df
