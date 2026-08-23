"""Fetch and cache raw Basketball-Reference season advanced-stats tables
(BPM, VORP, Age per player per season). Same cache-first pattern as
nba_data.py: check the cache, only hit the network if it's missing."""

import io
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import html as lxml_html

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bref_advanced"

# Basketball-Reference blocks requests that look like a generic scraper
# (default python-requests / no headers) - a realistic browser UA avoids that.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def season_to_bref_year(season: str) -> int:
    """'2011-12' -> 2012. Basketball-Reference names each season's page
    after the year it ends in."""
    start_year = int(season.split("-")[0])
    return start_year + 1


def fetch_season_advanced(season: str, sleep: float = 3.0) -> pd.DataFrame:
    """One row per player per season: Age, BPM, VORP, etc. Players traded
    mid-season get one row per team plus a 'TOT' (total) row - we keep only
    TOT for those players so they aren't triple-counted."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{season}.json"
    if path.exists():
        return pd.read_json(path)

    year = season_to_bref_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "advanced"})
    df = tables[0]

    # pandas.read_html only grabs visible cell text, but player *names*
    # aren't unique (BR has had multiple different players share a name
    # in this era) - pull BR's real per-player ID out of the raw HTML so
    # season-to-season joins are keyed on the actual person, not a string.
    tree = lxml_html.fromstring(resp.text)
    rows = tree.xpath('//table[@id="advanced"]/tbody/tr')
    player_ids = [row.xpath('./td[@data-stat="name_display"]/@data-append-csv') for row in rows]
    player_ids = [ids[0] if ids else None for ids in player_ids]
    df["player_id"] = player_ids[: len(df)]

    # drop the repeated header rows BR embeds every ~20 rows in the raw table
    df = df[df["Player"] != "Player"].copy()

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["BPM"] = pd.to_numeric(df["BPM"], errors="coerce")
    df["VORP"] = pd.to_numeric(df["VORP"], errors="coerce")
    df = df.dropna(subset=["Age", "BPM", "VORP"])

    has_multi_team = df.duplicated(subset=["player_id"], keep=False) & (df["Team"] != "TOT")
    traded_players = set(df.loc[has_multi_team, "player_id"])
    df = df[~(df["player_id"].isin(traded_players) & (df["Team"] != "TOT"))]

    df["season"] = season
    df = df.reset_index(drop=True)
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_seasons_advanced(seasons: list[str]) -> pd.DataFrame:
    frames = [fetch_season_advanced(s) for s in seasons]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(2011, 2026)]
    print(f"Fetching {len(seasons)} seasons: {seasons[0]} through {seasons[-1]}")
    df = fetch_seasons_advanced(seasons)
    print(f"{len(df)} player-seasons total")
    print(df[["Player", "season", "Age", "BPM", "VORP"]].head(10).to_string())
