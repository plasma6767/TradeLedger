"""Run the possession-row builder across every game in a season, saving
each game's results as soon as they're done (so a crash midway doesn't
lose earlier work), then combine everything into one dataset file."""

import sys
import traceback
from pathlib import Path

import pandas as pd

from nba_data import fetch_season_games
from possession_rows import build_game_rows

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PER_GAME_DIR = PROCESSED_DIR / "by_game"


def process_game(game_id):
    out_path = PER_GAME_DIR / f"{game_id}.parquet"
    if out_path.exists():
        return "skipped"

    rows = build_game_rows(game_id)
    df = pd.DataFrame(rows)
    df["offense_players"] = df["offense_players"].apply(lambda ids: ",".join(map(str, ids)))
    df["defense_players"] = df["defense_players"].apply(lambda ids: ",".join(map(str, ids)))
    PER_GAME_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return "built"


def run(season="2025-26"):
    games = fetch_season_games(season)["GAME_ID"].drop_duplicates().tolist()
    print(f"{len(games)} games in {season}")

    built, skipped, failed = 0, 0, []
    for i, gid in enumerate(games, 1):
        try:
            status = process_game(gid)
            built += status == "built"
            skipped += status == "skipped"
        except Exception as e:
            failed.append((gid, str(e)))
            print(f"  FAILED {gid}: {e}")
            traceback.print_exc(file=sys.stderr)
        if i % 50 == 0:
            print(f"  ...{i}/{len(games)} (built={built}, skipped={skipped}, failed={len(failed)})")

    print(f"\nDone: built={built}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("Failed games:")
        for gid, err in failed:
            print(f"  {gid}: {err}")

    per_game_files = sorted(PER_GAME_DIR.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(f) for f in per_game_files], ignore_index=True)
    combined_path = PROCESSED_DIR / f"possessions_{season}.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nCombined dataset: {len(combined)} rows across {len(per_game_files)} games -> {combined_path}")


if __name__ == "__main__":
    run()
