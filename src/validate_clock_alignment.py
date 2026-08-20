"""Confirm play-by-play's period+countdown clock and GameRotation's
elapsed-game-time clock actually agree once converted to the same frame.

Method: for every substitution event in play-by-play, convert its
timestamp to elapsed tenths-of-a-second, then check that the outgoing
player has a stint in GameRotation ending at that same timestamp.
"""

import re

from nba_data import fetch_game_rotation, fetch_playbyplay, fetch_season_games

REGULATION_PERIOD_TENTHS = 7200  # 12 min
OT_PERIOD_TENTHS = 3000  # 5 min


def clock_to_remaining_tenths(clock_str):
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    minutes, seconds = int(m.group(1)), float(m.group(2))
    return round((minutes * 60 + seconds) * 10)


def pbp_elapsed_tenths(period, clock_str):
    remaining = clock_to_remaining_tenths(clock_str)
    if period <= 4:
        return (period - 1) * REGULATION_PERIOD_TENTHS + (REGULATION_PERIOD_TENTHS - remaining)
    return (
        4 * REGULATION_PERIOD_TENTHS
        + (period - 5) * OT_PERIOD_TENTHS
        + (OT_PERIOD_TENTHS - remaining)
    )


def check_game(game_id):
    pbp = fetch_playbyplay(game_id)
    rot = fetch_game_rotation(game_id)

    subs = pbp[pbp["actionType"] == "Substitution"]
    diffs = []

    for _, row in subs.iterrows():
        elapsed = pbp_elapsed_tenths(row["period"], row["clock"])
        out_id = row["personId"]

        player_stints = rot[rot["PERSON_ID"] == out_id]
        if player_stints.empty:
            continue

        closest_out_time = player_stints["OUT_TIME_REAL"].sub(elapsed).abs().idxmin()
        best_out_time = player_stints.loc[closest_out_time, "OUT_TIME_REAL"]
        diffs.append(elapsed - best_out_time)

    return diffs


if __name__ == "__main__":
    games = fetch_season_games("2025-26")["GAME_ID"].drop_duplicates().head(5).tolist()

    all_diffs = []
    for gid in games:
        diffs = check_game(gid)
        all_diffs.extend(diffs)
        print(f"  {gid}: {len(diffs)} subs checked, max abs diff = {max(abs(d) for d in diffs)} tenths")

    print()
    print(f"total subs checked: {len(all_diffs)}")
    print(f"max abs diff overall: {max(abs(d) for d in all_diffs)} tenths of a second")
    print(f"exact matches (diff==0): {sum(1 for d in all_diffs if d == 0)}/{len(all_diffs)}")
