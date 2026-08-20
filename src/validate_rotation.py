"""Validate GameRotation as the source of truth for lineups: check that at
every moment in the game, each team has exactly 5 players on the floor,
using the NBA's own official check-in/check-out timestamps directly."""

from nba_data import fetch_game_rotation, fetch_season_games


def check_game(game_id):
    df = fetch_game_rotation(game_id)
    issues = []

    for team_id, team_df in df.groupby("TEAM_ID"):
        team = team_df["TEAM_NAME"].iloc[0]
        boundaries = sorted(set(team_df["IN_TIME_REAL"]) | set(team_df["OUT_TIME_REAL"]))
        game_end = team_df["OUT_TIME_REAL"].max()

        for t in boundaries:
            if t >= game_end:
                continue
            on_court = team_df[(team_df["IN_TIME_REAL"] <= t) & (team_df["OUT_TIME_REAL"] > t)]
            if len(on_court) != 5:
                issues.append(f"{team} at t={t}: {len(on_court)} players on court")

    return issues


if __name__ == "__main__":
    games = fetch_season_games("2025-26")["GAME_ID"].drop_duplicates().head(10).tolist()

    clean = 0
    for gid in games:
        issues = check_game(gid)
        status = "CLEAN" if not issues else f"{len(issues)} issues (e.g. {issues[0]})"
        print(f"  {gid}: {status}")
        clean += not issues
    print(f"\n{clean}/{len(games)} clean")
