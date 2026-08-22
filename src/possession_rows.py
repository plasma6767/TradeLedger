"""Build the final training rows: one row per possession (or per sub-segment,
if a substitution falls inside a possession), with the exact 10-man lineup
and net points, ready to feed the RAPM regression."""

from nba_data import fetch_game_rotation, fetch_playbyplay
from possessions import detect_possessions


def lineup_at(rot_df, team_id, t):
    team_df = rot_df[rot_df["TEAM_ID"] == team_id]
    on = team_df[(team_df["IN_TIME_REAL"] <= t) & (team_df["OUT_TIME_REAL"] > t)]
    return set(on["PERSON_ID"])


def check_rotation_consistency(rot_df):
    """Some games have genuinely corrupted GameRotation data. Two known
    failure modes, both checked here:
    1. A player with two overlapping stints - recorded on the court twice
       at once.
    2. Missing stints scattered through the game - the union of all
       players' time covers the full game with no gaps, but at some
       individual moments fewer than 5 players are actually on the floor,
       because some player's stint for that stretch just isn't in the
       data. Checking this requires walking every boundary and counting
       who's on the floor at each one, same as the original
       validate_rotation.py check - that full check just hadn't been
       carried over into this production path, which is exactly how this
       slipped through initially."""
    for _, player_stints in rot_df.groupby("PERSON_ID"):
        stints = player_stints.sort_values("IN_TIME_REAL")
        ins = stints["IN_TIME_REAL"].tolist()
        outs = stints["OUT_TIME_REAL"].tolist()
        for i in range(len(ins) - 1):
            if ins[i + 1] < outs[i]:
                return False

    for team_id, team_df in rot_df.groupby("TEAM_ID"):
        boundaries = sorted(set(team_df["IN_TIME_REAL"]) | set(team_df["OUT_TIME_REAL"]))
        game_end = team_df["OUT_TIME_REAL"].max()
        for t in boundaries:
            if t >= game_end:
                continue
            on_court = team_df[(team_df["IN_TIME_REAL"] <= t) & (team_df["OUT_TIME_REAL"] > t)]
            if len(on_court) != 5:
                return False

    return True


def build_game_rows(game_id):
    pbp = fetch_playbyplay(game_id)
    rot = fetch_game_rotation(game_id)

    if not check_rotation_consistency(rot):
        raise ValueError(f"{game_id}: GameRotation is inconsistent - corrupted source data, excluding game")

    sub_boundaries = sorted(set(rot["IN_TIME_REAL"]) | set(rot["OUT_TIME_REAL"]))
    possessions, team_ids = detect_possessions(pbp, sub_boundaries=sub_boundaries)

    rows = []
    for p in possessions:
        offense_id = p["offense_team_id"]
        defense_id = team_ids[0] if offense_id == team_ids[1] else team_ids[1]
        # Query just before the midpoint, not exactly at it: for a
        # zero-width window sitting at the literal end of the game (e.g. a
        # buzzer-beater at start=end=28800), querying exactly at 28800
        # finds no lineup at all, since no stint extends past the buzzer -
        # even though real players (and real points) existed right up to
        # that instant. 0.5 is smaller than the finest real time
        # resolution (whole tenths), so this can never cross into a
        # different, wrong window for any normal (non-zero-width) case.
        mid = max(0, (p["start"] + p["end"]) / 2 - 0.5)

        offense_players = lineup_at(rot, offense_id, mid)
        defense_players = lineup_at(rot, defense_id, mid)

        if len(offense_players) != 5 or len(defense_players) != 5:
            # Boundary artifact: a zero-width window sitting exactly at the
            # game's final tick (e.g. start=end=28800, the literal instant
            # the game ends) - no player's rotation stint covers that exact
            # instant, and no real action is possible there. Always 0
            # points by construction; safe to drop.
            assert p["offense_points"] == 0 and p["defense_points"] == 0
            continue

        rows.append(
            {
                "game_id": game_id,
                "start": p["start"],
                "end": p["end"],
                "offense_team_id": offense_id,
                "defense_team_id": defense_id,
                "offense_players": sorted(offense_players),
                "defense_players": sorted(defense_players),
                "offense_points": p["offense_points"],
                "defense_points": p["defense_points"],
                "points": p["points"],
            }
        )
    return rows
