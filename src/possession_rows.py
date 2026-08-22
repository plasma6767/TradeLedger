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
    """Some games have genuinely corrupted GameRotation data - a player
    with two overlapping stints, recorded as on the court twice at once.
    Check for that directly (per player) rather than only noticing it
    indirectly as a bad lineup count downstream."""
    for _, player_stints in rot_df.groupby("PERSON_ID"):
        stints = player_stints.sort_values("IN_TIME_REAL")
        ins = stints["IN_TIME_REAL"].tolist()
        outs = stints["OUT_TIME_REAL"].tolist()
        for i in range(len(ins) - 1):
            if ins[i + 1] < outs[i]:
                return False
    return True


def build_game_rows(game_id):
    pbp = fetch_playbyplay(game_id)
    rot = fetch_game_rotation(game_id)

    if not check_rotation_consistency(rot):
        raise ValueError(f"{game_id}: GameRotation has overlapping stints - corrupted source data, excluding game")

    sub_boundaries = sorted(set(rot["IN_TIME_REAL"]) | set(rot["OUT_TIME_REAL"]))
    possessions, team_ids = detect_possessions(pbp, sub_boundaries=sub_boundaries)

    rows = []
    for p in possessions:
        offense_id = p["offense_team_id"]
        defense_id = team_ids[0] if offense_id == team_ids[1] else team_ids[1]
        mid = (p["start"] + p["end"]) / 2

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
