"""Detect real possession boundaries by walking play-by-play events in
order and tracking whose possession it is, rather than estimating a count
from box-score totals."""

import re

from validate_clock_alignment import pbp_elapsed_tenths


def add_elapsed_column(pbp_df):
    pbp_df = pbp_df.copy()
    pbp_df["elapsed"] = pbp_df.apply(
        lambda r: pbp_elapsed_tenths(r["period"], r["clock"]), axis=1
    )
    return pbp_df.sort_values(["elapsed", "actionNumber"]).reset_index(drop=True)


def _parse_ft_subtype(subtype):
    """Technical: play resumes with whoever had the ball before, regardless
    of which team the tech was on - no possession effect either way.
    Flagrant: the fouled team is awarded the ball afterward, period,
    whether the free throw is made or missed. The shooter is always the
    fouled player (you can't shoot FTs for your own foul), so the
    shooter's team on the row IS the team that gets the ball - this
    correctly handles the offense committing a flagrant too, not just
    the common defense-fouls-offense case.
    Normal shooting foul: same as a made/missed shot - made switches
    possession, missed goes to a rebound."""
    if "Technical" in subtype:
        return "technical", None, None
    if "Flagrant" in subtype:
        m = re.match(r"Free Throw Flagrant (\d+) of (\d+)", subtype)
        return "flagrant", int(m.group(1)), int(m.group(2))
    m = re.match(r"Free Throw (\d+) of (\d+)", subtype)
    return "normal", int(m.group(1)), int(m.group(2))


def detect_possessions(pbp_df):
    """Returns a list of dicts: start (elapsed tenths), end, offense_team_id."""
    pbp_df = add_elapsed_column(pbp_df)
    team_ids = [t for t in pbp_df["teamId"].unique() if t]
    assert len(team_ids) == 2, f"expected 2 teams, got {team_ids}"

    def opponent(team_id):
        return team_ids[0] if team_id == team_ids[1] else team_ids[1]

    possessions = []
    offense_team = None
    possession_start = 0
    last_shot_team = None

    def close(end_elapsed):
        nonlocal possession_start
        if offense_team is not None and end_elapsed > possession_start:
            possessions.append(
                {"start": possession_start, "end": end_elapsed, "offense_team_id": offense_team}
            )
        possession_start = end_elapsed

    for _, row in pbp_df.iterrows():
        at = row["actionType"]
        tt = row["teamId"]
        elapsed = row["elapsed"]

        if offense_team is None and tt and at in ("Made Shot", "Missed Shot", "Turnover", "Free Throw"):
            offense_team = tt
            possession_start = elapsed

        if at == "Made Shot":
            close(elapsed)
            offense_team = opponent(tt)
            last_shot_team = tt

        elif at == "Missed Shot":
            last_shot_team = tt

        elif at == "Turnover":
            close(elapsed)
            offense_team = opponent(tt)

        elif at == "Violation":
            close(elapsed)
            offense_team = opponent(tt)

        elif at == "Free Throw":
            kind, x, y = _parse_ft_subtype(row["subType"])
            if kind == "technical":
                continue  # no possession effect at all
            last_shot_team = tt
            is_last = x == y
            if not is_last:
                continue
            if kind == "flagrant":
                close(elapsed)
                offense_team = tt  # fouled team is awarded the ball, make or miss
            else:
                is_miss = row["description"].strip().upper().startswith("MISS")
                if not is_miss:
                    close(elapsed)
                    offense_team = opponent(tt)
                # if missed, wait for the rebound event as usual

        elif at == "Rebound":
            if tt == last_shot_team:
                pass  # offensive rebound, possession continues
            else:
                close(elapsed)
                offense_team = tt

        elif at == "period" and row["subType"] == "end":
            close(elapsed)
            offense_team = None

    return possessions, team_ids
