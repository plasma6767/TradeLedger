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
    # Technical and Flagrant free throws don't end a possession - the
    # fouled team retains the ball regardless of makes/misses.
    if "Technical" in subtype or "Flagrant" in subtype:
        return None, None
    m = re.match(r"Free Throw (\d+) of (\d+)", subtype)
    return int(m.group(1)), int(m.group(2))


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
            x, y = _parse_ft_subtype(row["subType"])
            if x is None:
                continue  # technical FT: no possession effect
            last_shot_team = tt
            is_last = x == y
            is_miss = row["description"].strip().upper().startswith("MISS")
            if is_last and not is_miss:
                close(elapsed)
                offense_team = opponent(tt)

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
