"""Detect real possession boundaries by walking play-by-play events in
order, tracking whose possession it is and tallying points live as we go
(never via a timestamp lookup, which can't disambiguate events that share
a clock reading - e.g. a turnover immediately followed by a free throw,
since the game clock doesn't move during dead-ball sequences).

Also splits a possession at any lineup-substitution boundary that falls
inside it, using the same live event walk, so a sub mid-possession (e.g.
between free throws) attributes points to the correct lineup segment."""

import re

from nba_api.stats.static import teams as static_teams

from validate_clock_alignment import pbp_elapsed_tenths


def _build_nickname_map(team_ids):
    """Team-level bookkeeping rows (team rebounds, shot-clock turnovers)
    have teamId=0 - the team is only identifiable from description text
    like 'CAVALIERS Rebound' or 'Wizards Turnover'."""
    all_teams = {t["id"]: t["nickname"].upper() for t in static_teams.get_teams()}
    return {tid: all_teams[tid] for tid in team_ids}


def _resolve_team(tt, description, nickname_map):
    if tt:
        return tt
    desc_upper = description.upper()
    matches = [tid for tid, nick in nickname_map.items() if nick in desc_upper]
    return matches[0] if len(matches) == 1 else None


def add_elapsed_column(pbp_df):
    pbp_df = pbp_df.copy()
    pbp_df["elapsed"] = pbp_df.apply(
        lambda r: pbp_elapsed_tenths(r["period"], r["clock"]), axis=1
    )
    return pbp_df.sort_values(["elapsed", "actionNumber"]).reset_index(drop=True)


def _parse_ft_subtype(subtype):
    """Technical: play resumes with whoever had the ball before, regardless
    of which team the tech was on - no possession effect either way.
    Flagrant/Clear Path: the fouled team is awarded the ball afterward,
    period, whether the free throw is made or missed. The shooter is
    always the fouled player (you can't shoot FTs for your own foul), so
    the shooter's team on the row IS the team that gets the ball.
    Normal (including any foul subtype we haven't specifically named,
    like Away From Play or Transition Take): usually the ball just
    switches to the other team on a make, except some foul subtypes
    award continued possession instead - so we don't hard-code "opponent
    gets it" for the plain case; the next real event reveals who actually
    has the ball, same as at period starts. This means an unrecognized
    future foul subtype degrades safely to "wait and see" rather than a
    wrong hard-coded assumption.

    The "X of Y" numbers are always the last two numbers in the string
    regardless of which foul-type words appear before them, so we extract
    them positionally instead of hard-coding the exact text for every
    subtype we've seen - keeps a totally new wording from crashing this.
    Technical FTs don't need x/y at all (the caller ignores them), and
    some games format "Free Throw Technical" with no trailing number, so
    check for that case before attempting to parse one."""
    if "Technical" in subtype:
        return "technical", None, None
    m = re.search(r"(\d+) of (\d+)$", subtype)
    x, y = int(m.group(1)), int(m.group(2))
    if "Flagrant" in subtype or "Clear Path" in subtype:
        return "flagrant", x, y
    return "normal", x, y


def detect_possessions(pbp_df, sub_boundaries=None):
    """Returns a list of dicts: start, end (elapsed tenths), offense_team_id,
    points (net differential from the offense's perspective, so a
    technical scored by the defense during the offense's possession is
    correctly reflected)."""
    pbp_df = add_elapsed_column(pbp_df)
    team_ids = [t for t in pbp_df["teamId"].unique() if t]
    assert len(team_ids) == 2, f"expected 2 teams, got {team_ids}"
    nickname_map = _build_nickname_map(team_ids)
    sub_boundaries = sorted(sub_boundaries or [])

    def opponent(team_id):
        return team_ids[0] if team_id == team_ids[1] else team_ids[1]

    possessions = []
    offense_team = None
    possession_start = 0
    last_shot_team = None
    points_by_team = {tid: 0 for tid in team_ids}
    boundary_idx = 0
    pending_made = None  # (team, elapsed) of a made shot not yet confirmed complete - an and-1's bonus FT can still follow at the same elapsed

    def flush(end_elapsed, next_offense):
        """Record the current window if it's non-empty, then reset the
        point tally. next_offense=None means 'uncertain - let the next
        real event reveal who has the ball' (same as a period start);
        pass the current offense_team to split without changing it."""
        nonlocal possession_start, offense_team, points_by_team
        if offense_team is not None:
            # No minimum-duration check: a zero-width window can only ever
            # hold points from real events that happened at that exact
            # instant (e.g. a turnover immediately followed by a same-team
            # free throw at the same clock reading) - dropping it would
            # silently lose real points, and there's no way a zero-width
            # window can contain fabricated data.
            off_pts = points_by_team[offense_team]
            def_pts = points_by_team[opponent(offense_team)]
            possessions.append(
                {
                    "start": possession_start,
                    "end": end_elapsed,
                    "offense_team_id": offense_team,
                    "offense_points": off_pts,
                    "defense_points": def_pts,
                    "points": off_pts - def_pts,
                }
            )
        possession_start = end_elapsed
        offense_team = next_offense
        points_by_team = {tid: 0 for tid in team_ids}

    for _, row in pbp_df.iterrows():
        at = row["actionType"]
        elapsed = row["elapsed"]

        if pending_made is not None:
            p_team, p_elapsed = pending_made
            if elapsed == p_elapsed:
                # still the same tick - a shooting foul + substitutions can
                # sit between the made shot and its and-1 free throw, all at
                # this same elapsed. Only resolve on the free throw itself;
                # anything else at this tick just passes through untouched.
                if at == "Free Throw":
                    peek_tt = _resolve_team(row["teamId"], row["description"], nickname_map)
                    if peek_tt == p_team:
                        pending_made = None  # hand off to the FT's own handling below, no flush
                    else:
                        flush(p_elapsed, next_offense=None)
                        pending_made = None
            else:
                # moved to a later tick with no and-1 continuation - safe to close
                flush(p_elapsed, next_offense=None)
                pending_made = None

        # split at any lineup-substitution boundary crossed since the last event -
        # dead-ball only, so no points can have occurred exactly there
        while boundary_idx < len(sub_boundaries) and sub_boundaries[boundary_idx] < elapsed:
            flush(sub_boundaries[boundary_idx], offense_team)  # same team, just split
            boundary_idx += 1

        if at == "period":
            if row["subType"] == "end":
                flush(elapsed, next_offense=None)
            continue

        tt = _resolve_team(row["teamId"], row["description"], nickname_map)
        if tt is None:
            continue  # couldn't identify the team for a team-attributed event - skip rather than guess

        if offense_team is None and at in ("Made Shot", "Missed Shot", "Turnover", "Free Throw", "Violation"):
            offense_team = tt

        if at == "Made Shot":
            points_by_team[tt] += row["shotValue"]
            last_shot_team = tt
            pending_made = (tt, elapsed)  # defer closing - an and-1 free throw may follow at this same elapsed

        elif at == "Missed Shot":
            last_shot_team = tt

        elif at == "Turnover":
            flush(elapsed, next_offense=opponent(tt))

        elif at == "Violation":
            flush(elapsed, next_offense=opponent(tt))

        elif at == "Free Throw":
            kind, x, y = _parse_ft_subtype(row["subType"])
            is_miss = row["description"].strip().upper().startswith("MISS")
            if not is_miss:
                points_by_team[tt] += 1
            if kind == "technical":
                continue  # no possession effect at all
            last_shot_team = tt
            is_last = x == y
            if not is_last:
                continue
            if kind == "flagrant":
                flush(elapsed, next_offense=tt)  # fouled team is awarded the ball, make or miss - a known fact
            elif not is_miss:
                flush(elapsed, next_offense=None)
            # if missed (normal), wait for the rebound event as usual

        elif at == "Rebound":
            if tt == last_shot_team:
                offense_team = tt  # offensive rebound, possession continues
            else:
                flush(elapsed, next_offense=tt)  # defensive rebound - a known fact, not an assumption

    if pending_made is not None:
        flush(pending_made[1], next_offense=None)

    return possessions, team_ids
