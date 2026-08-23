"""Roster fit: does a candidate player's shot selection and play-type usage
complement or duplicate what a specific team's current roster already
generates a lot of.

Each player is reduced to two "diets" - % of shots by court zone, % of
offensive possessions by play type. A team's diet is its current roster's
individual diets blended together, weighted by playing time. A candidate's
fit against that team is 1 minus how much his diets overlap with the
team's (overlap = the shared ground between the two distributions, summed
zone/play-type by zone/play-type - the same idea as shading in wherever two
bar charts drawn on top of each other cover the same space).

That raw number is structurally compressed - nobody's diet is 100% unlike
every team's, since every player takes some shots at the rim and every
team gets some transition points - so a genuinely great complement still
only scores ~60-70%, not 90-100%. Reading it as a face-value percentage
undersells it. The fix: grade it against the real spread of fit scores
across the actual candidate pool being compared, not fixed thresholds."""

import pandas as pd
from nba_api.stats.static import teams as nba_teams

from src.roster_data import fetch_all_rosters
from src.shot_profile_data import fetch_all_playtypes, fetch_players_shots

SHOT_ZONES = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3",
]  # Backcourt heaves excluded - desperation buzzer shots, not a real
# shot-selection signal, and including them would just dilute every
# player's real zones by ~0-1%

# top 10% -> A, next 20% -> B, middle 40% -> C, next 20% -> D, bottom 10% -> F,
# calibrated against the real distribution of raw_fit_pct within a candidate
# pool (assign_fit_grades), not against raw_fit_pct's own 0-100 scale
GRADE_BANDS = [(0, "F"), (10, "D"), (30, "C"), (70, "B"), (90, "A")]


def player_shot_zone_diet(shots: pd.DataFrame) -> pd.Series:
    """% of a player's field goal attempts from each real shot zone."""
    zones = shots[shots["SHOT_ZONE_BASIC"] != "Backcourt"]
    counts = zones["SHOT_ZONE_BASIC"].value_counts()
    diet = counts / counts.sum()
    return diet.reindex(SHOT_ZONES, fill_value=0.0)


def player_playtype_diet(playtype_rows: pd.DataFrame, all_play_types: list[str]) -> pd.Series:
    """% of a player's offensive possessions in each play type. Players
    traded mid-season have one row per team stint in this data (no
    combined row exists here the way BR's contracts table has one) -
    stints are combined via a games-played-weighted average per play type
    rather than picking one stint or summing raw percentages, which would
    double-count. Play types the player never ran (he's simply absent from
    that play type's table, not present with 0) fill in as 0 against the
    full league-wide category list, not just whatever categories this one
    player happens to have rows for."""
    weighted = playtype_rows.groupby("PLAY_TYPE").apply(
        lambda g: (g["POSS_PCT"] * g["GP"]).sum() / g["GP"].sum(), include_groups=False
    )
    return weighted.reindex(all_play_types, fill_value=0.0)


def player_total_possessions(playtype_rows: pd.DataFrame) -> float:
    """Total offensive possessions used across all play types this season -
    the playing-time weight used in team_diet, so a bench player's habits
    don't count as much toward the team's diet as a starter's."""
    return playtype_rows["POSS"].sum()


def team_diet(player_diets: dict, weights: dict) -> pd.Series:
    """Blend rostered players' individual diets into one team-level diet,
    weighted by each player's share of playing time."""
    total_weight = sum(weights[pid] for pid in player_diets)
    blended = sum(player_diets[pid] * weights[pid] for pid in player_diets) / total_weight
    return blended


def overlap_score(diet_a: pd.Series, diet_b: pd.Series) -> float:
    """Distributional overlap between two diets: for each shared category,
    take the smaller share and add them up. Bounded 0 (no shared ground)
    to 1 (identical diets)."""
    categories = diet_a.index.union(diet_b.index)
    a = diet_a.reindex(categories, fill_value=0.0)
    b = diet_b.reindex(categories, fill_value=0.0)
    return pd.concat([a, b], axis=1).min(axis=1).sum()


def raw_fit_pct(shot_overlap: float, playtype_overlap: float, shot_weight: float = 0.5) -> float:
    """1 - the weighted-average overlap, as a percent. 100 = the
    candidate's habits share no ground with what the team already
    generates (a pure gap-filler); 0 = a total duplicate."""
    overlap = shot_weight * shot_overlap + (1 - shot_weight) * playtype_overlap
    return (1 - overlap) * 100


def assign_fit_grades(raw_fit: pd.Series) -> pd.Series:
    """Letter grade calibrated against the real spread of raw_fit_pct
    within this candidate pool, not fixed thresholds - since raw_fit_pct
    is structurally compressed well under 100 even for a great fit, fixed
    thresholds would grade almost every real player a D or F regardless of
    how good the fit actually is."""
    percentile = raw_fit.rank(pct=True) * 100
    grade = pd.Series(index=raw_fit.index, dtype=object)
    for threshold, letter in GRADE_BANDS:
        grade[percentile >= threshold] = letter
    return grade


def build_fit_table(
    team_shot_diet: pd.Series,
    team_playtype_diet: pd.Series,
    candidate_shot_diets: dict,
    candidate_playtype_diets: dict,
    shot_weight: float = 0.5,
) -> pd.DataFrame:
    """One row per candidate: shot overlap, play-type overlap, blended raw
    fit %, and a letter grade calibrated against this same candidate
    pool's own spread of fit scores."""
    rows = []
    for player_id in candidate_shot_diets:
        shot_overlap = overlap_score(candidate_shot_diets[player_id], team_shot_diet)
        playtype_overlap = overlap_score(candidate_playtype_diets[player_id], team_playtype_diet)
        rows.append({
            "player_id": player_id,
            "shot_overlap": shot_overlap,
            "playtype_overlap": playtype_overlap,
            "raw_fit_pct": raw_fit_pct(shot_overlap, playtype_overlap, shot_weight),
        })
    table = pd.DataFrame(rows)
    table["fit_grade"] = assign_fit_grades(table["raw_fit_pct"])
    return table.sort_values("raw_fit_pct", ascending=False).reset_index(drop=True)


def team_id_for_abbreviation(abbreviation: str) -> int:
    for team in nba_teams.get_teams():
        if team["abbreviation"] == abbreviation:
            return team["id"]
    raise ValueError(f"no team found for abbreviation {abbreviation!r}")


def build_team_fit_table(team_abbreviation: str, season: str = "2025-26") -> pd.DataFrame:
    """Fit table for every other currently-rostered player in the league
    against one team's current roster. Ties together roster_data (who's on
    what team right now) and shot_profile_data (shot/play-type raw pulls)
    with this module's pure diet/overlap math."""
    team_id = team_id_for_abbreviation(team_abbreviation)
    rosters = fetch_all_rosters(season)
    playtypes = fetch_all_playtypes(season)
    all_play_types = sorted(playtypes["PLAY_TYPE"].unique())

    player_ids = rosters["PLAYER_ID"].tolist()
    shots_by_player = fetch_players_shots(player_ids, season)

    shot_diets, playtype_diets, weights = {}, {}, {}
    skipped_no_sample = []
    for player_id in player_ids:
        shots = shots_by_player[player_id]
        pt_rows = playtypes[playtypes["PLAYER_ID"] == player_id]
        # a two-way/deep-bench player who never registered a real FGA or a
        # synergy possession this season has nothing to build a diet from -
        # including him would divide by zero and poison the team blend
        # with NaN, so he's dropped rather than counted as a blank profile
        if len(shots) == 0 or pt_rows["POSS"].sum() == 0:
            skipped_no_sample.append(player_id)
            continue
        shot_diets[player_id] = player_shot_zone_diet(shots)
        playtype_diets[player_id] = player_playtype_diet(pt_rows, all_play_types)
        weights[player_id] = player_total_possessions(pt_rows)

    if skipped_no_sample:
        print(f"WARNING: {len(skipped_no_sample)} rostered players had no usable shot/play-type sample: {skipped_no_sample}")

    team_player_ids = [pid for pid in rosters.loc[rosters["TeamID"] == team_id, "PLAYER_ID"] if pid in shot_diets]
    team_shot_diet = team_diet(
        {pid: shot_diets[pid] for pid in team_player_ids}, {pid: weights[pid] for pid in team_player_ids}
    )
    team_playtype_diet = team_diet(
        {pid: playtype_diets[pid] for pid in team_player_ids}, {pid: weights[pid] for pid in team_player_ids}
    )

    candidate_ids = [pid for pid in shot_diets if pid not in team_player_ids]
    table = build_fit_table(
        team_shot_diet, team_playtype_diet,
        {pid: shot_diets[pid] for pid in candidate_ids},
        {pid: playtype_diets[pid] for pid in candidate_ids},
    )
    names = rosters.drop_duplicates(subset=["PLAYER_ID"]).set_index("PLAYER_ID")["PLAYER"]
    table["player_name"] = table["player_id"].map(names)
    return table


if __name__ == "__main__":
    import sys

    team_abbr = sys.argv[1] if len(sys.argv) > 1 else "OKC"
    table = build_team_fit_table(team_abbr)
    pd.set_option("display.width", 200)

    print(f"\n{len(table)} candidates ranked for {team_abbr}\n")
    display_cols = ["player_name", "shot_overlap", "playtype_overlap", "raw_fit_pct", "fit_grade"]

    print("Best fits (top 15):")
    print(table[display_cols].head(15).to_string(index=False))

    print("\nWorst fits / most redundant (bottom 15):")
    print(table[display_cols].tail(15).to_string(index=False))

    from pathlib import Path
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"fit_{team_abbr}.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")
