import pandas as pd
import pytest

from src.roster_fit import (
    assign_fit_grades,
    build_fit_table,
    overlap_score,
    player_playtype_diet,
    player_shot_zone_diet,
    player_total_possessions,
    raw_fit_pct,
    team_diet,
)


# --- player_shot_zone_diet ------------------------------------------------

def test_player_shot_zone_diet_computes_pct_of_attempts_per_zone():
    shots = pd.DataFrame({
        "SHOT_ZONE_BASIC": ["Restricted Area"] * 6 + ["Mid-Range"] * 2 + ["Above the Break 3"] * 2,
    })
    diet = player_shot_zone_diet(shots)
    assert diet["Restricted Area"] == pytest.approx(0.6)
    assert diet["Mid-Range"] == pytest.approx(0.2)
    assert diet["Above the Break 3"] == pytest.approx(0.2)


def test_player_shot_zone_diet_zones_never_taken_fill_in_as_zero():
    shots = pd.DataFrame({"SHOT_ZONE_BASIC": ["Restricted Area"] * 4})
    diet = player_shot_zone_diet(shots)
    assert diet["Left Corner 3"] == 0.0
    assert diet["Mid-Range"] == 0.0


def test_player_shot_zone_diet_excludes_backcourt_heaves():
    shots = pd.DataFrame({
        "SHOT_ZONE_BASIC": ["Restricted Area"] * 9 + ["Backcourt"],
    })
    diet = player_shot_zone_diet(shots)
    # the backcourt heave shouldn't dilute the real zones or appear in the output
    assert diet["Restricted Area"] == pytest.approx(1.0)
    assert "Backcourt" not in diet.index


# --- player_playtype_diet --------------------------------------------------

def test_player_playtype_diet_reindexes_missing_categories_to_zero():
    """A player absent from a play type's table (rather than present with
    POSS_PCT=0) must still show up as 0 in his diet, not get dropped."""
    rows = pd.DataFrame({
        "PLAY_TYPE": ["Isolation"],
        "POSS_PCT": [0.4],
        "GP": [50],
    })
    diet = player_playtype_diet(rows, all_play_types=["Isolation", "Spotup", "Cut"])
    assert diet["Isolation"] == pytest.approx(0.4)
    assert diet["Spotup"] == 0.0
    assert diet["Cut"] == 0.0


def test_player_playtype_diet_handles_a_player_with_zero_rows():
    """A player with no synergy rows at all (ran no recorded play type
    this season) should get an all-zero diet, not crash - groupby().apply()
    on a totally empty frame returns a DataFrame instead of a Series, which
    used to break the reindex below it."""
    rows = pd.DataFrame({"PLAY_TYPE": pd.array([], dtype=object), "POSS_PCT": pd.array([], dtype="float64"), "GP": pd.array([], dtype="int64")})
    diet = player_playtype_diet(rows, all_play_types=["Isolation", "Spotup"])
    assert diet["Isolation"] == 0.0
    assert diet["Spotup"] == 0.0


def test_player_playtype_diet_combines_traded_players_two_stints_by_games_played():
    """A player traded mid-season gets one row per team stint with no
    combined row in this data (unlike BR's contracts table) - the two
    stints must be blended via a GP-weighted average, not summed (which
    would double-count) or left as duplicate rows."""
    rows = pd.DataFrame({
        "PLAY_TYPE": ["Isolation", "Isolation"],
        "POSS_PCT": [0.421, 0.365],
        "GP": [41, 26],
    })
    diet = player_playtype_diet(rows, all_play_types=["Isolation"])
    expected = (0.421 * 41 + 0.365 * 26) / (41 + 26)
    assert diet["Isolation"] == pytest.approx(expected)


# --- player_total_possessions ----------------------------------------------

def test_player_total_possessions_sums_across_play_types():
    rows = pd.DataFrame({"PLAY_TYPE": ["Isolation", "Spotup"], "POSS": [400, 90]})
    assert player_total_possessions(rows) == 490


# --- team_diet ---------------------------------------------------------

def test_team_diet_weights_players_by_playing_time():
    diets = {
        "starter": pd.Series({"A": 1.0, "B": 0.0}),
        "bench": pd.Series({"A": 0.0, "B": 1.0}),
    }
    weights = {"starter": 900, "bench": 100}
    blended = team_diet(diets, weights)
    # the starter's all-A habit should dominate a bench player's all-B habit
    assert blended["A"] == pytest.approx(0.9)
    assert blended["B"] == pytest.approx(0.1)


# --- overlap_score -----------------------------------------------------

def test_overlap_score_identical_diets_is_one():
    diet = pd.Series({"A": 0.6, "B": 0.4})
    assert overlap_score(diet, diet) == pytest.approx(1.0)


def test_overlap_score_completely_disjoint_categories_is_zero():
    a = pd.Series({"A": 1.0})
    b = pd.Series({"B": 1.0})
    assert overlap_score(a, b) == pytest.approx(0.0)


def test_overlap_score_takes_the_smaller_share_per_category():
    a = pd.Series({"A": 0.7, "B": 0.3})
    b = pd.Series({"A": 0.2, "B": 0.8})
    assert overlap_score(a, b) == pytest.approx(0.2 + 0.3)


# --- raw_fit_pct / build_fit_table: the worked bruiser-team-vs-3&D example --

def test_raw_fit_pct_matches_the_worked_bruiser_vs_shooter_example():
    """Reproduces the exact scenario walked through with the user: a
    paint-bully, isolation-heavy team paired against a catch-and-shoot 3&D
    specialist. Both the shot-zone and play-type overlap come out to 38%,
    so the blended raw fit should land at exactly 62%."""
    team_shots = pd.Series({
        "Restricted Area": 0.55, "In The Paint (Non-RA)": 0.15, "Mid-Range": 0.12,
        "Left Corner 3": 0.0, "Right Corner 3": 0.03, "Above the Break 3": 0.15,
    })
    player_shots = pd.Series({
        "Restricted Area": 0.10, "In The Paint (Non-RA)": 0.05, "Mid-Range": 0.05,
        "Left Corner 3": 0.0, "Right Corner 3": 0.30, "Above the Break 3": 0.50,
    })
    team_playtypes = pd.Series({
        "Isolation": 0.30, "Postup": 0.20, "PRBallHandler": 0.20,
        "Spotup": 0.10, "Cut": 0.10, "Transition": 0.10,
    })
    player_playtypes = pd.Series({
        "Isolation": 0.03, "Postup": 0.0, "PRBallHandler": 0.05,
        "Spotup": 0.65, "Cut": 0.12, "Transition": 0.15,
    })

    shot_overlap = overlap_score(player_shots, team_shots)
    playtype_overlap = overlap_score(player_playtypes, team_playtypes)
    assert shot_overlap == pytest.approx(0.38)
    assert playtype_overlap == pytest.approx(0.38)
    assert raw_fit_pct(shot_overlap, playtype_overlap) == pytest.approx(62.0)


# --- assign_fit_grades --------------------------------------------------

def test_assign_fit_grades_buckets_by_percentile_not_raw_value():
    """The whole point of grading: even though every one of these raw fit
    scores is well under 100, the best-in-pool candidate must still grade
    out as an A, since raw_fit_pct is structurally compressed and 100 is
    never actually reachable. 20 values so the bottom one actually clears
    into F territory - pandas' rank(pct=True) is rank/N, never a literal
    0, so with too small a pool the worst score can't reach the bottom
    bucket at all (see the N=2 case in build_fit_table below)."""
    raw_fit = pd.Series(list(range(10, 210, 10)), name="raw_fit_pct")  # 20 values, 10..200
    grades = assign_fit_grades(raw_fit)
    # top scorer should be an A even though its raw value alone (the
    # worked example landed at 62) looks mediocre on a 0-100 scale
    assert grades.iloc[-1] == "A"
    assert grades.iloc[0] == "F"


def test_build_fit_table_ranks_the_worked_example_player_as_a_top_grade():
    team_shots = pd.Series({
        "Restricted Area": 0.55, "In The Paint (Non-RA)": 0.15, "Mid-Range": 0.12,
        "Left Corner 3": 0.0, "Right Corner 3": 0.03, "Above the Break 3": 0.15,
    })
    team_playtypes = pd.Series({
        "Isolation": 0.30, "Postup": 0.20, "PRBallHandler": 0.20,
        "Spotup": 0.10, "Cut": 0.10, "Transition": 0.10,
    })
    shooter_shots = pd.Series({
        "Restricted Area": 0.10, "In The Paint (Non-RA)": 0.05, "Mid-Range": 0.05,
        "Left Corner 3": 0.0, "Right Corner 3": 0.30, "Above the Break 3": 0.50,
    })
    shooter_playtypes = pd.Series({
        "Isolation": 0.03, "Postup": 0.0, "PRBallHandler": 0.05,
        "Spotup": 0.65, "Cut": 0.12, "Transition": 0.15,
    })
    # a second post-up bruiser nearly identical to the team - should duplicate, not fit
    clone_shots = team_shots.copy()
    clone_playtypes = team_playtypes.copy()

    table = build_fit_table(
        team_shots, team_playtypes,
        candidate_shot_diets={"shooter": shooter_shots, "clone": clone_shots},
        candidate_playtype_diets={"shooter": shooter_playtypes, "clone": clone_playtypes},
    )

    shooter_row = table[table["player_id"] == "shooter"].iloc[0]
    clone_row = table[table["player_id"] == "clone"].iloc[0]
    assert shooter_row["raw_fit_pct"] == pytest.approx(62.0)
    assert shooter_row["raw_fit_pct"] > clone_row["raw_fit_pct"]
    assert shooter_row["fit_grade"] == "A"
    # only 2 candidates means the worse one still sits at the 50th
    # percentile (rank/N with N=2), not the bottom bucket - a real
    # candidate pool of ~400+ players is what actually lets a true
    # duplicate fall all the way to a D or F
    assert clone_row["fit_grade"] == "C"
