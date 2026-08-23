import pandas as pd
import pytest

from src.decision_views import (
    _ordinal,
    player_detail,
    rank_by_current_bpm,
    rank_by_fit,
    rank_by_projected_bpm,
    rank_by_rapm,
    rank_by_surplus,
)


# --- _ordinal ----------------------------------------------------------

def test_ordinal_uses_the_right_suffix_including_the_teens_exception():
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(0) == "0th"
    # the 11th-13th teens exception - not 11st/12nd/13rd
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(100) == "100th"


def make_universe():
    return pd.DataFrame({
        "nba_player_id": [1, 2, 3],
        "player": ["Star", "Average", "Bench"],
        "team": ["OKC", "DEN", "LAL"],
        "age": [25, 28, 22],
        "rapm": [6.0, 1.0, -2.0],
        "current_bpm": [8.0, 2.0, None],
        "bpm_+1y": [8.5, 1.5, None],
        "bpm_+1y_band": ["[7, 10] (68%)", "[0, 3] (68%)", None],
        "bpm_+3y": [7.0, 0.5, None],
        "bpm_+3y_band": ["[5, 9] (68%)", "[-1, 2] (68%)", None],
        "surplus_score": [40.0, -10.0, 5.0],
        "value_percentile": [95.0, 40.0, 55.0],
        "cost_percentile": [55.0, 50.0, 50.0],
        "avg_value": [6.5, 1.0, -1.5],
        "avg_cap_pct": [0.30, 0.10, 0.01],
        "years_left": [3, 2, 1],
    })


# --- rank_by_surplus -------------------------------------------------------

def test_rank_by_surplus_sorts_descending_and_explains_why():
    result = rank_by_surplus(make_universe())
    assert list(result["player"]) == ["Star", "Bench", "Average"]
    assert "surplus score +40" in result.iloc[0]["reasoning"]
    assert "95th percentile production" in result.iloc[0]["reasoning"]


# --- rank_by_rapm -----------------------------------------------------------

def test_rank_by_rapm_sorts_descending_with_percentile_reasoning():
    result = rank_by_rapm(make_universe())
    assert list(result["player"]) == ["Star", "Average", "Bench"]
    assert "100th percentile" in result.iloc[0]["reasoning"]


# --- rank_by_current_bpm -----------------------------------------------------

def test_rank_by_current_bpm_drops_players_with_no_bpm():
    result = rank_by_current_bpm(make_universe())
    # Bench has no current_bpm (below the minutes floor upstream) - excluded, not crashed on
    assert list(result["player"]) == ["Star", "Average"]


# --- rank_by_projected_bpm ---------------------------------------------------

def test_rank_by_projected_bpm_uses_the_requested_horizon():
    result = rank_by_projected_bpm(make_universe(), years=1)
    assert list(result["player"]) == ["Star", "Average"]
    assert "projected 1y BPM +8.5" in result.iloc[0]["reasoning"]


def test_rank_by_projected_bpm_rejects_an_unsupported_horizon():
    with pytest.raises(ValueError):
        rank_by_projected_bpm(make_universe(), years=5)


# --- rank_by_fit -------------------------------------------------------------

def test_rank_by_fit_joins_surplus_context_without_changing_the_fit_sort(monkeypatch):
    fit_table = pd.DataFrame({
        "player_id": [2, 1],  # deliberately not pre-sorted by surplus
        "player_name": ["Average", "Star"],
        "raw_fit_pct": [70.0, 62.0],
        "fit_grade": ["A", "B"],
    })
    monkeypatch.setattr("src.decision_views.build_team_fit_table", lambda team, season: fit_table)

    result = rank_by_fit(make_universe(), "OKC")

    # sort order comes straight from build_team_fit_table's own ordering - not re-sorted by surplus
    assert list(result["player_name"]) == ["Average", "Star"]
    assert result.iloc[0]["surplus_score"] == pytest.approx(-10.0)
    assert "fit grade A" in result.iloc[0]["reasoning"]
    assert "surplus score -10" in result.iloc[0]["reasoning"]


def test_rank_by_fit_handles_a_candidate_with_no_surplus_data(monkeypatch):
    fit_table = pd.DataFrame({
        "player_id": [999],
        "player_name": ["Unmatched Rookie"],
        "raw_fit_pct": [55.0],
        "fit_grade": ["C"],
    })
    monkeypatch.setattr("src.decision_views.build_team_fit_table", lambda team, season: fit_table)

    result = rank_by_fit(make_universe(), "OKC")
    assert "no contract/surplus data" in result.iloc[0]["reasoning"]


# --- player_detail -----------------------------------------------------------

def test_player_detail_assembles_projection_and_diets():
    row = make_universe().set_index("player").loc["Star"]
    row["player"] = "Star"
    shots = pd.DataFrame({"SHOT_ZONE_BASIC": ["Restricted Area"] * 3 + ["Mid-Range"]})
    playtype_rows = pd.DataFrame({"PLAY_TYPE": ["Isolation"], "POSS_PCT": [0.5], "GP": [40]})

    detail = player_detail(row, shots, playtype_rows, all_play_types=["Isolation", "Spotup"])

    assert detail["player"] == "Star"
    assert detail["surplus_score"] == pytest.approx(40.0)
    assert [p["years"] for p in detail["projection"]] == [1, 3]
    assert detail["shot_diet"]["Restricted Area"] == pytest.approx(0.75)
    assert detail["playtype_diet"]["Isolation"] == pytest.approx(0.5)
    assert detail["playtype_diet"]["Spotup"] == 0.0


def test_player_detail_handles_a_player_with_no_projection_data():
    row = make_universe().set_index("player").loc["Bench"]
    row["player"] = "Bench"
    shots = pd.DataFrame({"SHOT_ZONE_BASIC": []})
    playtype_rows = pd.DataFrame({"PLAY_TYPE": [], "POSS_PCT": [], "GP": []})

    detail = player_detail(row, shots, playtype_rows, all_play_types=["Isolation"])
    assert detail["projection"] == []
    assert detail["current_bpm"] is None or pd.isna(detail["current_bpm"])
