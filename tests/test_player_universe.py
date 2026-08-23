import pandas as pd

from src.player_universe import add_current_team, add_nba_player_ids


# --- add_nba_player_ids --------------------------------------------------

def test_add_nba_player_ids_matches_on_name_and_attaches_id():
    table = pd.DataFrame({"player": ["Shai Gilgeous-Alexander", "Nikola Jokic"], "surplus_score": [40.0, 30.0]})
    rapm = pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Shai Gilgeous-Alexander", "Nikola Jokic", "Someone Else"],
        "rapm": [8.0, 7.0, 1.0],
    })
    result = add_nba_player_ids(table, rapm)
    assert result.set_index("player")["nba_player_id"].to_dict() == {
        "Shai Gilgeous-Alexander": 1,
        "Nikola Jokic": 2,
    }


def test_add_nba_player_ids_drops_and_warns_on_unmatched(capsys):
    table = pd.DataFrame({"player": ["Totally Unknown Player"], "surplus_score": [10.0]})
    rapm = pd.DataFrame({"player_id": [1], "name": ["Someone Else"], "rapm": [1.0]})
    result = add_nba_player_ids(table, rapm)
    assert len(result) == 0
    assert "Totally Unknown Player" in capsys.readouterr().out


def test_add_nba_player_ids_normalizes_suffixes_and_accents():
    table = pd.DataFrame({"player": ["Luka Doncic"], "surplus_score": [50.0]})
    rapm = pd.DataFrame({"player_id": [7], "name": ["Luka Dončić"], "rapm": [9.0]})
    result = add_nba_player_ids(table, rapm)
    assert result["nba_player_id"].iloc[0] == 7


# --- add_current_team -----------------------------------------------------

def test_add_current_team_joins_on_nba_player_id():
    table = pd.DataFrame({"player": ["A", "B"], "nba_player_id": [1, 2]})
    rosters = pd.DataFrame({"PLAYER_ID": [1, 2], "team": ["OKC", "DEN"]})
    result = add_current_team(table, rosters)
    assert result.set_index("player")["team"].to_dict() == {"A": "OKC", "B": "DEN"}


def test_add_current_team_keeps_players_off_any_current_roster():
    """A player who matched to RAPM but isn't on any team's current roster
    (e.g. waived since the roster snapshot) should keep his row with a
    missing team, not get silently dropped - his surplus/RAPM numbers are
    still real."""
    table = pd.DataFrame({"player": ["A"], "nba_player_id": [1]})
    rosters = pd.DataFrame({"PLAYER_ID": [999], "team": ["OKC"]})
    result = add_current_team(table, rosters)
    assert len(result) == 1
    assert pd.isna(result["team"].iloc[0])


def test_add_current_team_dedupes_a_player_appearing_on_two_roster_rows():
    """fetch_all_rosters concatenates one call per team; drop_duplicates on
    PLAYER_ID guards against a player somehow showing up twice rather than
    fanning the join out into duplicate rows."""
    table = pd.DataFrame({"player": ["A"], "nba_player_id": [1]})
    rosters = pd.DataFrame({"PLAYER_ID": [1, 1], "team": ["OKC", "OKC"]})
    result = add_current_team(table, rosters)
    assert len(result) == 1
