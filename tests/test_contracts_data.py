from pathlib import Path

import pandas as pd

from src.contracts_data import parse_cap_history_html, parse_contracts_html

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_contracts_html_extracts_player_ids_and_year_columns():
    html_text = (FIXTURES / "bref_contracts_sample.html").read_text()
    df = parse_contracts_html(html_text)

    assert list(df["player_id"]) == ["curryst01", "tatumja01", "banchpa01"]
    assert list(df["player"]) == ["Stephen Curry", "Jayson Tatum", "Paolo Banchero"]
    assert "2026-27" in df.columns and "2031-32" in df.columns


def test_parse_contracts_html_converts_dollar_strings_to_numbers():
    html_text = (FIXTURES / "bref_contracts_sample.html").read_text()
    df = parse_contracts_html(html_text)

    curry = df[df["player_id"] == "curryst01"].iloc[0]
    assert curry["2026-27"] == 62_587_158
    assert curry["guaranteed"] == 62_587_158

    tatum = df[df["player_id"] == "tatumja01"].iloc[0]
    assert tatum["2027-28"] == 62_786_682
    assert tatum["2029-30"] == 71_446_914


def test_parse_contracts_html_leaves_uncontracted_years_as_nan_not_zero():
    html_text = (FIXTURES / "bref_contracts_sample.html").read_text()
    df = parse_contracts_html(html_text)

    curry = df[df["player_id"] == "curryst01"].iloc[0]
    assert pd.isna(curry["2027-28"])


def test_parse_contracts_html_survives_a_colspan_divider_row_mid_table():
    """Basketball-Reference repeats its header every ~20 rows, sometimes as
    a blank colspan divider row rather than literal 'Player' text - the
    fixture has one between Tatum and Banchero. If it isn't filtered out
    consistently on both the pandas side and the lxml player_id side, every
    row after it silently gets the wrong player_id."""
    html_text = (FIXTURES / "bref_contracts_sample.html").read_text()
    df = parse_contracts_html(html_text)

    assert len(df) == 3
    assert list(df["player_id"]) == ["curryst01", "tatumja01", "banchpa01"]
    banchero = df[df["player_id"] == "banchpa01"].iloc[0]
    assert banchero["player"] == "Paolo Banchero"
    assert banchero["2026-27"] == 12_585_900


def test_parse_cap_history_html_handles_the_html_comment_wrapper():
    html_text = (FIXTURES / "bref_cap_history_sample.html").read_text()
    df = parse_cap_history_html(html_text)

    assert list(df["season"]) == ["2024-25", "2025-26", "2026-27"]
    assert df.loc[df["season"] == "2025-26", "cap"].iloc[0] == 154_647_000
    assert df.loc[df["season"] == "2026-27", "cap"].iloc[0] == 164_961_000
