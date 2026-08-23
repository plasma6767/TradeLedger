import pandas as pd
import pytest

from src.surplus_value import (
    add_surplus_score,
    average_projected_value,
    compute_avg_cap_pct,
    compute_recent_cap_growth_rate,
    estimate_cap_for_season,
    years_left_columns,
)


# --- years_left_columns -----------------------------------------------

def test_years_left_columns_stops_at_first_missing_year():
    row = pd.Series({"2026-27": 10.0, "2027-28": 12.0, "2028-29": None, "2029-30": 14.0})
    assert years_left_columns(row, ["2026-27", "2027-28", "2028-29", "2029-30"]) == ["2026-27", "2027-28"]


def test_years_left_columns_all_years_present():
    row = pd.Series({"2026-27": 10.0, "2027-28": 12.0})
    assert years_left_columns(row, ["2026-27", "2027-28"]) == ["2026-27", "2027-28"]


def test_years_left_columns_expiring_deal_returns_one_year():
    row = pd.Series({"2026-27": 10.0, "2027-28": None})
    assert years_left_columns(row, ["2026-27", "2027-28"]) == ["2026-27"]


# --- cap growth / estimation --------------------------------------------

def test_compute_recent_cap_growth_rate_matches_known_compounding():
    # cap doubles over 5 seasons -> ~14.87% CAGR
    cap_history = pd.DataFrame({
        "season": ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
        "cap": [100_000_000, 110_000_000, 121_000_000, 133_100_000, 146_410_000, 200_000_000],
    })
    rate = compute_recent_cap_growth_rate(cap_history, lookback=5)
    assert rate == pytest.approx(0.1487, abs=0.001)


def test_estimate_cap_for_season_returns_real_value_when_published():
    cap_history = pd.DataFrame({"season": ["2025-26", "2026-27"], "cap": [154_647_000, 164_961_000]})
    assert estimate_cap_for_season(cap_history, "2026-27", growth_rate=0.05) == 164_961_000


def test_estimate_cap_for_season_extrapolates_unpublished_years():
    cap_history = pd.DataFrame({"season": ["2025-26", "2026-27"], "cap": [100_000_000, 110_000_000]})
    # 2028-29 is 2 seasons past the last published year, at a 10% growth rate
    estimated = estimate_cap_for_season(cap_history, "2028-29", growth_rate=0.10)
    assert estimated == pytest.approx(110_000_000 * 1.10 ** 2)


def test_compute_avg_cap_pct_averages_each_years_share_of_that_years_cap():
    cap_history = pd.DataFrame({"season": ["2026-27", "2027-28"], "cap": [160_000_000, 176_000_000]})
    # $20M against a $160M cap (12.5%), then $22M against a $176M cap (12.5%) - flat cap share
    avg = compute_avg_cap_pct([20_000_000, 22_000_000], ["2026-27", "2027-28"], cap_history, growth_rate=0.1)
    assert avg == pytest.approx(0.125)


# --- average_projected_value ---------------------------------------------

def test_average_projected_value_uses_only_the_plain_average_delta():
    # age 25 -> +1.0, age 26 -> +0.5, starting from a RAPM of 2.0
    curve = pd.DataFrame({
        "age_from": [25, 26, 27],
        "avg_delta": [1.0, 0.5, -0.5],
        "reliable": [True, True, True],
    })
    # year 1 (age 26): 2.0 + 1.0 = 3.0; year 2 (age 27): 3.0 + 0.5 = 3.5
    avg_value = average_projected_value(curve, current_age=25, current_rapm=2.0, years=2)
    assert avg_value == pytest.approx((3.0 + 3.5) / 2)


def test_average_projected_value_one_year_equals_base_plus_first_delta():
    curve = pd.DataFrame({"age_from": [30], "avg_delta": [-1.2], "reliable": [True]})
    avg_value = average_projected_value(curve, current_age=30, current_rapm=4.0, years=1)
    assert avg_value == pytest.approx(4.0 - 1.2)


# --- add_surplus_score: the worked example from the design discussion ----

def test_add_surplus_score_matches_the_worked_five_player_example():
    """Reproduces the exact scenario used to validate this formula: a
    below-average-but-cheap player must NOT come out looking like a bargain
    (the bug in the earlier division-based version), and a star and an
    average player who are each paid exactly market rate for their
    production should both land at 0, regardless of how much they're
    actually paid in absolute terms."""
    df = pd.DataFrame({
        "player": ["Star", "Bargain", "Average", "BadExpensive", "BadCheap"],
        "avg_value": [6.0, 3.0, 0.0, -1.5, -2.0],
        "avg_cap_pct": [0.25, 0.05, 0.10, 0.15, 0.01],
    })
    result = add_surplus_score(df).set_index("player")

    # pandas' rank(pct=True) uses rank/N (5 players land at 20/40/60/80/100,
    # never a literal 0), unlike the evenly-spaced 0/25/50/75/100 scale used
    # when hand-walking this example - same shape, different convention
    assert result.loc["Star", "surplus_score"] == pytest.approx(0)
    assert result.loc["Average", "surplus_score"] == pytest.approx(0)
    assert result.loc["BadCheap", "surplus_score"] == pytest.approx(0)
    assert result.loc["Bargain", "surplus_score"] == pytest.approx(40)
    assert result.loc["BadExpensive", "surplus_score"] == pytest.approx(-40)

    # the whole point: the worst-value contract is the expensive/bad one,
    # not the merely bad-but-minimum-salary one
    assert result.loc["BadExpensive", "surplus_score"] < result.loc["BadCheap", "surplus_score"]


def test_add_surplus_score_is_bounded_even_with_a_near_zero_value_player():
    """The scenario that broke the earlier ratio-based formula: a player
    whose value is a tiny positive number right near the replacement
    floor. Percentile subtraction must stay finite and bounded, unlike
    cap_pct / value which blows up as value -> 0."""
    df = pd.DataFrame({
        "player": ["NearZero", "Normal"],
        "avg_value": [0.01, 3.0],
        "avg_cap_pct": [0.01, 0.10],
    })
    result = add_surplus_score(df)
    assert result["surplus_score"].between(-100, 100).all()
