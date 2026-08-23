"""Rank every player's contract from best to worst value, relative to the
rest of the league.

Value number: this season's real RAPM (Phase 1), walked forward using only
the *average* year-over-year delta the aging curve (Phase 2) measured for
that age - never the simulated band, that's reserved for a future
player-detail view. The curve only ever contributes a rate of change; the
starting point is always real RAPM, never blended with BPM itself.

Cost number: each contracted year's salary as a percentage of that
season's real salary cap (estimated via recent real cap growth for years
the league hasn't published a cap for yet), averaged across however many
years are left on the deal - so a 1-year deal and a 5-year deal are judged
on the same footing instead of the longer one being penalized just for
reaching into years where the whole league naturally pays more.

Combine: convert both numbers to a percentile rank across the league and
subtract (value_percentile - cost_percentile), rather than dividing one by
the other - division blows up/becomes unstable for any player whose value
sits near zero, percentile subtraction stays bounded and stable for
everyone, so nobody has to be excluded from the ranking."""

import re
from pathlib import Path

import pandas as pd

from src.aging_curve import (
    apply_reliability_cutoff,
    build_curve,
    build_transitions,
    load_all_seasons,
    project_bpm,
)
from src.contracts_data import fetch_cap_history, fetch_contracts
from src.name_matching import match_names

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bref_advanced"

CURRENT_SEASON = "2025-26"
CAP_GROWTH_LOOKBACK_YEARS = 5


def load_current_season_ages(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Every player with a real age this season, regardless of minutes -
    unlike aging_curve.load_all_seasons() (which floors on minutes because
    it's fitting the curve itself off a reliable sample), we need age for
    anyone who has a contract, however little they played."""
    df = pd.read_json(RAW_DIR / f"{season}.json")
    return df.dropna(subset=["Age", "player_id"]).copy()


def years_left_columns(contract_row: pd.Series, year_cols: list[str]) -> list[str]:
    """Contract years starting from the soonest column, stopping at the
    first missing one. A real contract's guaranteed years are contiguous
    starting now, so a NaN here means 'the deal doesn't reach that far',
    not a gap to skip over and keep looking past."""
    active = []
    for col in year_cols:
        if pd.isna(contract_row[col]):
            break
        active.append(col)
    return active


def compute_recent_cap_growth_rate(cap_history: pd.DataFrame, lookback: int = CAP_GROWTH_LOOKBACK_YEARS) -> float:
    """Real compound annual growth rate of the cap over the last `lookback`
    published seasons - used only to estimate cap figures for future years
    the league hasn't published yet, not as a stand-in for real data where
    real data exists."""
    ordered = cap_history.sort_values("season")
    recent = ordered.tail(lookback + 1)
    start_cap, end_cap = recent["cap"].iloc[0], recent["cap"].iloc[-1]
    years = len(recent) - 1
    return (end_cap / start_cap) ** (1 / years) - 1


def estimate_cap_for_season(cap_history: pd.DataFrame, season: str, growth_rate: float) -> float:
    """The real published cap for `season` if we have it, otherwise the
    last known real cap compounded forward at the recent real growth
    rate."""
    row = cap_history[cap_history["season"] == season]
    if not row.empty:
        return row["cap"].iloc[0]

    ordered = cap_history.sort_values("season")
    last_season, last_cap = ordered["season"].iloc[-1], ordered["cap"].iloc[-1]
    last_start_year = int(last_season.split("-")[0])
    target_start_year = int(season.split("-")[0])
    years_out = target_start_year - last_start_year
    return last_cap * (1 + growth_rate) ** years_out


def compute_avg_cap_pct(
    salaries: list[float], seasons: list[str], cap_history: pd.DataFrame, growth_rate: float
) -> float:
    pct_by_year = [
        salary / estimate_cap_for_season(cap_history, season, growth_rate)
        for salary, season in zip(salaries, seasons)
    ]
    return sum(pct_by_year) / len(pct_by_year)


def average_projected_value(curve: pd.DataFrame, current_age: float, current_rapm: float, years: int) -> float:
    """Walk `current_rapm` forward `years` steps using the aging curve's
    plain average delta per age (project_bpm, not the simulated-band
    version) and average the resulting trajectory. The curve is fed RAPM
    here instead of BPM on purpose - RAPM and BPM already share units
    (points of impact per 100 possessions), so the curve's measured rate of
    change is a valid thing to apply to either one, even though the two
    metrics are never blended into a single value."""
    projections = project_bpm(curve, current_age, current_rapm, years)
    return sum(p["bpm"] for p in projections) / len(projections)


def add_surplus_score(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank each of avg_value and avg_cap_pct across the given
    players and subtract, rather than dividing avg_cap_pct by avg_value -
    a ratio is unstable (and undefined at exactly zero) for anyone whose
    value sits near the replacement-level floor; percentile subtraction
    stays bounded between -100 and +100 for every player, so nobody needs
    to be dropped from the ranking to keep the math sane."""
    df = df.copy()
    df["value_percentile"] = df["avg_value"].rank(pct=True) * 100
    df["cost_percentile"] = df["avg_cap_pct"].rank(pct=True) * 100
    df["surplus_score"] = df["value_percentile"] - df["cost_percentile"]
    return df


def build_surplus_table() -> pd.DataFrame:
    rapm = pd.read_csv(PROCESSED_DIR / "rapm_2025-26.csv")
    ages = load_current_season_ages()

    matched, unmatched = match_names(list(ages["Player"]), list(rapm["name"]))
    if unmatched:
        print(f"WARNING: {len(unmatched)} players couldn't be matched to RAPM: {unmatched}")
    ages["rapm_name"] = ages["Player"].map(matched)
    ages = ages.dropna(subset=["rapm_name"]).merge(
        rapm[["name", "rapm"]], left_on="rapm_name", right_on="name", how="left"
    )

    contracts = fetch_contracts()
    year_cols = sorted(
        (c for c in contracts.columns if re.match(r"^\d{4}-\d{2}$", c)),
        key=lambda s: int(s.split("-")[0]),
    )
    cap_history = fetch_cap_history()
    growth_rate = compute_recent_cap_growth_rate(cap_history)

    # BR ids (player_id) are shared between the bref_advanced season file
    # and the contracts table - both come from the same site - so this
    # join needs no fuzzy name matching, unlike the RAPM join above
    merged = ages.merge(contracts, on="player_id", how="inner", suffixes=("", "_contract"))

    historical = load_all_seasons()
    transitions = build_transitions(historical)
    curve = build_curve(transitions)
    curve, cutoff_age = apply_reliability_cutoff(curve, transitions)

    rows = []
    skipped_no_contract_years = []
    for _, row in merged.iterrows():
        active_years = years_left_columns(row, year_cols)
        if not active_years:
            skipped_no_contract_years.append(row["Player"])
            continue

        avg_value = average_projected_value(curve, row["Age"], row["rapm"], len(active_years))
        avg_cap_pct = compute_avg_cap_pct(
            [row[c] for c in active_years], active_years, cap_history, growth_rate
        )
        rows.append({
            "player": row["Player"],
            "age": row["Age"],
            "rapm": row["rapm"],
            "years_left": len(active_years),
            "avg_value": avg_value,
            "avg_cap_pct": avg_cap_pct,
        })

    if skipped_no_contract_years:
        print(
            f"WARNING: {len(skipped_no_contract_years)} matched players had no usable "
            f"contract years: {skipped_no_contract_years}"
        )

    table = pd.DataFrame(rows)
    table = add_surplus_score(table)
    return table.sort_values("surplus_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    table = build_surplus_table()
    pd.set_option("display.width", 200)

    print(f"\n{len(table)} players ranked\n")
    display_cols = ["player", "age", "years_left", "rapm", "avg_value", "avg_cap_pct", "surplus_score"]

    print("Best value contracts (top 15):")
    print(table[display_cols].head(15).to_string(index=False))

    print("\nWorst value contracts (bottom 15):")
    print(table[display_cols].tail(15).to_string(index=False))

    out_path = PROCESSED_DIR / "surplus_value.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")
