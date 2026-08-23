"""Combine three separate, non-blended signals per player: their real
2025-26 RAPM (Phase 1, untouched), their real 2025-26 BPM, and where the
aging curve says their BPM is trending over the next few years. Nothing
here converts one metric into another - it's three honest numbers shown
side by side."""

from pathlib import Path

import pandas as pd

from src.aging_curve import (
    apply_reliability_cutoff,
    build_curve,
    build_transitions,
    load_all_seasons,
    project_bpm_with_band,
)
from src.name_matching import match_names

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bref_advanced"

MIN_CURRENT_SEASON_MINUTES = 500
YEARS_FORWARD = 3


def load_current_season(season: str = "2025-26") -> pd.DataFrame:
    df = pd.read_json(RAW_DIR / f"{season}.json")
    return df[df["MP"] >= MIN_CURRENT_SEASON_MINUTES].copy()


def build_report(years_forward: int = YEARS_FORWARD) -> pd.DataFrame:
    rapm = pd.read_csv(PROCESSED_DIR / "rapm_2025-26.csv")
    current = load_current_season()

    matched, unmatched = match_names(list(current["Player"]), list(rapm["name"]))
    if unmatched:
        print(f"WARNING: {len(unmatched)} current-season players couldn't be matched to RAPM: {unmatched}")
    current["rapm_name"] = current["Player"].map(matched)
    current = current.dropna(subset=["rapm_name"])
    current = current.merge(rapm[["name", "rapm"]], left_on="rapm_name", right_on="name", how="left")

    historical = load_all_seasons()
    transitions = build_transitions(historical)
    curve = build_curve(transitions)
    curve, cutoff_age = apply_reliability_cutoff(curve, transitions)

    rows = []
    for _, row in current.iterrows():
        projections = project_bpm_with_band(
            transitions, cutoff_age,
            current_age=row["Age"], current_bpm=row["BPM"], years_forward=years_forward,
            seed=42,
        )
        out = {
            "player": row["Player"],
            "age": row["Age"],
            "rapm": row["rapm"],
            "current_bpm": row["BPM"],
        }
        for step in projections:
            n = step["age"] - row["Age"]
            out[f"bpm_+{n}y"] = round(step["projected_bpm"], 2)
            out[f"bpm_+{n}y_band"] = f"[{step['band_low']:.1f}, {step['band_high']:.1f}] ({step['confidence_pct']:.0f}%)"
        rows.append(out)

    return pd.DataFrame(rows).sort_values("rapm", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    report = build_report()
    pd.set_option("display.width", 200)

    print(f"{len(report)} players (>= {MIN_CURRENT_SEASON_MINUTES} min this season)\n")

    display_cols = ["player", "age", "rapm", "current_bpm", "bpm_+1y", "bpm_+3y"]
    print("Top 15 by current RAPM:")
    print(report[display_cols].head(15).to_string(index=False))

    # note: at any single age, the curve applies roughly the same average
    # bump to everyone - so this mostly just re-sorts young players by who
    # already has the best current BPM, not a "hidden gem" detector. Real
    # over/under-valued detection needs a baseline to compare against
    # (draft slot, role, minutes trend), which belongs in a later phase.
    print("\nYoung players (age <= 24), current BPM vs +3y projection:")
    young = report[report["age"] <= 24].copy()
    young["projected_gain"] = young["bpm_+3y"] - young["current_bpm"]
    climbers = young.sort_values("projected_gain", ascending=False).head(15)
    print(climbers[["player", "age", "rapm", "current_bpm", "bpm_+3y", "projected_gain"]].to_string(index=False))

    out_path = PROCESSED_DIR / "player_report_2025-26.csv"
    report.to_csv(out_path, index=False)
    print(f"\nSaved full report to {out_path}")
