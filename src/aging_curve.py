"""Fit a BPM aging curve: for every age N -> N+1 transition, average how
much BPM typically moves, pooled across all players who made that
transition. Stays entirely in BPM's own units - this does not touch RAPM
in any way, it only projects a player's own BPM forward."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bref_advanced"

MIN_MINUTES = 500


def load_all_seasons() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.json"))
    df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
    return df[df["MP"] >= MIN_MINUTES].copy()


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player per age transition they actually made (age N and
    age N+1 both present, both above the minutes floor)."""
    df = df.sort_values(["player_id", "Age"])
    rows = []
    for player_id, g in df.groupby("player_id"):
        g = g.drop_duplicates(subset="Age").set_index("Age")
        for age in g.index:
            if (age + 1) in g.index:
                rows.append({
                    "player_id": player_id,
                    "age_from": age,
                    "bpm_from": g.loc[age, "BPM"],
                    "bpm_to": g.loc[age + 1, "BPM"],
                    "delta": g.loc[age + 1, "BPM"] - g.loc[age, "BPM"],
                    "weight": min(g.loc[age, "MP"], g.loc[age + 1, "MP"]),
                })
    return pd.DataFrame(rows)


def build_curve(transitions: pd.DataFrame) -> pd.DataFrame:
    """One row per age_from: minutes-weighted average delta to the next
    age, plus the raw count of player-transitions behind it (so thin,
    low-sample ages at the tails are visible, not hidden)."""
    def weighted_avg(g):
        return (g["delta"] * g["weight"]).sum() / g["weight"].sum()

    curve = transitions.groupby("age_from").apply(
        lambda g: pd.Series({
            "avg_delta": weighted_avg(g),
            "n_transitions": len(g),
        }),
        include_groups=False,
    )
    return curve.reset_index().sort_values("age_from")


# below this many player-transitions, an age's own delta is too small a
# sample to trust on its own (e.g. age 41->42 above is a single player's
# one outlier year) - ages past the cutoff share one pooled late-decline
# rate instead of each claiming their own number
MIN_RELIABLE_TRANSITIONS = 20


def apply_reliability_cutoff(curve: pd.DataFrame, transitions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Ages with enough transitions keep their own delta. Everything past
    the last reliably-sampled age gets replaced with one minutes-weighted
    average pooled across all of them, since none has enough sample size
    on its own."""
    reliable = curve[curve["n_transitions"] >= MIN_RELIABLE_TRANSITIONS]
    cutoff_age = int(reliable["age_from"].max())

    thin = transitions[transitions["age_from"] > cutoff_age]
    pooled_delta = (thin["delta"] * thin["weight"]).sum() / thin["weight"].sum()

    curve = curve.copy()
    curve.loc[curve["age_from"] > cutoff_age, "avg_delta"] = pooled_delta
    curve["reliable"] = curve["age_from"] <= cutoff_age
    return curve, cutoff_age


def project_bpm(curve: pd.DataFrame, current_age: float, current_bpm: float, years_forward: int) -> list[dict]:
    """Walk a player's own current BPM forward N years using the curve.
    Returns a dict per year with the projected BPM and whether that step
    used a reliably-sampled age or the pooled late-decline fallback."""
    deltas = curve.set_index("age_from")["avg_delta"]
    reliable = curve.set_index("age_from")["reliable"]
    min_age, max_age = deltas.index.min(), deltas.index.max()

    projections = []
    age, bpm = current_age, current_bpm
    for _ in range(years_forward):
        lookup_age = min(max(int(age), min_age), max_age)
        bpm = bpm + deltas.loc[lookup_age]
        age += 1
        projections.append({"age": age, "bpm": bpm, "reliable": bool(reliable.loc[lookup_age])})
    return projections


if __name__ == "__main__":
    df = load_all_seasons()
    print(f"{len(df)} player-seasons after the {MIN_MINUTES}-minute floor")

    transitions = build_transitions(df)
    print(f"{len(transitions)} player age-transitions")

    curve = build_curve(transitions)
    curve, cutoff_age = apply_reliability_cutoff(curve, transitions)
    print(f"\nreliability cutoff: age {cutoff_age} (last age with >= {MIN_RELIABLE_TRANSITIONS} transitions)")
    print("age -> next age: avg BPM delta (n transitions behind it, reliable?)")
    print(curve.to_string(index=False))

    print("\nexample: a 24-year-old with a 2.0 BPM season, projected 5 years out")
    for step in project_bpm(curve, current_age=24, current_bpm=2.0, years_forward=5):
        flag = "" if step["reliable"] else "  (low confidence - pooled late-decline rate)"
        print(f"  age {step['age']}: {step['bpm']:.2f}{flag}")
