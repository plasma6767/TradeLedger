"""Fit a BPM aging curve: for every age N -> N+1 transition, average how
much BPM typically moves, pooled across all players who made that
transition. Stays entirely in BPM's own units - this does not touch RAPM
in any way, it only projects a player's own BPM forward."""

from pathlib import Path

import numpy as np
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


def weighted_mean_std(deltas: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = np.average(deltas, weights=weights)
    variance = np.average((deltas - mean) ** 2, weights=weights)
    return mean, np.sqrt(variance)


def band_coverage(deltas: np.ndarray, weights: np.ndarray, low: float, high: float) -> float:
    """What fraction of the real (weighted) data actually falls inside
    [low, high] - the measured confidence for that band, not an assumed
    one."""
    in_band = (deltas >= low) & (deltas <= high)
    return 100 * weights[in_band].sum() / weights.sum()


def build_curve(transitions: pd.DataFrame) -> pd.DataFrame:
    """One row per age_from: minutes-weighted average delta to the next
    age, the typical (weighted) spread around it, and what percentage of
    real players actually landed inside a mean +/- 1 spread band - that
    percentage is measured from the data, not a target we picked."""
    def summarize(g):
        deltas, weights = g["delta"].to_numpy(), g["weight"].to_numpy()
        mean, std = weighted_mean_std(deltas, weights)
        confidence_pct = band_coverage(deltas, weights, mean - std, mean + std)
        return pd.Series({
            "avg_delta": mean,
            "std_delta": std,
            "confidence_pct": confidence_pct,
            "n_transitions": len(g),
        })

    curve = transitions.groupby("age_from").apply(summarize, include_groups=False)
    return curve.reset_index().sort_values("age_from")


# below this many player-transitions, an age's own delta is too small a
# sample to trust on its own (e.g. age 41->42 above is a single player's
# one outlier year) - ages past the cutoff share one pooled late-decline
# rate instead of each claiming their own number
MIN_RELIABLE_TRANSITIONS = 20


def apply_reliability_cutoff(curve: pd.DataFrame, transitions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Ages with enough transitions keep their own delta/spread/confidence.
    Everything past the last reliably-sampled age gets replaced with one
    pooled average, spread, and confidence computed across all of them
    together, since none has enough sample size on its own."""
    reliable = curve[curve["n_transitions"] >= MIN_RELIABLE_TRANSITIONS]
    cutoff_age = int(reliable["age_from"].max())

    thin = transitions[transitions["age_from"] > cutoff_age]
    deltas, weights = thin["delta"].to_numpy(), thin["weight"].to_numpy()
    pooled_mean, pooled_std = weighted_mean_std(deltas, weights)
    pooled_confidence = band_coverage(deltas, weights, pooled_mean - pooled_std, pooled_mean + pooled_std)

    curve = curve.copy()
    past_cutoff = curve["age_from"] > cutoff_age
    curve.loc[past_cutoff, "avg_delta"] = pooled_mean
    curve.loc[past_cutoff, "std_delta"] = pooled_std
    curve.loc[past_cutoff, "confidence_pct"] = pooled_confidence
    curve["reliable"] = curve["age_from"] <= cutoff_age
    return curve, cutoff_age


def project_bpm(curve: pd.DataFrame, current_age: float, current_bpm: float, years_forward: int) -> list[dict]:
    """Walk a player's own current BPM forward N years using just the
    average delta at each age - a single center-line trajectory, no
    band. See project_bpm_with_band for the uncertainty-aware version."""
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


def _build_delta_pools(transitions: pd.DataFrame, cutoff_age: int) -> dict:
    """One (deltas, weights) array pair per reliably-sampled age, plus one
    'thin' pool pooling everything past the cutoff - the same pools the
    point curve's averages were computed from, kept raw so we can resample
    from them instead of just reading off their mean."""
    pools = {}
    reliable = transitions[transitions["age_from"] <= cutoff_age]
    for age, g in reliable.groupby("age_from"):
        pools[age] = (g["delta"].to_numpy(), g["weight"].to_numpy())

    thin = transitions[transitions["age_from"] > cutoff_age]
    pools["thin"] = (thin["delta"].to_numpy(), thin["weight"].to_numpy())
    return pools


def project_bpm_with_band(
    transitions: pd.DataFrame,
    cutoff_age: int,
    current_age: float,
    current_bpm: float,
    years_forward: int,
    n_simulations: int = 5000,
    seed: int | None = None,
) -> list[dict]:
    """Simulate many plausible trajectories by repeatedly resampling a real
    player's actual delta at each age (weighted by minutes), rather than
    just applying the average every time. At each future year, the band is
    mean +/- 1 spread of the simulated outcomes, and confidence_pct is the
    percentage of simulated trajectories that actually landed inside that
    band - measured from the simulation, not assumed."""
    rng = np.random.default_rng(seed)
    pools = _build_delta_pools(transitions, cutoff_age)
    # clamp range must span ALL observed ages, not just the reliably-pooled
    # ones - otherwise ages past the cutoff get clamped back down to the
    # cutoff age and never actually reach the pooled "thin" fallback pool
    min_age = int(transitions["age_from"].min())
    max_age = int(transitions["age_from"].max())

    bpm_paths = np.full(n_simulations, current_bpm, dtype=float)
    age = current_age
    results = []
    for _ in range(years_forward):
        lookup_age = min(max(int(age), min_age), max_age)
        pool_key = lookup_age if lookup_age <= cutoff_age else "thin"
        deltas, weights = pools[pool_key]
        sampled = rng.choice(deltas, size=n_simulations, p=weights / weights.sum())
        bpm_paths = bpm_paths + sampled
        age += 1

        mean, std = bpm_paths.mean(), bpm_paths.std()
        low, high = mean - std, mean + std
        confidence_pct = 100 * np.mean((bpm_paths >= low) & (bpm_paths <= high))
        results.append({
            "age": age,
            "projected_bpm": mean,
            "band_low": low,
            "band_high": high,
            "confidence_pct": confidence_pct,
            "reliable": lookup_age <= cutoff_age,
        })
    return results


if __name__ == "__main__":
    df = load_all_seasons()
    print(f"{len(df)} player-seasons after the {MIN_MINUTES}-minute floor")

    transitions = build_transitions(df)
    print(f"{len(transitions)} player age-transitions")

    curve = build_curve(transitions)
    curve, cutoff_age = apply_reliability_cutoff(curve, transitions)
    print(f"\nreliability cutoff: age {cutoff_age} (last age with >= {MIN_RELIABLE_TRANSITIONS} transitions)")
    print("age -> next age: avg delta, spread, measured confidence of the avg +/- spread band")
    print(curve.to_string(index=False))

    print("\nexample: a 24-year-old with a 2.0 BPM season, simulated 5 years out")
    for step in project_bpm_with_band(transitions, cutoff_age, current_age=24, current_bpm=2.0, years_forward=5):
        flag = "" if step["reliable"] else "  (low confidence - pooled late-decline rate)"
        print(
            f"  age {step['age']}: {step['projected_bpm']:.2f}  "
            f"({step['confidence_pct']:.0f}% confidence: {step['band_low']:.2f} to {step['band_high']:.2f})"
            f"{flag}"
        )
