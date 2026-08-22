"""Fit a ridge-regression RAPM model from the possession-level dataset.

One row per possession, one column per player (+1 offense, -1 defense, 0
not playing), target is net points for that possession. Lambda (ridge's
penalty strength) is chosen by cross-validation, splitting by GAME so
possessions from the same game never end up on both sides of a split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def build_player_name_map():
    """GameRotation includes each player's name directly, and we've
    already cached it for every game - no separate name lookup needed,
    and unlike nba_api's offline static player list, this covers brand
    new rookies from the current season."""
    names = {}
    for f in (RAW_DIR / "rotation").glob("*.json"):
        for row in json.loads(f.read_text()):
            names[row["PERSON_ID"]] = f"{row['PLAYER_FIRST']} {row['PLAYER_LAST']}"
    return names
PER_GAME_DIR = PROCESSED_DIR / "by_game"


def load_possessions(season="2025-26"):
    combined_path = PROCESSED_DIR / f"possessions_{season}.parquet"
    if combined_path.exists():
        df = pd.read_parquet(combined_path)
    else:
        files = sorted(PER_GAME_DIR.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["offense_players"] = df["offense_players"].apply(lambda s: [int(x) for x in s.split(",")])
    df["defense_players"] = df["defense_players"].apply(lambda s: [int(x) for x in s.split(",")])
    return df


def build_design_matrix(df):
    all_players = sorted(
        set(p for lst in df["offense_players"] for p in lst)
        | set(p for lst in df["defense_players"] for p in lst)
    )
    player_to_col = {p: i for i, p in enumerate(all_players)}

    rows_idx, cols_idx, vals = [], [], []
    for row_i, (off_list, def_list) in enumerate(zip(df["offense_players"], df["defense_players"])):
        for p in off_list:
            rows_idx.append(row_i)
            cols_idx.append(player_to_col[p])
            vals.append(1)
        for p in def_list:
            rows_idx.append(row_i)
            cols_idx.append(player_to_col[p])
            vals.append(-1)

    X = sparse.csr_matrix((vals, (rows_idx, cols_idx)), shape=(len(df), len(all_players)))
    y = df["points"].to_numpy(dtype=float)
    groups = df["game_id"].to_numpy()
    return X, y, player_to_col, groups


def cross_validate_lambda(X, y, groups, lambdas, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    baseline_errors = []
    results = {}

    for lam in lambdas:
        fold_errors = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = Ridge(alpha=lam)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            fold_errors.append(mean_squared_error(y[test_idx], preds))
            if lam == lambdas[0]:
                # baseline: always predict the training mean, computed once
                baseline_pred = np.full(len(test_idx), y[train_idx].mean())
                baseline_errors.append(mean_squared_error(y[test_idx], baseline_pred))
        results[lam] = np.mean(fold_errors)

    return results, np.mean(baseline_errors)


def build_rapm_table(model, player_to_col):
    id_to_name = build_player_name_map()
    rows = [
        {"player_id": pid, "name": id_to_name.get(pid, f"Unknown({pid})"), "rapm": model.coef_[col] * 100}
        for pid, col in player_to_col.items()
    ]
    return pd.DataFrame(rows).sort_values("rapm", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Loading possessions...")
    df = load_possessions()
    print(f"{len(df)} possessions across {df['game_id'].nunique()} games")

    X, y, player_to_col, groups = build_design_matrix(df)
    print(f"design matrix: {X.shape[0]} rows x {X.shape[1]} players")

    lambdas = [1, 10, 100, 500, 1000, 3000, 5000, 10000, 30000]
    results, baseline_error = cross_validate_lambda(X, y, groups, lambdas)

    print("\nlambda -> cross-validated MSE:")
    for lam, err in results.items():
        print(f"  {lam:>7}: {err:.4f}")
    print(f"  baseline (always predict mean): {baseline_error:.4f}")

    best_lambda = min(results, key=results.get)
    print(f"\nbest lambda: {best_lambda}")

    final_model = Ridge(alpha=best_lambda)
    final_model.fit(X, y)
    table = build_rapm_table(final_model, player_to_col)

    print("\nTop 15:")
    print(table.head(15).to_string())
    print("\nBottom 15:")
    print(table.tail(15).to_string())
