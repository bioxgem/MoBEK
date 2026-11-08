"""Full MoBEK model — the single canonical model definition.

The full MoBEK setting: compound features (1,048-d) + protein features
(1,700-d) = 2,748 inputs, Random Forest with the exact hyperparameters
used throughout the v1-v11 experiments:

    RandomForestClassifier(bootstrap=True, max_depth=None,
                           max_features="sqrt", min_samples_leaf=1,
                           min_samples_split=2, n_estimators=100,
                           random_state=42)

Bootstrap convention (identical to v5-v11):
    b = 0  -> train on the FULL dataset (no resampling), "strict"
    b > 0  -> bootstrap sample via RandomState(seed + b).choice(n, n,
             replace=True)

Data contract: any CSV whose non-ID columns are exactly the released
2,748 feature columns (first 1,048 compound, last 1,700 protein
`#AA_*`), plus the ID columns `smiles`, `kinase`, `label`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

N_COMP = 1048          # released compound block (1041 non-stereo + 7 stereo)
N_PROT = 1700          # released protein block (#AA_*)
N_FEAT = N_COMP + N_PROT   # 2748
ID_COLS = ("smiles", "kinase", "label")

DEFAULT_N_ESTIMATORS = 100
DEFAULT_SEED = 42


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All non-ID columns; raises unless the data is exactly 2,748-d."""
    cols = [c for c in df.columns if c not in ID_COLS]
    if len(cols) != N_FEAT:
        raise ValueError(
            f"expected {N_FEAT} feature columns "
            f"({N_COMP} compound + {N_PROT} protein), got {len(cols)}")
    return cols


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float32)
    if np.isnan(X).any():
        raise ValueError("feature matrix contains NaNs")
    return X


def make_rf(n_estimators: int = DEFAULT_N_ESTIMATORS,
            seed: int = DEFAULT_SEED,
            n_jobs: int = 1) -> RandomForestClassifier:
    """The full MoBEK Random Forest (unfitted)."""
    return RandomForestClassifier(
        bootstrap=True, max_depth=None, max_features="sqrt",
        min_samples_leaf=1, min_samples_split=2,
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)


def bootstrap_index(n: int, b: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Row index for replicate b (0 = strict full data)."""
    if b == 0:
        return np.arange(n)
    return np.random.RandomState(seed + b).choice(n, size=n,
                                                  replace=True)


def train_one(df: pd.DataFrame, b: int,
              n_estimators: int = DEFAULT_N_ESTIMATORS,
              seed: int = DEFAULT_SEED,
              n_jobs: int = 1) -> RandomForestClassifier:
    """Fit the full MoBEK RF on replicate b of `df`."""
    X = feature_matrix(df)
    y = df["label"].to_numpy()
    idx = bootstrap_index(len(df), b, seed)
    m = make_rf(n_estimators, seed, n_jobs)
    m.fit(X[idx], y[idx])
    return m
