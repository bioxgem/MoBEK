"""Interpretation for the full MoBEK model (feature importance + SHAP).

Refactor of the interpretation part of the original
``train_evaluate_and_interpret_mobek.py`` (feature-importance ranking,
global SHAP summary, single-prediction SHAP explanation — the original
demonstrated on row 1520 of train.csv, an ABL1-active pyridopyrimidine).

Modernized:
- modern ``shap`` API (>= 0.40): ``shap.TreeExplainer``; no ``shap.initjs``
  (Jupyter-only, removed upstream), no legacy ``shap.Explainer(model)``
- headless-safe (matplotlib Agg), no IPython dependency
- reproducible background subsampling (``random_state``)
- works with any model produced by ``scripts/train_model.py``
  (``model_b*.joblib``)

SHAP notes
----------
Tree SHAP values for an sklearn classifier are in **log-odds space**:
``base + sum(shap_values) == log(p1 / p0)``. The force plot / single-row
table below use the class-1 base value and class-1 SHAP values, matching
the convention of the original script (``expected_value[1]``).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from model import ID_COLS, feature_columns  # src/model.py of the refac repo


# ---------------------------------------------------------------------------
# model + data helpers
# ---------------------------------------------------------------------------
def load_model(path) -> RandomForestClassifier:
    import joblib

    m = joblib.load(path)
    if not isinstance(m, RandomForestClassifier):
        raise TypeError(f"{path} is not a fitted RandomForestClassifier")
    return m


def load_features(df: pd.DataFrame):
    """(X: n x 2748 float32 ndarray, feature_names) from a MoBEK data CSV."""
    names = feature_columns(df)
    return df[names].to_numpy(dtype=np.float32), names


def background_index(n_rows: int, n_bg: int, seed: int) -> np.ndarray:
    n_bg = min(n_bg, n_rows)
    if n_bg == n_rows:
        return np.arange(n_rows)
    return np.random.RandomState(seed).choice(n_rows, size=n_bg, replace=False)


def subforest(model, k: int):
    """A fitted RF using only the first ``k`` trees of ``model``.

    Tree-SHAP cost is linear in the total tree-node count; an unpruned RF on
    ~150k rows has ~1e5 nodes per tree, so the full 100-tree forest costs
    ~3 s/row. The first-k subforest (k=10 -> ~0.3 s/row) gives essentially
    the same feature ranking for the GLOBAL summary. Single-row explanations
    should use the full model (one row is affordable).
    """
    k = min(k, model.n_estimators)
    if k == model.n_estimators:
        return model
    sub = model.__class__(**model.get_params())
    sub.n_estimators = k
    sub.estimators_ = model.estimators_[:k]
    for attr in ("classes_", "n_features_in_", "n_classes_", "oob_score_"):
        if hasattr(model, attr):
            setattr(sub, attr, getattr(model, attr))
    return sub


# ---------------------------------------------------------------------------
# 1. global feature importance (model attribute, no SHAP)
# ---------------------------------------------------------------------------
def feature_importance(model, feature_names, top_n: int | None = None) -> pd.DataFrame:
    imp = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    if top_n:
        return imp.head(top_n)
    return imp


# ---------------------------------------------------------------------------
# 2. global SHAP summary
# ---------------------------------------------------------------------------
def _shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """Class-1 SHAP values, tolerant to old (list) / new (array) APIs.

    Returns (n, d). Older shap returns [class0, class1]; some versions
    return a single (n, d, 2) array.
    """
    sv = explainer.shap_values(X)
    if isinstance(sv, (list, tuple)):  # [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim == 3:  # (n, d, n_classes)
        sv = sv[:, :, -1]
    return sv


def shap_summary(model, X: np.ndarray, feature_names, out_png: str,
                 sample_size: int = 1000, max_display: int = 30,
                 seed: int = 42, n_trees: int = 10) -> pd.DataFrame:
    """Top-feature summary plot + mean(|SHAP|) table over a data subsample.

    Uses a first-``n_trees`` subforest (tree-SHAP cost is linear in total
    node count; see ``subforest``).
    """
    idx = background_index(len(X), sample_size, seed)
    Xs = X[idx]
    explainer = shap.TreeExplainer(subforest(model, n_trees))
    sv = _shap_values(explainer, Xs)
    mean_abs = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    plt.figure(figsize=(8, 10))
    shap.summary_plot(sv, Xs, feature_names=feature_names,
                      max_display=max_display, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close("all")
    return mean_abs


# ---------------------------------------------------------------------------
# 3. single-prediction SHAP explanation (original: imatinib row 1520)
# ---------------------------------------------------------------------------
def shap_single(model, X: np.ndarray, feature_names, row: int,
                waterfall_png: str | None = None,
                max_display: int = 20) -> tuple[dict, pd.DataFrame]:
    """Per-feature SHAP values for one row (FULL model — one row is affordable).

    Returns (info, table) where info = dict(base_value, pred_proba, n_features)
    and table = DataFrame(feature, shap, value) sorted by |shap|.
    With ``waterfall_png``, also writes the modern replacement of the
    original script's ``shap.force_plot`` (legacy ``force`` was removed in
    shap >= 0.42; ``waterfall`` is the headless-safe equivalent).
    """
    Xrow = X[row][None, :]
    explainer = shap.TreeExplainer(model)
    sv = _shap_values(explainer, Xrow)[0]
    base = np.asarray(explainer.expected_value, dtype=np.float64)
    base1 = base[-1] if base.ndim > 0 else base
    p1 = float(model.predict_proba(Xrow)[0, 1])
    table = (
        pd.DataFrame({"feature": feature_names, "shap": sv,
                      "value": Xrow[0]})
        .reindex(np.abs(sv).argsort()[::-1])
        .reset_index(drop=True)
    )
    if waterfall_png:
        expl = shap.Explanation(values=sv, base_values=float(base1),
                                data=Xrow[0], feature_names=feature_names)
        out = shap.plots.waterfall(expl, show=False, max_display=max_display)
        fig = out[0] if isinstance(out, tuple) else out
        if hasattr(fig, "figure"):  # matplotlib Axes -> its Figure
            fig = fig.figure
        fig.savefig(waterfall_png, dpi=150, bbox_inches="tight")
        plt.close("all")
    info = dict(base_value=float(base1), pred_proba_positive=p1,
                n_features=len(feature_names))
    return info, table
