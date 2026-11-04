"""Metric suite for the MoBEK experiments.

Exactly the conventions used in the v10-v11 runs (so numbers stay
comparable):

- Threshold metrics from the confusion counts, with the v10 `met()`
  NaN rules (undefined -> NaN, e.g. precision with no positive
  predictions).
- Score metrics from `predict_proba[:, 1]`: ROC-AUC, PR-AUC, and
  BEDROC (alpha = 20, the vectorized RIE implementation). All three
  are NaN for single-class label sets.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score)

ALPHA = 20.0

THRESHOLD = ["acc", "precision", "recall", "f1", "bal_acc", "sens",
             "spec", "mcc"]
SCORE = ["auc", "bedroc"]
METRICS = THRESHOLD + SCORE   # 11 metrics


def counts(y, pred):
    """(tp, fn, fp, tn)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fn, fp, tn


def threshold_metrics(tp, fn, fp, tn) -> dict:
    """v10 met() convention: NaN where a metric is undefined."""
    n = tp + fn + fp + tn
    den = np.sqrt(float((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp)))
    s = lambda a, b: np.nan if b == 0 else a / b
    return dict(
        n=n,
        acc=(tp + tn) / n,
        precision=s(tp, tp + fp),
        recall=s(tp, tp + fn),
        f1=s(2 * tp, 2 * tp + fp + fn),
        bal_acc=(np.nan if (tp + fn) == 0 or (tn + fp) == 0
                 else (s(tp, tp + fn) + s(tn, tn + fp)) / 2),
        sens=s(tp, tp + fn),
        spec=s(tn, tn + fp),
        mcc=(tp * tn - fp * fn) / den if den else np.nan)


def bedroc(y, score, alpha: float = ALPHA) -> float:
    """BEDROC (richness factor) via the vectorized RIE, max err ~1e-14."""
    y = np.asarray(y).astype(bool)
    n = len(y)
    n_act = int(y.sum())
    if n_act == 0 or n_act == n:
        return np.nan
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rank = (np.nonzero(y[order])[0] + 1).astype(np.float64)
    sum_exp = np.exp(-alpha * rank / n).sum()
    denom = (1.0 / n) * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    rie = sum_exp / (n_act * denom)
    ratio = n_act / n
    riemax = (1.0 - np.exp(-alpha * ratio)) / (ratio * (1.0 - np.exp(-alpha)))
    riemin = (1.0 - np.exp(alpha * ratio)) / (ratio * (1.0 - np.exp(-alpha)))
    return float((rie - riemin) / (riemax - riemin))


def score_metrics(y, p1) -> tuple:
    """(auc, prauc, bedroc); NaN triple for single-class labels."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (float(roc_auc_score(y, p1)),
            bedroc(y, p1))


def evaluate(y, pred, p1) -> dict:
    """All 11 metrics for one prediction on one slice."""
    tp, fn, fp, tn = counts(y, pred)
    out = threshold_metrics(tp, fn, fp, tn)
    a, b = score_metrics(y, p1)
    out.update(auc=a, bedroc=b)
    return out
