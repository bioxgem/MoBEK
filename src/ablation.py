"""Ablation study over the full MoBEK feature set (1,048 compound + 1,700 protein).

Disables whole feature blocks by name (order-robust):

  compound (1,048):  ecfp (512)  moiety (315: 168 Pubchem + 147 Ring)
                     checkmol (204)  ac (10)  stereo (7)
  protein  (1,700):  type (340, #AA_*_subs 1-4)
                     evolution (340, subs 5-8)
                     environment (1020, subs 9-20)

When the protein side is completely disabled (environment + type +
evolution all off), the single global model is switched to **per-kinase
training**: one RF per kinase, fit on that kinase's train rows only,
evaluated per kinase, then pooled (micro) and averaged (macro, strict
"defined only" convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import metrics as Mx
import model as M

BLOCK_ORDER = ("ecfp", "moiety", "checkmol", "ac", "stereo",
               "type", "evolution", "environment")

_AC_NAMES = ("C.ring", "C.other", "N.ring", "N.other", "O.ring", "O.other",
             "P", "S", "X", "#.of.Ring")
_STEREO_NAMES = ("count_@", "arr_con_m", "arr_con_mm", "arr_fir_m",
                 "arr_fir_mm", "count_/", "count_\\")

_EXPECTED = dict(ecfp=512, moiety=315, checkmol=204, ac=10, stereo=7,
                 type=340, evolution=340, environment=1020)


def _aa_sub(col: str) -> int:
    """Sub-index of a ``#AA_<pos>_<sub> <word>`` column."""
    head = col.split(" ", 1)[0]      # '#AA_<pos>_<sub>'
    return int(head.split("_")[2])


def classify_blocks(cols: list[str]) -> dict[str, list[str]]:
    """Map every feature column to its block; validates the counts."""
    blocks: dict[str, list[str]] = {b: [] for b in BLOCK_ORDER}
    for c in cols:
        if c.startswith("#ECFP_"):
            blocks["ecfp"].append(c)
        elif c.startswith("#Pubchem_") or c.startswith("#Ring_"):
            blocks["moiety"].append(c)
        elif c.startswith("#Checkmol_"):
            blocks["checkmol"].append(c)
        elif c in _AC_NAMES:
            blocks["ac"].append(c)
        elif c in _STEREO_NAMES:
            blocks["stereo"].append(c)
        elif c.startswith("#AA_"):
            s = _aa_sub(c)
            blocks["type" if s <= 4 else
                    "evolution" if s <= 8 else
                    "environment"].append(c)
        else:
            raise ValueError(f"unclassified feature column: {c!r}")
    for b, n in _EXPECTED.items():
        if len(blocks[b]) != n:
            raise ValueError(f"block {b!r}: expected {n} columns, "
                             f"got {len(blocks[b])}")
    return blocks


def selected_columns(cols: list[str], disable: set[str]) -> list[str]:
    """Columns of all blocks NOT in ``disable`` (original file order kept)."""
    bad = disable - set(BLOCK_ORDER)
    if bad:
        raise ValueError(f"unknown blocks to disable: {sorted(bad)}")
    blocks = classify_blocks(cols)
    keep = {c for b, cs in blocks.items() if b not in disable for c in cs}
    return [c for c in cols if c in keep]


def protein_fully_disabled(disable: set[str]) -> bool:
    return {"environment", "type", "evolution"} <= disable


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _fit(X, y, n_estimators, seed, n_jobs) -> RandomForestClassifier:
    m = M.make_rf(n_estimators, seed, n_jobs)
    m.fit(X, y)
    return m


def train_global(df: pd.DataFrame, cols: list[str], b: int,
                 n_estimators: int, seed: int, n_jobs: int):
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    idx = M.bootstrap_index(len(df), b, seed)
    return _fit(X[idx], y[idx], n_estimators, seed, n_jobs)


def train_per_kinase(df: pd.DataFrame, cols: list[str], b: int,
                     n_estimators: int, seed: int, n_jobs: int
                     ) -> dict[str, RandomForestClassifier]:
    """One RF per kinase; bootstrap is per-kinase (v10/v11 convention)."""
    models: dict[str, RandomForestClassifier] = {}
    X = df[cols].to_numpy(dtype=np.float32)
    for kin, sub in df.groupby("kinase", sort=True):
        rows = sub.index.to_numpy()
        n = len(rows)
        if b == 0:
            sel = np.arange(n)
        else:
            sel = np.random.RandomState(seed + b).choice(n, size=n,
                                                         replace=True)
        y = sub["label"].to_numpy()[sel]
        if len(np.unique(y)) < 2:      # single-class train set: unusable
            continue
        models[str(kin)] = _fit(X[rows][sel], y, n_estimators, seed, n_jobs)
    return models


# ---------------------------------------------------------------------------
# evaluation (micro = pooled; macro = per-kinase mean, strict)
# ---------------------------------------------------------------------------
def _macro(per_kin: dict[str, dict]) -> dict:
    out = {}
    for m in Mx.METRICS:
        vals = [v[m] for v in per_kin.values()
                if v is not None and np.isfinite(v.get(m, np.nan))]
        out[f"macro_{m}"] = float(np.mean(vals)) if vals else np.nan
    return out


def evaluate_slice(pred_of_row, df_ev: pd.DataFrame) -> tuple[dict, int]:
    """Run metrics on one eval slice given a row-index -> (pred, p1) fn.

    Returns (metrics dict with micro_* and macro_* keys, n_models_used).
    """
    per_kin: dict[str, dict] = {}
    y_all, pred_all, p_all = [], [], []
    n_models = 0
    for kin, sub in df_ev.groupby("kinase", sort=True):
        rows = sub.index.to_numpy()
        res = pred_of_row(rows)
        if res is None:
            continue
        pred, p1 = res
        y = sub["label"].to_numpy()
        per_kin[str(kin)] = Mx.evaluate(y, pred, p1)
        y_all.append(y); pred_all.append(pred); p_all.append(p1)
        n_models += 1
    if not y_all:
        raise ValueError("no eval rows could be scored")
    y = np.concatenate(y_all); pred = np.concatenate(pred_all)
    p1 = np.concatenate(p_all)
    out = {f"micro_{k}": v for k, v in Mx.evaluate(y, pred, p1).items()}
    out.update(_macro(per_kin))
    out["n_eval"] = len(y)
    out["n_kinases"] = len(per_kin)
    return out, n_models


def slice_global(model, df_ev: pd.DataFrame, cols: list[str]):
    X = df_ev[cols].to_numpy(dtype=np.float32)
    pred = model.predict(X); p1 = model.predict_proba(X)[:, 1]
    def pred_of_row(rows):
        i = np.searchsorted(df_ev.index.to_numpy(), rows)
        return pred[i], p1[i]
    met, _ = evaluate_slice(pred_of_row, df_ev)
    return met, 1


def slice_per_kinase(models, df_ev: pd.DataFrame, cols: list[str]):
    def pred_of_row(rows):
        out_p, out_s = [], []
        ok = True
        for kin, sub in df_ev.loc[rows].groupby("kinase", sort=True):
            m = models.get(str(kin))
            if m is None:
                ok = False
                break
            X = sub[cols].to_numpy(dtype=np.float32)
            out_p.append(m.predict(X)); out_s.append(m.predict_proba(X)[:, 1])
        if not ok:
            return None
        return np.concatenate(out_p), np.concatenate(out_s)
    met, _ = evaluate_slice(pred_of_row, df_ev)
    return met, len(models)


def setting_label(disable: set[str]) -> str:
    return "full" if not disable else "full-" + ",".join(
        sorted(disable, key=BLOCK_ORDER.index))
