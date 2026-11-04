"""Evaluate trained full-MoBEK model(s) on a holdout data file.

Usage:
  python scripts/evaluate_model.py \
      --model results/trained/full_train \
      --data data/final/independent_deduplicated.csv \
      --out results/eval_independent.csv \
      [--per-kinase]

--model is a single model_b*.joblib file OR a directory (then all
model_b*.joblib inside, in b order — e.g. all bootstrap replicates).

Data contract: same as train_model.py (smiles,kinase,label + 1,048 +
1,700 features). Metrics: the 11-metric MoBEK suite (src/metrics.py),
pooled (micro) over all rows of the eval file — one row per model.

--per-kinase additionally writes <out>_per_kinase.csv with one row per
(model, kinase): counts + the 11 metrics (NaN where undefined, the
strict v10-v11 convention). Macro = mean over kinases where the metric
is defined; compute it from that file as needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model as M
import metrics as MET


def model_paths(spec: Path) -> list[Path]:
    if spec.is_file():
        return [spec]
    paths = sorted(spec.glob("model_b*.joblib"),
                   key=lambda p: int(p.stem.split("b")[1]))
    if not paths:
        raise FileNotFoundError(f"no model_b*.joblib under {spec}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="model_b*.joblib file or directory of them")
    ap.add_argument("--data", required=True,
                    help="eval CSV (smiles,kinase,label + 2,748 features)")
    ap.add_argument("--out", required=True, help="output metrics CSV")
    ap.add_argument("--per-kinase", action="store_true",
                    help="also write <out>_per_kinase.csv")
    ap.add_argument("--n-jobs", type=int, default=16,
                    help="OpenMP threads for predict/predict_proba")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = M.feature_matrix(df)
    y = df["label"].to_numpy()
    kin = df["kinase"].astype(str).to_numpy()
    print(f"eval: {len(df)} rows, {int(y.sum())} positives "
          f"({y.mean():.4f}), {len(np.unique(kin))} kinases")

    paths = model_paths(Path(args.model))
    rows, pk_rows = [], []
    for p in paths:
        m = joblib.load(p)
        m.set_params(n_jobs=args.n_jobs)
        pred = m.predict(X).astype(int)
        p1 = m.predict_proba(X)[:, 1]
        ev = MET.evaluate(y, pred, p1)
        row = dict(model=p.name)
        row.update(n=int(len(y)), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))
        for k in MET.METRICS:
            row[k] = ev[k]
        rows.append(row)
        if args.per_kinase:
            for k in np.unique(kin):
                sel = kin == k
                tp, fn, fp, tn = MET.counts(y[sel], pred[sel])
                a, bd = MET.score_metrics(y[sel], p1[sel])
                th = MET.threshold_metrics(tp, fn, fp, tn)
                pk_rows.append(dict(model=p.name, kinase=k,
                                    n=int(sel.sum()), tp=tp, fn=fn,
                                    fp=fp, tn=tn, **
                                    {c: th[c] for c in MET.THRESHOLD},
                                    auc=a, bedroc=bd))
        print(f"{p.name}: mcc={ev['mcc']:.4f} auc={ev['auc']:.4f} "
              f"f1={ev['f1']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out} ({len(rows)} model row(s))")
    if args.per_kinase:
        pkout = out.with_name(out.name + "_per_kinase.csv")
        pd.DataFrame(pk_rows).to_csv(pkout, index=False)
        print(f"saved {pkout} ({len(pk_rows)} rows)")


if __name__ == "__main__":
    main()
