"""Train the full MoBEK model (1,048 compound + 1,700 protein = 2,748-d
Random Forest, exact v1-v11 hyperparameters).

Usage:
  python scripts/train_model.py --data data/final/train_deduplicated.csv \
      --out results/trained/full_train [--bootstrap 0]

Bootstrap setting (--bootstrap, default 0):
  0   -> ONE model trained on the FULL data file (no resampling)
  B>0 -> B bootstrap replicates: b=0 is the strict full-data model,
         b=1..B-1 resample rows with RandomState(seed + b)
         (the v5-v11 convention)

Output (in --out):
  model_b0.joblib, [model_b1.joblib, ...]   (joblib-dumped RFs)
  run_config.json                            (provenance)

No evaluation here — use scripts/evaluate_model.py on holdout data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="CSV with smiles,kinase,label + 1,048+1,700 features")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="0 = one full-data model (default); B>0 = B "
                         "bootstrap replicates (b=0 strict, seeds "
                         "seed+b)")
    ap.add_argument("--seed", type=int, default=M.DEFAULT_SEED)
    ap.add_argument("--n-estimators", type=int, default=M.DEFAULT_N_ESTIMATORS)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="OpenMP threads inside each RF fit")
    args = ap.parse_args()

    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(args.data)
    X = M.feature_matrix(df)          # validates the 2,748-d contract
    y = df["label"].to_numpy()
    print(f"loaded {len(df)} rows, {X.shape[1]} features, "
          f"{int(y.sum())} positives ({y.mean():.4f})")

    n_models = max(1, args.bootstrap)
    for b in range(n_models):
        rb = time.time()
        idx = M.bootstrap_index(len(df), b, args.seed)
        m = M.make_rf(args.n_estimators, args.seed, args.n_jobs)
        m.fit(X[idx], y[idx])
        path = out / f"model_b{b}.joblib"
        joblib.dump(m, path)
        tag = " (strict full data)" if b == 0 else f" (seed {args.seed + b})"
        print(f"b={b}{tag}: saved {path.name} "
              f"({time.time() - rb:.0f}s)", flush=True)

    cfg = dict(data=str(args.data), n_rows=int(len(df)),
               n_features=int(X.shape[1]), n_positives=int(y.sum()),
               n_estimators=args.n_estimators, seed=args.seed,
               bootstrap=int(args.bootstrap),
               seeds=[args.seed + b for b in range(n_models)],
               n_jobs=args.n_jobs,
               train_seconds=round(time.time() - t0, 1),
               model_files=[f"model_b{b}.joblib" for b in range(n_models)],
               note="full MoBEK: 1,048 compound + 1,700 protein; "
                    "b=0 is always the strict full-data model")
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"DONE in {time.time() - t0:.0f}s -> {out}/ "
          f"({n_models} model(s) + run_config.json)")


if __name__ == "__main__":
    main()
