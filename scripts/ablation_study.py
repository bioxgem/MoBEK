"""Ablation study over the full MoBEK feature set (2,748 = 1,048 + 1,700).

Disable whole blocks with --disable (repeatable / comma-separated):
    ecfp, moiety, checkmol, ac, stereo           (compound, 1,048)
    environment, type, evolution                 (protein, 1,700)
--no-protein is sugar for disabling all three protein blocks.

When the protein side is fully disabled the single global RF is replaced
by PER-KINASE training (one model per kinase, v10/v11 style aggregation:
micro = pooled, macro = per-kinase mean over defined metrics).

Usage:
  python scripts/ablation_study.py \
      --data data/train.csv --eval data/test.csv --eval data/independent.csv \
      --disable ecfp ...                                  # full model
  python scripts/ablation_study.py \
      --data data/train.csv --eval data/test.csv \
      --disable ecfp --disable stereo                      # ablations
  python scripts/ablation_study.py \
      --data data/train.csv --eval data/test.csv \
      --no-protein                                         # per-kinase mode

  # cheap sweep over b with fewer trees:
  for b in 0 1 2 3 4; do
    python scripts/ablation_study.py --data data/train.csv \
        --eval data/test.csv --no-protein --b $b --n-estimators 100 \
        --out-dir results/ablation
  done

Outputs:
  <out-dir>/ablation_<setting>_<evalname>_b<b>.csv   one row per run
  <out-dir>/ablation_summary.csv                     appended across runs
  <out-dir>/models_<setting>_b<b>/                   (with --save-models)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ablation as A   # noqa: E402
import metrics as Mx   # noqa: E402
import model as M      # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training CSV (MoBEK 2,748-d)")
    ap.add_argument("--eval", action="append", required=True, dest="evals",
                    help="eval slice CSV (repeatable)")
    ap.add_argument("--disable", action="append", default=[],
                    help="comma-separated blocks to disable "
                         "(ecfp,moiety,checkmol,ac,stereo,environment,type,evolution)")
    ap.add_argument("--no-protein", action="store_true",
                    help="disable environment+type+evolution "
                         "(switches to per-kinase training)")
    ap.add_argument("--mode", choices=("auto", "global", "per-kinase"),
                    default="auto",
                    help="auto: per-kinase iff protein fully disabled")
    ap.add_argument("--b", type=int, default=0,
                    help="bootstrap index (0 = strict full data; seed 42+b)")
    ap.add_argument("--n-estimators", type=int,
                    default=M.DEFAULT_N_ESTIMATORS)
    ap.add_argument("--seed", type=int, default=M.DEFAULT_SEED)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-models", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    disable: set[str] = set()
    for chunk in args.disable:
        disable.update(s.strip() for s in chunk.split(",") if s.strip())
    if args.no_protein:
        disable |= {"environment", "type", "evolution"}

    df_tr = pd.read_csv(args.data)
    cols_all = M.feature_columns(df_tr)
    cols = A.selected_columns(cols_all, disable)
    per_kinase = (A.protein_fully_disabled(disable)
                  if args.mode == "auto" else args.mode == "per-kinase")
    setting = A.setting_label(disable)
    print(f"setting={setting}  mode={'per-kinase' if per_kinase else 'global'}  "
          f"features={len(cols)}/2748  b={args.b}  "
          f"n_estimators={args.n_estimators}")

    # train
    t0 = time.time()
    if per_kinase:
        trained = A.train_per_kinase(df_tr, cols, args.b,
                                     args.n_estimators, args.seed, args.n_jobs)
        n_train_models = len(trained)
    else:
        trained = A.train_global(df_tr, cols, args.b,
                                 args.n_estimators, args.seed, args.n_jobs)
        n_train_models = 1
    print(f"trained {n_train_models} model(s) in {time.time() - t0:.0f}s")

    if args.save_models:
        import joblib
        mdir = Path(args.out_dir) / f"models_{setting}_b{args.b}"
        mdir.mkdir(parents=True, exist_ok=True)
        if per_kinase:
            for k, m in trained.items():
                joblib.dump(m, mdir / f"{k}.joblib")
        else:
            joblib.dump(trained, mdir / "model.joblib")
        print(f"models -> {mdir}/")

    # evaluate
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ev_path in args.evals:
        df_ev = pd.read_csv(ev_path)
        ev_name = Path(ev_path).stem
        if df_ev["label"].nunique() < 2 or df_ev["kinase"].nunique() < 2:
            print(f"WARNING: {ev_path} has only "
                  f"{df_ev['label'].nunique()} class(es) / "
                  f"{df_ev['kinase'].nunique()} kinase(s) — score metrics and "
                  f"macro will be undefined")
        t0 = time.time()
        if per_kinase:
            met, n_used = A.slice_per_kinase(trained, df_ev, cols)
        else:
            met, n_used = A.slice_global(trained, df_ev, cols)
        row = dict(setting=setting, mode="per-kinase" if per_kinase else "global",
                   b=args.b, eval=ev_name, n_features=len(cols),
                   n_train_models=n_train_models,
                   disabled=",".join(sorted(disable, key=A.BLOCK_ORDER.index))
                   if disable else "",
                   **{k: (round(v, 6) if np.isfinite(v) else np.nan)
                      for k, v in met.items()})
        rows.append(row)
        print(f"\n== {ev_name}  ({time.time() - t0:.0f}s)  "
              f"n_eval={met['n_eval']}  n_kinases={met['n_kinases']}")
        for k in Mx.METRICS:
            print(f"   micro_{k:8s} {row[f'micro_{k}']:.4f}   "
                  f"macro_{k:8s} {row[f'macro_{k}']:.4f}")

    frame = pd.DataFrame(rows)
    per_run = out_dir / f"ablation_{setting}_b{args.b}.csv"
    frame.to_csv(per_run, index=False)
    summary = out_dir / "ablation_summary.csv"
    frame.to_csv(summary, mode="a", header=not summary.exists(), index=False)
    print(f"\nDONE in {time.time() - t_start:.0f}s -> {per_run}, {summary}")


if __name__ == "__main__":
    main()
