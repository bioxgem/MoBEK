"""Interpret a trained full-MoBEK model (feature importance + SHAP).

Refactored counterpart of the interpretation part of the original
train_evaluate_and_interpret_mobek.py. Headless, reproducible, modern shap.

Usage (all flags optional; default runs importance + summary + the
original example row):
  python scripts/interpret_model.py \
      --model results/trained/full_train/model_b0.joblib \
      --data data/train.csv \
      --out-dir results/interpret_b0 \
      [--importance] [--top-n 50] \
      [--shap-summary --shap-sample 1000 --max-display 30] \
      [--shap-row 1520 | --shap-smiles "CN1CC(...)"] \
      [--force-html] [--seed 42]

Outputs (in --out-dir):
  feature_importance.csv          (all 2,748, or --top-n)
  shap_summary.png                (top --max-display features)
  shap_summary_mean_abs.csv       (mean |SHAP| ranking)
  shap_row{idx}.csv               (per-feature SHAP for one row)
  shap_row{idx}_waterfall.png     (with --waterfall)

Requires: shap (>= 0.40). e.g. pip install shap==0.46.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import interpret as I  # noqa: E402  (src/interpret.py of the refac repo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model_b*.joblib (from train_model.py)")
    ap.add_argument("--data", required=True, help="MoBEK data CSV (smiles,kinase,label + 2,748)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--importance", action="store_true", default=True,
                    help="write feature_importance.csv (on by default)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="limit importance CSV to top N (0 = all 2,748)")
    ap.add_argument("--shap-summary", action="store_true",
                    help="global SHAP summary plot")
    ap.add_argument("--shap-sample", type=int, default=1000,
                    help="rows used for the SHAP summary (default 1000)")
    ap.add_argument("--max-display", type=int, default=30,
                    help="features shown in the summary plot")
    ap.add_argument("--shap-trees", type=int, default=10,
                    help="trees used for the SUMMARY (tree-SHAP is linear in "
                         "total node count; the unpruned 100-tree forest is "
                         "~3 s/row). Single-row explanations always use the "
                         "full model.")
    ap.add_argument("--shap-row", type=int, default=None,
                    help="0-based row index to explain (original script: 1520)")
    ap.add_argument("--shap-smiles", type=str, default=None,
                    help="explain the first row matching this SMILES")
    ap.add_argument("--waterfall", action="store_true",
                    help="also write the single-row waterfall plot (PNG; "
                         "modern replacement of the original force_plot)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = I.load_model(args.model)
    df = pd.read_csv(args.data)
    X, names = I.load_features(df)
    print(f"loaded model ({args.model}) and {len(df)} rows x {X.shape[1]} features "
          f"in {time.time() - t0:.0f}s")

    meta = {"model": str(args.model), "data": str(args.data),
            "n_rows": int(len(df)), "n_features": int(X.shape[1]),
            "seed": args.seed}

    # 1) feature importance
    if args.importance:
        imp = I.feature_importance(model, names, top_n=args.top_n or None)
        p = out / "feature_importance.csv"
        imp.to_csv(p, index=False)
        print(f"importance -> {p} (top3: {', '.join(map(str, imp.feature.head(3)))})")

    # 2) global SHAP summary
    if args.shap_summary:
        p_png = out / "shap_summary.png"
        t0 = time.time()
        mabs = I.shap_summary(model, X, names, str(p_png),
                              sample_size=args.shap_sample,
                              max_display=args.max_display, seed=args.seed,
                              n_trees=args.shap_trees)
        p_csv = out / "shap_summary_mean_abs.csv"
        mabs.to_csv(p_csv, index=False)
        meta["shap"] = dict(sample=args.shap_sample, max_display=args.max_display,
                            trees=args.shap_trees,
                            seconds=round(time.time() - t0, 1),
                            top5=list(mabs.feature.head(5)))
        print(f"shap summary ({args.shap_sample} rows) in {time.time() - t0:.0f}s "
              f"-> {p_png}, {p_csv}")
        print("  top5 by mean|SHAP|:", ", ".join(mabs.feature.head(5)))

    # 3) single-prediction SHAP (original script's demo: row 1520)
    row = args.shap_row
    if args.shap_smiles is not None:
        hit = df.index[df.smiles == args.shap_smiles]
        if len(hit) == 0:
            ap.error(f"SMILES not found in {args.data}: {args.shap_smiles}")
        row = int(hit[0])
    if row is not None:
        info, table = I.shap_single(
            model, X, names, row,
            waterfall_png=str(out / f"shap_row{row}_waterfall.png")
            if args.waterfall else None,
            max_display=args.max_display)
        r = df.iloc[row]
        p = out / f"shap_row{row}.csv"
        table.to_csv(p, index=False)
        meta["shap_row"] = dict(row=row, smiles=r["smiles"],
                                kinase=r["kinase"], label=int(r["label"]),
                                **info, top5=list(table.feature.head(5)))
        print(f"\nshap row {row}: {r['smiles'][:60]}...  kinase={r['kinase']} "
              f"label={int(r['label'])}")
        print(f"  base(log-odds)={info['base_value']:.4f}  "
              f"P(+)={info['pred_proba_positive']:.4f}  -> {p}")
        print("  top5 drivers:", ", ".join(
            f"{f}({s:+.3f})" for f, s in zip(table.feature.head(5), table.shap.head(5))))

    meta["seconds_total"] = round(time.time() - meta.get("shap", {}).get("seconds", 0) + 0, 1)
    (out / "interpret_config.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
