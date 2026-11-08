"""
Unified Dataset Featurizer for MoBEK Compound-Kinase Interactions.

Input: CSV file containing `smiles`, `kinase`, and `label` columns.
Output: CSV or NPZ file with `smiles`, `kinase`, `label` + 2,748 feature columns:
  - Compound features (1,048 dims): ECFP(512) + Ring-in-drug(315) + Checkmol(204) + AC(10) + Stereo(7)
  - Protein features  (1,700 dims): 85 KLIFS positions × 20 bits (Type, Conservation, Micro-environment)

Project Structure Resolution:
  .
  ├── data/
  ├── scripts/
  │   └── build_dataset_features.py  <-- (This script)
  └── src/
      └── features/
          ├── compound.py
          ├── protein.py
          ├── Filter_BBA.csv
          └── vendors/
              ├── bin/               <-- Executables (checkmol, matchmol, mod_ac)
              └── moieties/          <-- Substructure pattern .mol files
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dynamic Path Setup & Imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJ_DIR / "src"
FEATURES_DIR = SRC_DIR / "features"

# Ensure project root & src are in Python path
for p in (PROJ_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.features.compound import featurize_smiles, _get_column_names as get_compound_cols
    from src.features.protein import featurize_protein, _get_column_names as get_protein_cols
except ImportError:
    from features.compound import featurize_smiles, _get_column_names as get_compound_cols
    from features.protein import featurize_protein, _get_column_names as get_protein_cols

# Project Default Paths
DEFAULT_DATA_DIR = PROJ_DIR / "data"
DEFAULT_BIN_DIR = FEATURES_DIR / "vendors" / "bin"
DEFAULT_MOIETY_DIR = FEATURES_DIR / "vendors" / "moieties"
DEFAULT_BBA_CSV = FEATURES_DIR / "Filter_BBA.csv"


def get_full_feature_column_names() -> list[str]:
    """Generates the full list of 2,748 feature column names (1048 compound + 1700 protein)."""
    compound_cols = get_compound_cols()
    protein_cols = get_protein_cols()[3:]  # Skip Group, Kinase, Uniprot metadata headers
    return compound_cols + protein_cols


def _feat_smi_worker(smi: str, bin_dir: Path, moiety_dir: Path) -> tuple[str, np.ndarray | None]:
    return smi, featurize_smiles(smi, bin_dir=bin_dir, moiety_dir=moiety_dir, n_threads=1)


def featurize_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    kinase_col: str = "kinase",
    bin_dir: Path | str | None = None,
    moiety_dir: Path | str | None = None,
    bba_path: Path | str | None = None,
    n_jobs: int = 8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Featurizes a dataset DataFrame with caching for unique SMILES and Kinases."""
    bin_dir = Path(bin_dir) if bin_dir else DEFAULT_BIN_DIR
    moiety_dir = Path(moiety_dir) if moiety_dir else DEFAULT_MOIETY_DIR
    bba_path = Path(bba_path) if bba_path else DEFAULT_BBA_CSV

    unique_smiles = df[smiles_col].dropna().unique()
    unique_kinases = df[kinase_col].dropna().unique()

    print(f"Dataset summary: {len(df)} rows | {len(unique_smiles)} unique SMILES | {len(unique_kinases)} unique Kinases")

    # 1. Featurize unique SMILES
    print("Featurizing unique SMILES...")
    smiles_cache: dict[str, np.ndarray | None] = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        n = len(unique_smiles)
        for smi, vec in tqdm(
            ex.map(_feat_smi_worker, unique_smiles,
                       [bin_dir] * n,
                       [moiety_dir] * n),
            total=len(unique_smiles)):
            smiles_cache[smi] = vec

    # 2. Featurize unique Kinases
    print("Featurizing unique Kinases...")
    kinase_cache: dict[str, np.ndarray | None] = {}
    for k in tqdm(unique_kinases):
        kinase_cache[str(k)] = featurize_protein(k, bba_path=bba_path)

    # 3. Assemble combined vectors
    print("Assembling combined feature matrix...")
    valid_mask = []
    combined_vectors = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row[smiles_col]
        kin = str(row[kinase_col])

        comp_vec = smiles_cache.get(smi)
        prot_vec = kinase_cache.get(kin)

        if comp_vec is not None and prot_vec is not None:
            valid_mask.append(True)
            combined_vectors.append(np.concatenate([comp_vec, prot_vec]))
        else:
            valid_mask.append(False)
            reasons = []
            if comp_vec is None:
                reasons.append(f"invalid SMILES '{smi}'")
            if prot_vec is None:
                reasons.append(f"unknown Kinase '{kin}'")
            print(f"Skipping row: {', '.join(reasons)}")

    valid_df = df[valid_mask].reset_index(drop=True)
    if not combined_vectors:
        return valid_df, np.empty((0, 2748), dtype=np.float32)

    features_matrix = np.vstack(combined_vectors)
    return valid_df, features_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Featurize a SMILES/Kinase/Label CSV dataset into a 2,748-feature matrix."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path containing smiles, kinase, and label columns")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV/NPZ file path. Defaults to data/<input_stem>_full_features.[csv|npz]"
    )
    parser.add_argument(
        "--save-as-npz", action="store_true", default=False,
        help="Save output in NPZ format instead of default CSV"
    )
    parser.add_argument("--smiles-col", type=str, default="smiles", help="Name of SMILES column (default: smiles)")
    parser.add_argument("--kinase-col", type=str, default="kinase", help="Name of Kinase column (default: kinase)")
    parser.add_argument("--label-col", type=str, default="label", help="Name of Label column (default: label)")
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR, help="Path to compound binaries folder")
    parser.add_argument("--moiety-dir", type=Path, default=DEFAULT_MOIETY_DIR, help="Path to moieties folder")
    parser.add_argument("--bba-path", type=Path, default=DEFAULT_BBA_CSV, help="Path to Filter_BBA.csv")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of worker threads")

    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    for col in (args.smiles_col, args.kinase_col, args.label_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in input CSV. Existing columns: {list(df.columns)}")

    valid_df, features_matrix = featurize_dataset(
        df,
        smiles_col=args.smiles_col,
        kinase_col=args.kinase_col,
        bin_dir=args.bin_dir,
        moiety_dir=args.moiety_dir,
        bba_path=args.bba_path,
        n_jobs=args.n_jobs,
    )

    if features_matrix.shape[0] == 0:
        print("Error: Failed to featurize any rows in dataset.")
        return

    ext = ".npz" if args.save_as_npz else ".csv"
    if args.output is None:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_DATA_DIR / f"{args.input_csv.stem}_full_features{ext}"
    else:
        out_path = args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_as_npz:
        np.savez_compressed(
            out_path,
            features=features_matrix,
            smiles=valid_df[args.smiles_col].to_numpy(),
            kinase=valid_df[args.kinase_col].to_numpy(),
            label=valid_df[args.label_col].to_numpy(),
        )
    else:
        feature_cols = get_full_feature_column_names()
        feat_df = pd.DataFrame(features_matrix, columns=feature_cols)

        # Output columns: smiles, kinase, label + 2,748 binary feature columns
        meta_df = valid_df[[args.smiles_col, args.kinase_col, args.label_col]].reset_index(drop=True)
        final_df = pd.concat([meta_df, feat_df], axis=1)
        final_df.to_csv(out_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Successfully featurized {len(valid_df)}/{len(df)} rows.")
    print(f"Output saved to: {out_path}")
    print(f"Output dimensions: {len(valid_df)} rows × {3 + features_matrix.shape[1]} columns")


if __name__ == "__main__":
    main()
