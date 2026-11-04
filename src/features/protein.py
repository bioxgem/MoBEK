"""Unified Kinase Binding Site Featurizer for MoBEK.

Generates 1,700-dimensional binary feature vectors per kinase (85 KLIFS positions × 20 bits):
  - AA Type (4 bits: Basic, Acidic, Non-polar, Polar)
  - Evolutionary Conservation (4 bits thermometric binning)
  - Local Environment (12 bits: 3 bits × 4 property counts in ±3 window)

Path resolution is relative to this file's position in `MoBEK/src/features/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Absolute Path Agnostic Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJ_DIR = SRC_DIR.parent

DEFAULT_BBA_CSV = SCRIPT_DIR / "Filter_BBA.csv"
DEFAULT_DATA_DIR = PROJ_DIR / "data"

# 85 KLIFS Position Indices (1-based relative to alignment sequence)
BBA_85 = [
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 55, 56, 57, 58, 59, 60,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 173, 174, 175, 176, 177, 178, 179, 180, 182,
    183, 184, 185, 186, 187, 188, 189, 190, 191, 312, 313, 314, 315, 316, 317,
    318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 361, 362,
    363, 364, 365, 366, 367
]

# Hardcoded 85 KLIFS region labels (replaces KLIFS_85_feature_list.txt)
KLIFS_REGIONS = (
    "b-sheets", "b-sheets", "b-sheets", "G-rich", "G-rich", "G-rich", "G-rich", "G-rich", "G-rich",
    "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets",
    "b-sheets", "b-sheets", "aC-helix", "aC-helix", "aC-helix", "aC-helix", "aC-helix", "aC-helix",
    "aC-helix", "aC-helix", "aC-helix", "aC-helix", "aC-helix", "b.l", "b.l", "b.l", "b.l", "b.l",
    "b.l", "b.l", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets", "b-sheets",
    "GK", "hinge", "hinge", "hinge", "linker", "linker", "linker", "linker", "aD-helix", "aD-helix",
    "aD-helix", "aD-helix", "aD-helix", "aD-helix", "aD-helix", "aE-helix", "aE-helix", "aE-helix",
    "aE-helix", "aE-helix", "b-sheets", "b-sheets", "b-sheets", "catalytic", "catalytic", "catalytic",
    "catalytic", "catalytic", "catalytic", "catalytic", "catalytic", "b-sheets", "b-sheets", "b-sheets",
    "b-sheets", "x", "DFG", "DFG", "DFG", "activation", "activation"
)

# Amino acid property groupings
BASIC = set("KR")
ACIDIC = set("DE")
NONPOLAR = set("AFILMPVW")
POLAR = set("CGHNQSTY")

_BBA_CACHE: dict = {}


def _load_bba_data(bba_path: Path) -> dict:
    if "df" not in _BBA_CACHE or _BBA_CACHE.get("path") != bba_path:
        if not bba_path.exists():
            raise FileNotFoundError(f"Alignment file not found: {bba_path}")
        df = pd.read_csv(bba_path)
        
        # Build mapping for Kinase name and UniProt ID lookup
        kinase_to_idx = {str(k): i for i, k in enumerate(df["Kinase"])}
        uniprot_to_idx = {str(u): i for i, u in enumerate(df["Uniprot"])}
        
        _BBA_CACHE["df"] = df
        _BBA_CACHE["kinase_to_idx"] = kinase_to_idx
        _BBA_CACHE["uniprot_to_idx"] = uniprot_to_idx
        _BBA_CACHE["n_kinases"] = len(df)
        _BBA_CACHE["path"] = bba_path
        
    return _BBA_CACHE


# ---------------------------------------------------------------------------
# Feature Vector Component Generators
# ---------------------------------------------------------------------------
def _type_bits(aa: str) -> list[int]:
    """4-bit amino acid type one-hot encoding."""
    if aa in BASIC: return [1, 0, 0, 0]
    if aa in ACIDIC: return [0, 1, 0, 0]
    if aa in NONPOLAR: return [0, 0, 1, 0]
    if aa in POLAR: return [0, 0, 0, 1]
    return [0, 0, 0, 0]


def _evo_bits(weight: float) -> list[int]:
    """4-bit thermometric encoding of evolutionary conservation weight."""
    if weight < 0.02: return [0, 0, 0, 0]
    if weight < 0.10: return [0, 0, 0, 1]
    if weight < 0.30: return [0, 0, 1, 1]
    if weight < 0.60: return [0, 1, 1, 1]
    return [1, 1, 1, 1]


def _env_bits(count: int) -> list[int]:
    """3-bit encoding for micro-environment property counts."""
    if count == 0: return [0, 0, 0]
    if 1 <= count <= 2: return [0, 0, 1]
    if 3 <= count <= 4: return [0, 1, 1]
    return [1, 1, 1]


def _calc_env_counts(sequence_row: pd.Series, seq_idx: int) -> tuple[int, int, int, int]:
    """Counts basic, acidic, nonpolar, and polar residues in ±3 window (excluding central position)."""
    envB = envA = envNP = envP = 0
    # ±3 window indices (1-based offset: seq_idx - 3 to seq_idx + 3)
    offsets = [-3, -2, -1, 1, 2, 3]
    for offset in offsets:
        pos = seq_idx + offset
        if 0 <= pos < len(sequence_row):
            aa = sequence_row.iloc[pos]
            if aa in BASIC: envB += 1
            elif aa in ACIDIC: envA += 1
            elif aa in NONPOLAR: envNP += 1
            elif aa in POLAR: envP += 1
    return envB, envA, envNP, envP


# ---------------------------------------------------------------------------
# Public Core API
# ---------------------------------------------------------------------------
def featurize_protein(
    kinase_id: str,
    bba_path: Path | str | None = None,
) -> np.ndarray | None:
    """Generates a 1,700-dimensional binary feature vector for a given kinase.

    Args:
        kinase_id: Kinase name (e.g. 'EGFR') or UniProt ID.
        bba_path: Path to Filter_BBA.csv. Defaults to MoBEK/src/generate_feature/protein/Filter_BBA.csv.

    Returns:
        np.ndarray of shape (1700,) with dtype float32, or None if kinase_id is not found.
    """
    bba_path = Path(bba_path) if bba_path else DEFAULT_BBA_CSV
    cache = _load_bba_data(bba_path)
    df = cache["df"]
    
    idx = cache["kinase_to_idx"].get(str(kinase_id))
    if idx is None:
        idx = cache["uniprot_to_idx"].get(str(kinase_id))
    if idx is None:
        return None

    kinase_row = df.iloc[idx]
    n_kinases = cache["n_kinases"]
    
    features = []
    for klifs_pos in BBA_85:
        # Columns in Filter_BBA.csv: Group(0), Kinase(1), Uniprot(2), seq_1(3), ...
        # Therefore, KLIFS position `klifs_pos` (1-based) corresponds to DataFrame column `klifs_pos + 2`
        col_idx = klifs_pos + 2
        target_aa = kinase_row.iloc[col_idx]
        
        # 1. AA Type (4 bits)
        type_vec = _type_bits(target_aa)
        
        # 2. Evolutionary Conservation (4 bits)
        col_data = df.iloc[:, col_idx]
        aa_count = (col_data == target_aa).sum()
        evo_weight = aa_count / float(n_kinases - 1)
        evo_vec = _evo_bits(evo_weight)
        
        # 3. Micro-environment (12 bits)
        envB, envA, envNP, envP = _calc_env_counts(kinase_row, col_idx)
        env_vec = _env_bits(envB) + _env_bits(envA) + _env_bits(envNP) + _env_bits(envP)
        
        features.extend(type_vec + evo_vec + env_vec)

    vec = np.array(features, dtype=np.float32)
    assert vec.shape[0] == 1700, f"Expected 1700 features, got {vec.shape[0]}"
    return vec


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def _get_column_names() -> list[str]:
    cols = ["Group", "Kinase", "Uniprot"]
    for pos_idx, region in enumerate(KLIFS_REGIONS, 1):
        for sub_idx in range(1, 21):
            cols.append(f"#AA_{pos_idx}_{sub_idx} {region}")
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Featurize a list of kinases into a 1,700-dimensional MoBEK binding site matrix."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to line-separated text file of kinase names/UniProt IDs, or a single kinase name."
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path. Defaults to MoBEK/data/<input_stem>_protein_features.[npz|csv]"
    )
    parser.add_argument(
        "--save-as-csv", action="store_true", default=False,
        help="Save output in CSV format instead of default NPZ"
    )
    parser.add_argument("--bba-path", type=Path, default=DEFAULT_BBA_CSV, help="Path to Filter_BBA.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, "r") as f:
            kinases = [line.strip() for line in f if line.strip()]
        stem = input_path.stem
    else:
        kinases = [args.input]
        stem = args.input

    print(f"Processing {len(kinases)} kinase queries...")

    valid_kinases = []
    features_list = []
    for k in kinases:
        vec = featurize_protein(k, bba_path=args.bba_path)
        if vec is not None:
            valid_kinases.append(k)
            features_list.append(vec)
        else:
            print(f"Warning: Kinase identifier not found in alignment database: '{k}'")

    if not features_list:
        print("Error: No valid kinases processed.")
        return

    matrix = np.vstack(features_list)

    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ext = ".csv" if args.save_as_csv else ".npz"
    out_path = args.output if args.output else DEFAULT_DATA_DIR / f"{stem}_protein_features{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_as_csv:
        cache = _load_bba_data(args.bba_path)
        df_bba = cache["df"]
        k_map = cache["kinase_to_idx"]
        u_map = cache["uniprot_to_idx"]

        meta_rows = []
        for k in valid_kinases:
            idx = k_map.get(k, u_map.get(k))
            row = df_bba.iloc[idx]
            meta_rows.append([row["Group"], row["Kinase"], row["Uniprot"]])

        cols = _get_column_names()
        df_out = pd.DataFrame(matrix, columns=cols[3:])
        df_meta = pd.DataFrame(meta_rows, columns=cols[:3])
        df_final = pd.concat([df_meta, df_out], axis=1)
        df_final.to_csv(out_path, index=False)
    else:
        np.savez_compressed(out_path, features=matrix, kinases=np.array(valid_kinases))

    print(f"Successfully processed {len(valid_kinases)} kinases.")
    print(f"Saved protein feature matrix shape {matrix.shape} to: {out_path}")


if __name__ == "__main__":
    main()
