"""
Unified SMILES Featurizer for MoBEK using original binary tools.

Generates 1,048-dimensional feature vectors per SMILES string in the following order:
  1. ECFP Morgan fingerprints (512 dim) via RDKit
  2. Ring-in-drug moieties (315 dim: 168 Pubchem + 147 Ring) via `matchmol` binary
  3. Functional groups (204 dim) via `checkmol` binary
  4. AC atom counts (10 dim) via `mod_ac` binary
  5. Stereochemistry (7 dim) via Python regex parsing

"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ---------------------------------------------------------------------------
# Path Resolutions (Absolute Path Agnostic)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJ_DIR = SRC_DIR.parent

DEFAULT_BIN_DIR = SCRIPT_DIR / "vendors" / "bin"
DEFAULT_MOIETY_DIR = SCRIPT_DIR / "vendors" / "moieties"
DEFAULT_DATA_DIR = PROJ_DIR / "data"

# ---------------------------------------------------------------------------
# Hardcoded Matchmol Per-Pattern Search Modes
# Maps pattern indices (0..167 for Pubchem, 168..314 for Ring) to matchmode.
# Unlisted indices fall back to "default", while indices mapped to "r" use `-r` (SSR ring search).
# ---------------------------------------------------------------------------
MATCHMOL_MODES: dict[int, str] = {
    # Populate indices requiring the '-r' SSR ring search mode here if applicable.
    # e.g., 0: "r", 12: "r"
}

# Global pattern caches
_PC_FILES: list[Path] | None = None
_RG_FILES: list[Path] | None = None


def _get_binaries(bin_dir: Path) -> tuple[Path, Path, Path]:
    cm_bin = bin_dir / "checkmol"
    mm_bin = bin_dir / "matchmol"
    ac_bin = bin_dir / "mod_ac"
    return cm_bin, mm_bin, ac_bin


def _get_moiety_files(moiety_dir: Path) -> tuple[list[Path], list[Path]]:
    global _PC_FILES, _RG_FILES
    if _PC_FILES is None or _RG_FILES is None:
        _PC_FILES = sorted((moiety_dir / "pubchem").glob("*.mol"))
        _RG_FILES = sorted((moiety_dir / "rings_in_drugs").glob("*.mol"))
    return _PC_FILES, _RG_FILES


def _run_cmd(args: list[str], timeout: int = 60) -> bytes:
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return r.stdout
    except Exception:
        return b""


# ---------------------------------------------------------------------------
# Feature Calculators
# ---------------------------------------------------------------------------
def checkmol_block(mol_path: Path, cm_bin: Path) -> np.ndarray:
    out = np.zeros(204, dtype=np.int64)
    stdout = _run_cmd([str(cm_bin), "-p", str(mol_path)])
    for line in stdout.decode(errors="replace").splitlines():
        parts = line.split(":")
        fea = parts[0].split("#")
        if len(fea) == 2:
            try:
                if int(parts[1]) >= 1:
                    out[int(fea[1]) - 1] = 1
            except (ValueError, IndexError):
                pass
    return out


def matchmol_pattern_block(
    mol_path: Path, files: list[Path], off: int, mm_bin: Path, modes: dict[int, str] | None = None
) -> np.ndarray:
    if modes is None:
        modes = MATCHMOL_MODES
    out = np.zeros(len(files), dtype=np.int64)
    for j, f in enumerate(files):
        flag = "-r" if modes.get(off + j, "default") == "r" else None
        cmd = [str(mm_bin)]
        if flag:
            cmd.append(flag)
        cmd.extend([str(f), str(mol_path)])
        out[j] = 1 if b"T" in _run_cmd(cmd) else 0
    return out


def ac_block(mol_path: Path, ac_bin: Path) -> np.ndarray:
    out = np.zeros(10, dtype=np.int64)
    stdout = _run_cmd([str(ac_bin), str(mol_path)])
    lines = stdout.decode(errors="replace").splitlines()
    try:
        if len(lines) >= 2:
            nums = lines[1].split("\t")
            if len(nums) == 11:
                out[:] = [int(x) for x in nums[1:]]
    except Exception:
        pass
    return out


def ecfp_vector(mol: Chem.Mol) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512)
    arr = np.zeros(512, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def stereo_vector(smiles: str) -> np.ndarray:
    count_at = len(re.findall(r'(?<!l)@(?![l])', smiles))
    count_sl = len(re.findall(r'(?<!l)\/(?![l])', smiles))
    count_bs = len(re.findall(r'(?<!l)\\(?![l])', smiles))

    continue_m = continue_mm = first_m = first_mm = 0
    max_m, max_mm = [], []
    n = len(smiles)
    for i in range(n):
        if smiles[i] == '@':
            nxt = smiles[i + 1] if i + 1 < n else None
            prv = smiles[i - 1] if i - 1 >= 0 else None
            if nxt != '@' and prv != '@':
                if first_mm == 0:
                    first_m = 1
                continue_m += 1
                if continue_mm != 0:
                    max_mm.append(0 if continue_mm == 1 else continue_mm)
                    continue_mm = 0
            elif nxt == '@':
                if first_m == 0:
                    first_mm = 1
                continue_mm += 1
                if continue_m != 0:
                    max_m.append(0 if continue_m == 1 else continue_m)
                    continue_m = 0
    max_m.append(0 if continue_m == 1 else continue_m)
    max_mm.append(0 if continue_mm == 1 else continue_mm)

    return np.array(
        [count_at, max(max_m), max(max_mm), first_m, first_mm, count_sl, count_bs],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Core Public API Function
# ---------------------------------------------------------------------------
def featurize_smiles(
    smiles: str,
    bin_dir: Path | str | None = None,
    moiety_dir: Path | str | None = None,
    workdir: Path | str | None = None,
    n_threads: int = 4,
    matchmol_modes: dict[int, str] | None = None,
) -> np.ndarray | None:
    """
    Takes a single SMILES string and returns a 1,048-dimensional float32 feature vector.

    Feature Ordering (1048 total):
      1. ECFP (512)
      2. Ring-in-drug / Moieties (315: 168 Pubchem + 147 Ring)
      3. Functional groups / Checkmol (204)
      4. AC Atom Counts (10)
      5. Stereochemistry (7)

    Returns None if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bin_dir = Path(bin_dir) if bin_dir else DEFAULT_BIN_DIR
    moiety_dir = Path(moiety_dir) if moiety_dir else DEFAULT_MOIETY_DIR

    cm_bin, mm_bin, ac_bin = _get_binaries(bin_dir)
    pc_files, rg_files = _get_moiety_files(moiety_dir)
    modes = matchmol_modes if matchmol_modes is not None else MATCHMOL_MODES

    mb = Chem.MolToMolBlock(mol)

    parent_td = Path(workdir) if workdir else Path(tempfile.gettempdir())
    td = Path(tempfile.mkdtemp(prefix="mobek_feat_", dir=str(parent_td)))

    try:
        mp = td / "mol.mol"
        mp.write_text(mb)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            f_cm = ex.submit(checkmol_block, mp, cm_bin)
            f_pc = ex.submit(matchmol_pattern_block, mp, pc_files, 0, mm_bin, modes)
            f_rg = ex.submit(matchmol_pattern_block, mp, rg_files, len(pc_files), mm_bin, modes)
            f_ac = ex.submit(ac_block, mp, ac_bin)

            cm = f_cm.result()
            pc = f_pc.result()
            rg = f_rg.result()
            ac = f_ac.result()

        # Concatenation order: ECFP (512) + Pubchem (168) + Ring (147) + Checkmol (204) + AC (10) + Stereo (7) = 1048
        vec = np.concatenate([
            ecfp_vector(mol),
            pc,
            rg,
            cm,
            ac,
            stereo_vector(smiles),
        ]).astype(np.float32)

        assert vec.shape[0] == 1048, f"Expected 1048 features, got {vec.shape[0]}"
        return vec
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def _get_column_names(n_pc: int = 168, n_rg: int = 147) -> list[str]:
    # 1. ECFP (512)
    names = [f"#ECFP_{i}" for i in range(1, 513)]
    # 2. Ring-in-drug (168 Pubchem + 147 Ring = 315)
    names += [f"#Pubchem_{i}" for i in range(1, n_pc + 1)]
    names += [f"#Ring_{i}" for i in range(1, n_rg + 1)]
    # 3. Functional Groups / Checkmol (204)
    names += [f"#Checkmol_{i}" for i in range(1, 205)]
    # 4. AC Atom Counts (10)
    names += ["C.ring", "C.other", "N.ring", "N.other", "O.ring", "O.other", "P", "S", "X", "#.of.Ring"]
    # 5. Stereochemistry (7)
    names += ["count_@", "arr_con_m", "arr_con_mm", "arr_fir_m", "arr_fir_mm", "count_/", "count_\\"]
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Featurize a text file of SMILES into a MoBEK feature matrix."
    )
    parser.add_argument("input_file", type=Path, help="Path to text file containing line-separated SMILES")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file path. Defaults to MoBEK/data/<input_stem>_features.[npz|csv]"
    )
    parser.add_argument(
        "--save-as-csv", action="store_true", default=False,
        help="Save output as CSV format instead of NPZ (default: NPZ)"
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR, help="Path to binaries folder")
    parser.add_argument("--moiety-dir", type=Path, default=DEFAULT_MOIETY_DIR, help="Path to moieties folder")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel worker threads for batch processing")

    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} SMILES strings from {args.input_file}")

    valid_smiles = []
    features_list = []

    for i, smi in enumerate(smiles_list, 1):
        vec = featurize_smiles(
            smi,
            bin_dir=args.bin_dir,
            moiety_dir=args.moiety_dir,
            n_threads=args.n_jobs,
        )
        if vec is not None:
            valid_smiles.append(smi)
            features_list.append(vec)
        else:
            print(f"Warning: Failed to parse SMILES at line {i}: {smi}")

    if not features_list:
        print("Error: No valid SMILES processed.")
        return

    features_matrix = np.vstack(features_list)

    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ext = ".csv" if args.save_as_csv else ".npz"
    if args.output is None:
        out_path = DEFAULT_DATA_DIR / f"{args.input_file.stem}_features{ext}"
    else:
        out_path = args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_as_csv:
        cols = _get_column_names()
        df = pd.DataFrame(features_matrix, columns=cols)
        df.insert(0, "smiles", valid_smiles)
        df.to_csv(out_path, index=False)
    else:
        np.savez_compressed(out_path, features=features_matrix, smiles=np.array(valid_smiles))

    print(f"Successfully featurized {len(valid_smiles)} molecules.")
    print(f"Saved feature matrix shape {features_matrix.shape} to: {out_path}")


if __name__ == "__main__":
    main()
