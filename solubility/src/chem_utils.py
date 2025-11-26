# Helper functions for chemistry preprocessing and dataset splits.

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
# to compute Bemis-Murcko scaffolds.
from rdkit.Chem.Scaffolds import MurckoScaffold
# helper for random data splitting.
from sklearn.model_selection import train_test_split


def load_aqsoldb(path: Path | str) -> pd.DataFrame:
    # Load the AqSolDB CSV and validate the expected columns exist.
    df = pd.read_csv(path)  # Read CSV into DataFrame.
    # Guard against missing required columns to fail fast.
    if "SMILES" not in df.columns or "Solubility" not in df.columns:
        raise ValueError("CSV must contain SMILES and Solubility columns.")
    return df  # Return validated DataFrame.


def canonicalize_and_clean(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "Solubility",
) -> pd.DataFrame:
    # Canonicalize SMILES, drop invalid rows, and coerce solubility to numeric.

    def _canonical(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(str(smiles))  # Parse SMILES into RDKit Mol.
        return Chem.MolToSmiles(mol) if mol else None  # Canonicalize or return None.

    out = df.copy()  # Work on a copy to avoid mutating caller data.
    out[smiles_col] = out[smiles_col].astype(str).map(_canonical)  # Canonicalize SMILES strings.
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")  # Force numeric target, set bad values to NaN.
    out = out.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)  # Drop rows with invalid SMILES/targets.
    return out  # Cleaned DataFrame.


def _scaffold(smiles: str) -> str:
    # Return the Bemis-Murcko scaffold SMILES or empty string if it fails.
    mol = Chem.MolFromSmiles(smiles)  # Parse SMILES into Mol.
    if mol is None:  # Handle invalid molecules.
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)  # Extract scaffold core.
    return Chem.MolToSmiles(core) if core else ""  # Return canonical scaffold or empty string.


def scaffold_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Group by scaffold, shuffle buckets, and fill train/val/test splits.
    if not np.isclose(sum(frac), 1.0):  # Ensure fractions sum to 1.
        raise ValueError("Split fractions must sum to 1.")

    buckets: dict[str, list[int]] = {}  # Map scaffold SMILES -> row indices.
    for scaffold_smiles, subset in df.groupby(df[smiles_col].map(_scaffold)):
        # Track the original row indices belonging to each scaffold bucket.
        buckets[scaffold_smiles] = subset.index.to_list()

    keys = list(buckets.keys())  # All scaffold identifiers.
    rng = np.random.default_rng(seed)  # Reproducible RNG.
    rng.shuffle(keys)  # Shuffle scaffold buckets.

    n = len(df)  # Total rows.
    n_train = int(frac[0] * n)  # Target train size.
    n_val = int(frac[1] * n)  # Target val size.

    train_idx: list[int] = []  # Accumulator for train indices.
    val_idx: list[int] = []  # Accumulator for val indices.
    test_idx: list[int] = []  # Accumulator for test indices.

    for key in keys:  # Fill splits scaffold by scaffold.
        inds = buckets[key]  # All rows for this scaffold.
        if len(train_idx) < n_train:
            # Keep filling train first to preserve scaffold grouping.
            train_idx += inds
        elif len(val_idx) < n_val:
            val_idx += inds
        else:
            test_idx += inds

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)  # Return index arrays.


def random_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
    target_col: str = "Solubility",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Simple random split fallback when scaffolds are not desired.
    if not np.isclose(sum(frac), 1.0):  # Validate split fractions.
        raise ValueError("Split fractions must sum to 1.")

    train_size, val_size = frac[0], frac[1]  # Desired proportions.
    tmp_size = 1.0 - train_size  # Combined val+test proportion.
    train_df, tmp_df = train_test_split(df, test_size=tmp_size, random_state=seed, shuffle=True)  # First split off train.
    rel_val = val_size / (val_size + frac[2])  # Relative val share of remaining data.
    val_df, test_df = train_test_split(tmp_df, test_size=1 - rel_val, random_state=seed, shuffle=True)  # Split remaining.
    return train_df.index.values, val_df.index.values, test_df.index.values  # Return index arrays.


def scaffold_or_random_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use scaffold split, falling back to random split if something goes wrong.
    try:
        return scaffold_split(df, frac=frac, seed=seed, smiles_col=smiles_col)  # Prefer scaffold grouping.
    except Exception:
        return random_split(df, frac=frac, seed=seed, smiles_col=smiles_col)  # Safe fallback.


def save_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path | str,
) -> None:
    # Persist splits to disk.
    out = Path(out_dir)  # Normalize output directory.
    out.mkdir(parents=True, exist_ok=True)  # Ensure directory exists.
    np.save(out / "train_idx.npy", train_idx)  # Save train indices.
    np.save(out / "val_idx.npy", val_idx)  # Save validation indices.
    np.save(out / "test_idx.npy", test_idx)  # Save test indices.
