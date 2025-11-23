"""Helper functions for chemistry preprocessing and dataset splits."""
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def load_aqsoldb(path: Path | str) -> pd.DataFrame:
    """Load the AqSolDB CSV and validate the expected columns exist."""
    df = pd.read_csv(path)
    if "SMILES" not in df.columns or "Solubility" not in df.columns:
        raise ValueError("CSV must contain SMILES and Solubility columns.")
    return df


def canonicalize_and_clean(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "Solubility",
) -> pd.DataFrame:
    """Canonicalize SMILES, drop invalid rows, and coerce solubility to numeric."""

    def _canonical(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol else None

    out = df.copy()
    out[smiles_col] = out[smiles_col].astype(str).map(_canonical)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
    return out


def _scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES or empty string if it fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core else ""


def scaffold_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group by scaffold, shuffle buckets, and fill train/val/test splits."""
    if not np.isclose(sum(frac), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    buckets: dict[str, list[int]] = {}
    for scaffold_smiles, subset in df.groupby(df[smiles_col].map(_scaffold)):
        buckets[scaffold_smiles] = subset.index.to_list()

    keys = list(buckets.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(df)
    n_train = int(frac[0] * n)
    n_val = int(frac[1] * n)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for key in keys:
        inds = buckets[key]
        if len(train_idx) < n_train:
            train_idx += inds
        elif len(val_idx) < n_val:
            val_idx += inds
        else:
            test_idx += inds

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def random_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
    target_col: str = "Solubility",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple random split fallback when scaffolds are not desired."""
    if not np.isclose(sum(frac), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    train_size, val_size = frac[0], frac[1]
    tmp_size = 1.0 - train_size
    train_df, tmp_df = train_test_split(df, test_size=tmp_size, random_state=seed, shuffle=True)
    rel_val = val_size / (val_size + frac[2])
    val_df, test_df = train_test_split(tmp_df, test_size=1 - rel_val, random_state=seed, shuffle=True)
    return train_df.index.values, val_df.index.values, test_df.index.values


def scaffold_or_random_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    smiles_col: str = "SMILES",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use scaffold split, falling back to random split if something goes wrong."""
    try:
        return scaffold_split(df, frac=frac, seed=seed, smiles_col=smiles_col)
    except Exception:
        return random_split(df, frac=frac, seed=seed, smiles_col=smiles_col)


def save_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path | str,
) -> None:
    """Persist splits to disk."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "val_idx.npy", val_idx)
    np.save(out / "test_idx.npy", test_idx)
