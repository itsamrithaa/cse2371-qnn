"""Tokenizer, dataset, model, and evaluation utilities for SMILES CNN."""
from collections import Counter
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset

# Handles Cl/Br, bracket atoms, %nn, and single characters.
TOKENIZER_RE = re.compile(r"(\\%\\d{2}|\\[[^\\]]+\\]|Br|Cl|.)")


def tokenize_smiles(smi: str) -> List[str]:
    return TOKENIZER_RE.findall(smi)


def build_vocab(smiles_list: Sequence[str], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    """Build <PAD>/<UNK>-aware vocab from a list of SMILES strings."""
    cnt = Counter()
    for smi in smiles_list:
        cnt.update(tokenize_smiles(smi))
    itos: List[str] = ["<PAD>", "<UNK>"]
    for tok, freq in cnt.most_common():
        if freq >= min_freq and tok not in itos:
            itos.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return stoi, itos


def encode_smiles(smi: str, stoi: Dict[str, int], max_len: int) -> np.ndarray:
    """Convert a SMILES string into a fixed-length index array."""
    ids = [stoi.get(tok, stoi["<UNK>"]) for tok in tokenize_smiles(smi)[:max_len]]
    if len(ids) < max_len:
        ids += [stoi["<PAD>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


class SmilesDataset(Dataset):
    """Minimal torch Dataset for SMILES regression."""

    def __init__(self, smiles: Sequence[str], y: Sequence[float] | None, stoi: Dict[str, int], max_len: int):
        self.X = [encode_smiles(s, stoi, max_len) for s in smiles]
        self.y = None if y is None else np.array(y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        ids = torch.from_numpy(self.X[idx])
        if self.y is None:
            return ids
        return ids, torch.tensor(self.y[idx])


class SmilesCNN(nn.Module):
    """Three-branch 1D CNN followed by a two-layer MLP head."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, max_len: int = 256, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
                nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
                nn.Conv1d(emb_dim, 128, kernel_size=7, padding=3),
            ]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(128 * 3, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)  # (B, L, E)
        emb = emb.transpose(1, 2)  # (B, E, L)
        feats = [torch.amax(self.act(conv(emb)), dim=-1) for conv in self.convs]  # global max pool per branch
        h = torch.cat(feats, dim=1)
        h = self.dropout(h)
        out = self.head(h).squeeze(1)
        return out


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Tuple[float, float, float]:
    """Compute RMSE/MAE/R2 on a DataLoader."""
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(r2_score(y, p))
    return rmse, mae, r2
