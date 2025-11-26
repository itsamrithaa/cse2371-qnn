# Tokenizer, dataset, model, and evaluation utilities for SMILES CNN.
from collections import Counter # counts tokens for vocabulary building.
import re # Regular expressions for SMILES tokenization.
from typing import Dict, List, Sequence, Tuple # Type hints.

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset

# Handles Cl/Br, bracket atoms, %nn, and single characters.
TOKENIZER_RE = re.compile(r"(\\%\\d{2}|\\[[^\\]]+\\]|Br|Cl|.)")


def tokenize_smiles(smi: str) -> List[str]:
    # Regex-based tokenizer that keeps common multi-character atoms together.
    return TOKENIZER_RE.findall(smi)  # Return list of tokens.


def build_vocab(smiles_list: Sequence[str], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    # Build <PAD>/<UNK>-aware vocab from a list of SMILES strings.
    cnt = Counter()  # Token frequency counter.
    for smi in smiles_list:
        cnt.update(tokenize_smiles(smi))  # Add tokens from each SMILES.
    itos: List[str] = ["<PAD>", "<UNK>"]  # Index-to-string list with special tokens.
    for tok, freq in cnt.most_common():  # Iterate tokens by frequency.
        if freq >= min_freq and tok not in itos:
            itos.append(tok)  # Keep tokens that meet frequency threshold.
    stoi = {tok: idx for idx, tok in enumerate(itos)}  # String-to-index mapping.
    return stoi, itos  # Return both mappings.


def encode_smiles(smi: str, stoi: Dict[str, int], max_len: int) -> np.ndarray:
    # Convert a SMILES string into a fixed-length index array.
    ids = [stoi.get(tok, stoi["<UNK>"]) for tok in tokenize_smiles(smi)[:max_len]]  # Map tokens to indices.
    if len(ids) < max_len:
        ids += [stoi["<PAD>"]] * (max_len - len(ids))  # Right-pad to fixed length.
    return np.array(ids, dtype=np.int64)  # Return numpy array of indices.


class SmilesDataset(Dataset):
    # Minimal torch Dataset for SMILES regression.

    def __init__(self, smiles: Sequence[str], y: Sequence[float] | None, stoi: Dict[str, int], max_len: int):
        self.X = [encode_smiles(s, stoi, max_len) for s in smiles]  # Pre-encode SMILES to indices.
        self.y = None if y is None else np.array(y, dtype=np.float32)  # Store targets if provided.

    def __len__(self) -> int:
        return len(self.X)  # Number of samples.

    def __getitem__(self, idx: int):
        ids = torch.from_numpy(self.X[idx])  # Token indices tensor.
        if self.y is None:
            return ids  # Inference-only mode.
        return ids, torch.tensor(self.y[idx])  # Return features and target.


class SmilesCNN(nn.Module):
    # Three-branch 1D CNN followed by a two-layer MLP head.

    def __init__(self, vocab_size: int, emb_dim: int = 128, max_len: int = 256, dropout: float = 0.2):
        super().__init__()  # Initialize nn.Module.
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # Token embedding layer with pad index.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),  # Narrow convolution.
                nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),  # Medium convolution.
                nn.Conv1d(emb_dim, 128, kernel_size=7, padding=3),  # Wide convolution.
            ]
        )
        self.act = nn.ReLU()  # Nonlinearity shared across conv branches.
        self.dropout = nn.Dropout(dropout)  # Regularization.
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 256),  # Combine pooled conv outputs.
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # Final regression head.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)  # (B, L, E) embedded tokens.
        emb = emb.transpose(1, 2)  # (B, E, L) for Conv1d channel-first format.
        # Apply each convolution, ReLU, and global max pool over sequence length.
        feats = [torch.amax(self.act(conv(emb)), dim=-1) for conv in self.convs]
        h = torch.cat(feats, dim=1)  # Concatenate pooled features from all kernels.
        h = self.dropout(h)  # Dropout before head.
        out = self.head(h).squeeze(1)  # Final scalar prediction per sample.
        return out  # Shape: (B,)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Tuple[float, float, float]:
    # Compute RMSE/MAE/R2 on a DataLoader.
    model.eval()  # Switch to eval mode.
    ys: List[np.ndarray] = []  # Ground-truth accumulator.
    ps: List[np.ndarray] = []  # Prediction accumulator.
    for xb, yb in loader:  # Iterate over batches.
        xb = xb.to(device)  # Move inputs to device.
        pred = model(xb).cpu().numpy()  # Forward pass and move to CPU numpy.
        ys.append(yb.numpy())  # Store targets.
        ps.append(pred)  # Store predictions.
    y = np.concatenate(ys)  # All ground truth.
    p = np.concatenate(ps)  # All predictions.
    rmse = float(np.sqrt(((y - p) ** 2).mean()))  # Root mean squared error.
    mae = float(np.mean(np.abs(y - p)))  # Mean absolute error.
    r2 = float(r2_score(y, p))  # Coefficient of determination.
    return rmse, mae, r2  # Metric tuple.
