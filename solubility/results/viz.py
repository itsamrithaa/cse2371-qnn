"""
Compare CNN, hybrid, and (non-hybrid) QNN test predictions and visualize four
functional-group series:

1) Relative baseline: hexane, methanol, phenol, decanol, hydroquinone
2) Stress test: ethyl acetate, butanol, acetophenone, glycerol, isobutanol
3) Alcohol chain length: methanol -> decanol
4) Aromatic substitutions: benzene/toluene/halogens/acetophenone vs polar ring substituents
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RESULT_FILES = {
    "cnn": "cnn_test_predictions.csv",
    "hybrid": "hybrid_test_predictions.csv",
    "qnn": "qnn_test_predictions.csv",
}

COLORS = {"cnn": "#1f77b4", "hybrid": "#2ca02c", "qnn": "#d62728"}

SERIES = {
    "relative_baseline": {
        "title": "Relative solubility reference",
        "desc": "Soluble (methanol/phenol/hydroquinone) vs less soluble (hexane/decanol).",
        "compounds": [
            {"name": "Hexane", "smiles": "CCCCCC", "expectation": "low"},
            {"name": "Methanol", "smiles": "CO", "expectation": "high"},
            {"name": "Phenol", "smiles": "c1cc(ccc1)O", "expectation": "high"},
            {"name": "Decanol", "smiles": "CCCCCCCCCCO", "expectation": "low"},
            {"name": "Hydroquinone", "smiles": "c1cc(ccc1O)O", "expectation": "high"},
        ],
    },
    "stress_test": {
        "title": "Stress test",
        "desc": "Small structural tweaks with large solubility swings.",
        "compounds": [
            {"name": "Ethyl acetate", "smiles": "CCOC(=O)C"},
            {"name": "Butanol", "smiles": "CCCCO"},
            {"name": "Acetophenone", "smiles": "CC(=O)c1ccccc1"},
            {"name": "Glycerol", "smiles": "C(C(CO)O)O"},
            {"name": "Isobutanol", "smiles": "CC(C)CO"},
        ],
    },
    "alcohol_chain": {
        "title": "Alcohol chain length",
        "desc": "Chain length vs solubility: short-chain alcohols more soluble than long-chain.",
        "compounds": [
            {"name": "Methanol", "smiles": "CO"},
            {"name": "Ethanol", "smiles": "CCO"},
            {"name": "Propanol", "smiles": "CCCO"},
            {"name": "Butanol", "smiles": "CCCCO"},
            {"name": "Pentanol", "smiles": "CCCCCO"},
            {"name": "Hexanol", "smiles": "CCCCCCO"},
            {"name": "Heptanol", "smiles": "CCCCCCCO"},
            {"name": "Octanol", "smiles": "CCCCCCCCO"},
            {"name": "Nonanol", "smiles": "CCCCCCCCCO"},
            {"name": "Decanol", "smiles": "CCCCCCCCCCO"},
        ],
    },
    "aromatic_subs": {
        "title": "Aromatic functional groups",
        "desc": "Aromatic rings with polar vs non-polar substituents (non-OH groups included).",
        "compounds": [
            {"name": "Phenol", "smiles": "c1ccccc1O", "expectation": "higher"},
            {"name": "Hydroquinone", "smiles": "c1cc(ccc1O)O", "expectation": "higher"},
            {"name": "Aniline", "smiles": "c1ccccc1N", "expectation": "higher"},
            {"name": "Nitrobenzene", "smiles": "c1ccccc1[N+](=O)[O-]", "expectation": "higher"},
            {"name": "Benzaldehyde", "smiles": "O=CC1=CC=CC=C1", "expectation": "higher"},
            {"name": "Benzene", "smiles": "c1ccccc1", "expectation": "lower"},
            {"name": "Toluene", "smiles": "Cc1ccccc1", "expectation": "lower"},
            {"name": "Chlorobenzene", "smiles": "c1ccccc1Cl", "expectation": "lower"},
            {"name": "Bromobenzene", "smiles": "c1ccccc1Br", "expectation": "lower"},
            {"name": "Acetophenone", "smiles": "CC(=O)c1ccccc1", "expectation": "lower"},
        ],
    },
}


def load_predictions(results_dir: Path):
    data = {}
    for name, fname in RESULT_FILES.items():
        path = results_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing results file for {name}: {path}")
        df = pd.read_csv(path)
        if not {"y_true", "y_pred"}.issubset(df.columns):
            raise ValueError(f"{path} must contain 'y_true' and 'y_pred' columns")
        data[name] = df
    return data


def attach_smiles(test_csv: Path, preds: dict):
    test_df = pd.read_csv(test_csv)
    if "SMILES" not in test_df.columns:
        raise ValueError(f"{test_csv} must include a SMILES column to map compounds")
    results = {}
    for name, df in preds.items():
        if len(df) != len(test_df):
            raise ValueError(f"Row count mismatch between {test_csv} and {RESULT_FILES[name]}")
        df = df.copy()
        df["SMILES"] = test_df["SMILES"].values
        results[name] = df
    return results


def compute_metrics(df: pd.DataFrame):
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_comparison(dfs: dict, metrics_df: pd.DataFrame, out_path: Path):
    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    for name, df in dfs.items():
        ax_scatter.scatter(
            df["y_true"],
            df["y_pred"],
            s=12,
            alpha=0.6,
            label=name.upper(),
            color=COLORS.get(name, None),
        )
    all_true = pd.concat([df["y_true"] for df in dfs.values()])
    lims = [all_true.min(), all_true.max()]
    ax_scatter.plot(lims, lims, "k--", linewidth=1)
    ax_scatter.set_xlabel("True solubility")
    ax_scatter.set_ylabel("Predicted solubility")
    ax_scatter.set_title("Test parity plot")
    ax_scatter.legend()

    metrics_to_plot = ["rmse", "mae", "r2"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    for i, (name, row) in enumerate(metrics_df.iterrows()):
        ax_bar.bar(
            x + i * width,
            [row[m] for m in metrics_to_plot],
            width,
            label=name.upper(),
            color=COLORS.get(name, None),
        )
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax_bar.set_title("Model metrics (lower is better for RMSE/MAE)")
    ax_bar.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved comparison figure to {out_path}")


def build_series_frame(series_key: str, series_spec: dict, preds_with_smiles: dict):
    records = []
    missing = []
    for comp in series_spec["compounds"]:
        row = {
            "Compound": comp["name"],
            "SMILES": comp["smiles"],
            "Expectation": comp.get("expectation", ""),
        }
        for model_name, df in preds_with_smiles.items():
            match = df.loc[df["SMILES"] == comp["smiles"], "y_pred"]
            row[model_name] = float(match.iloc[0]) if not match.empty else np.nan
            if match.empty:
                missing.append((comp["name"], comp["smiles"], model_name))
        records.append(row)
    if missing:
        print(f"[{series_key}] Missing predictions for: {missing}")
    return pd.DataFrame(records)


def plot_series(series_key: str, series_spec: dict, series_df: pd.DataFrame, out_dir: Path):
    melted = series_df.melt(
        id_vars=["Compound", "Expectation"],
        value_vars=list(RESULT_FILES.keys()),
        var_name="model",
        value_name="predicted",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(series_df))
    width = 0.2
    for i, model in enumerate(RESULT_FILES.keys()):
        vals = series_df[model]
        ax.bar(x + i * width, vals, width, label=model.upper(), color=COLORS.get(model, None))
    ax.set_xticks(x + width)
    ax.set_xticklabels(series_df["Compound"], rotation=45, ha="right")
    ax.set_ylabel("Predicted solubility")
    ax.set_title(series_spec["title"])
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / f"series_{series_key}.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved series plot to {out_path}")

    return melted


def main():
    results_dir = Path(__file__).resolve().parent
    project_root = results_dir.parent

    preds = load_predictions(results_dir)
    preds_with_smiles = attach_smiles(project_root / "data/test.csv", preds)

    metrics = []
    for name, df in preds.items():
        m = compute_metrics(df)
        m["model"] = name
        metrics.append(m)
    metrics_df = pd.DataFrame(metrics).set_index("model").loc[list(RESULT_FILES.keys())]

    print("Metrics on test split:")
    print(metrics_df)

    plot_comparison(preds, metrics_df, results_dir / "model_comparison.png")

    for key, spec in SERIES.items():
        series_df = build_series_frame(key, spec, preds_with_smiles)
        if series_df.empty:
            print(f"[{key}] No compounds found in test set; skipping plot.")
            continue
        plot_series(key, spec, series_df, results_dir)


if __name__ == "__main__":
    main()
