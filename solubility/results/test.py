"""Render metrics table and save as a PNG (table.png)."""

from pathlib import Path

import pandas as pd


DATA = {
    "model": ["cnn", "hybrid", "qnn"],
    "rmse": [1.539021, 1.503468, 2.189193],
    "mae": [1.163477, 1.121527, 1.685021],
    "r2": [0.656315, 0.672010, 0.304592],
}

COLORS = {
    "header": "#d8e9ff",
    "row_even": "#fffbe6",
    "row_odd": "#ffffff",
    "text": "#000000",
    "border": "#000000",
}


def metrics_table_html() -> str:
    """Return an HTML-styled metrics table (light-blue header, alternating rows)."""
    df = pd.DataFrame(DATA).set_index("model")

    def color_rows(row):
        return [
            f"background-color: {COLORS['row_even' if i % 2 == 0 else 'row_odd']}; color: {COLORS['text']};"
            for i in range(len(row))
        ]

    styled = (
        df.style.set_table_styles(
            [{"selector": "th", "props": f"background:{COLORS['header']}; color:{COLORS['text']}; border:1px solid {COLORS['border']};"}]
        )
        .apply(color_rows, axis=1)
        .format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"})
        .set_properties(border=f"1px solid {COLORS['border']}", **{"padding": "6px 10px"})
    )
    return styled.to_html()


def metrics_table_png(output_path: Path = Path("table.png")) -> Path:
    """Render the metrics table to a PNG using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("matplotlib is required to save the PNG. Install via `pip install matplotlib`.") from exc

    df = pd.DataFrame(DATA)
    fig, ax = plt.subplots(figsize=(5, 1.8))
    ax.axis("off")

    # Prepare cell text with formatted numbers
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([row["model"], f"{row['rmse']:.3f}", f"{row['mae']:.3f}", f"{row['r2']:.3f}"])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Model", "RMSE", "MAE", "R²"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Color header
    for key, cell in table.get_celld().items():
        row, col = key
        # Header row
        if row == 0:
            cell.set_facecolor(COLORS["header"])
        else:
            # Alternate row colors
            cell.set_facecolor(COLORS["row_even"] if (row % 2 == 1) else COLORS["row_odd"])
        cell.set_edgecolor(COLORS["border"])
        cell.get_text().set_color(COLORS["text"])

    fig.tight_layout()
    output_path = output_path.resolve()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    png_path = metrics_table_png()
    print(f"Saved metrics table to {png_path}")
