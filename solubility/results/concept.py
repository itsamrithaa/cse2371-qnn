from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PALETTE = {
    "io": "#f4f4f4",
    "embed": "#d8e9ff",
    "conv": "#fde5cc",
    "pool": "#fff2cc",
    "concat": "#e5f7e5",
    "dense": "#d8e9ff",
    "quantum": "#ffe0e0",
    "measure": "#e5f7e5",
}


def add_box(ax, xy, text, color, width=1.8, height=0.7, fontsize=9):
    """Draw a rounded box with centered text and return its right-middle point."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.12",
        fc=color,
        ec="#444444",
        linewidth=1,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x + width, y + height / 2)


def add_arrow(ax, start, end):
    """Draw a simple arrow between two points."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#444444", linewidth=1),
    )


def plot_cnn(ax):
    """Plot the CNN flow: embed -> 3 conv branches -> pool -> concat -> dense."""
    ax.set_title("CNN pipeline", fontsize=12, pad=10)
    y0 = 0
    x0 = -3

    # Input → token → embed
    p1 = add_box(ax, (x0, y0 - 0.35), "SMILES", PALETTE["io"])
    add_arrow(ax, p1, (p1[0] + 0.5, p1[1]))
    p2 = add_box(ax, (p1[0] + 0.7, y0 - 0.35), "Tokenize", PALETTE["io"])
    add_arrow(ax, p2, (p2[0] + 0.5, p2[1]))
    p_embed = add_box(ax, (p2[0] + 0.7, y0 - 0.35), "Embed\n128d", PALETTE["embed"])
    add_arrow(ax, p_embed, (p_embed[0] + 0.6, p_embed[1]))

    # Conv branches (vertical stack)
    branch_outputs = []
    conv_x = p_embed[0] + 1.2
    for idx, (k, offset) in enumerate(zip([3, 5, 7], [-0.9, 0.0, 0.9])):
        conv_end = add_box(ax, (conv_x, y0 + offset - 0.35), f"Conv k={k}", PALETTE["conv"])
        add_arrow(ax, (p_embed[0] + 0.6, p_embed[1]), (conv_x, y0 + offset + 0.0))
        add_arrow(ax, conv_end, (conv_end[0] + 0.3, conv_end[1]))
        pool_end = add_box(ax, (conv_end[0] + 0.5, y0 + offset - 0.35), "Global\nMaxPool", PALETTE["pool"])
        branch_outputs.append((pool_end[0], pool_end[1]))

    # Concat
    concat_x = max(px for px, _ in branch_outputs) + 1.0
    concat_y = y0 - 0.35
    concat_end = add_box(ax, (concat_x, concat_y), "Concat\n384d", PALETTE["concat"])
    for px, py in branch_outputs:
        add_arrow(ax, (px, py), (concat_x, py))
    add_arrow(ax, concat_end, (concat_end[0] + 0.5, concat_end[1]))

    # Dense head
    dense1 = add_box(
        ax,
        (concat_end[0] + 0.7, concat_y),
        "Dense 256\nReLU + Dropout",
        PALETTE["dense"],
        width=2.4,
    )
    add_arrow(ax, dense1, (dense1[0] + 0.5, dense1[1]))
    add_box(ax, (dense1[0] + 0.7, concat_y), "Dense 1\nOutput", PALETTE["dense"], width=1.8)

    ax.set_xlim(x0 - 0.5, dense1[0] + 3.5)
    ax.set_ylim(y0 - 1.8, y0 + 1.8)
    ax.axis("off")


def plot_qnn(ax):
    """Plot the conceptual QNN flow."""
    ax.set_title("QNN pipeline (concept)", fontsize=12, pad=10)
    y = 0
    x = -3

    steps = [
        ("SMILES", PALETTE["io"], 1.8),
        ("Tokenize", PALETTE["io"], 1.8),
        ("Embed\n(vector)", PALETTE["embed"], 1.9),
        ("Encode →\nQubit Rotations", PALETTE["embed"], 2.2),
        ("Rot + Entangle\nBlock 1", PALETTE["quantum"], 2.3),
        ("Rot + Entangle\nBlock 2", PALETTE["quantum"], 2.3),
        ("Measure\n(expectations)", PALETTE["measure"], 2.1),
        ("Dense 1\n(optional)", PALETTE["dense"], 1.9),
    ]

    last = (x, y)
    for label, color, width in steps:
        end = add_box(ax, (last[0] + 0.6, y - 0.35), label, color, width=width)
        add_arrow(ax, last, (end[0] - width + 0.05, end[1]))
        last = end

    ax.set_xlim(x - 0.5, last[0] + 2.5)
    ax.set_ylim(y - 1.2, y + 1.2)
    ax.axis("off")


def plot_hybrid(ax):
    """Plot a conceptual hybrid flow: classical embed + quantum core + classical head."""
    ax.set_title("Hybrid pipeline (concept)", fontsize=12, pad=10)
    y = 0
    x = -3

    steps = [
        ("SMILES", PALETTE["io"], 1.8),
        ("Tokenize", PALETTE["io"], 1.8),
        ("Embed\n(vector)", PALETTE["embed"], 1.9),
        ("Classical\nEncoder", PALETTE["dense"], 2.1),
        ("Encode →\nQubit Rotations", PALETTE["embed"], 2.2),
        ("Rot + Entangle\nBlock", PALETTE["quantum"], 2.3),
        ("Measure\n(expectations)", PALETTE["measure"], 2.1),
        ("Dense 1\nHead", PALETTE["dense"], 1.9),
    ]

    last = (x, y)
    for label, color, width in steps:
        end = add_box(ax, (last[0] + 0.6, y - 0.35), label, color, width=width)
        add_arrow(ax, last, (end[0] - width + 0.05, end[1]))
        last = end

    ax.set_xlim(x - 0.5, last[0] + 2.5)
    ax.set_ylim(y - 1.2, y + 1.2)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5))
    plot_cnn(axes[0])
    plot_qnn(axes[1])
    plt.tight_layout()

    results_dir = Path("solubility") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "concept_models.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved concept diagrams to {out_path}")

    # Hybrid-only figure
    fig_h, ax_h = plt.subplots(figsize=(10, 2.5))
    plot_hybrid(ax_h)
    plt.tight_layout()
    hybrid_path = results_dir / "hybrid.png"
    fig_h.savefig(hybrid_path, dpi=300, bbox_inches="tight")
    print(f"Saved hybrid diagram to {hybrid_path}")


if __name__ == "__main__":
    main()
