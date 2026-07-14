"""Schematic figures for the paper: generation pipeline, model architecture,
and the rigidity-test concept.

Usage:  python -m baselines.diagrams --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
SURFACE = "#ffffff"
BOX_FACE = "#f2f6fc"


def _box(ax, x, y, w, h, title, lines, edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=BOX_FACE, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=9.5, color=INK, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.115 - i * 0.052, ln, ha="center", va="top",
                fontsize=7.8, color=MUTED)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=MUTED, lw=1.4))


def pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.4), facecolor=SURFACE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")

    _box(ax, 0.015, 0.56, 0.17, 0.36, "N-D primitives",
         ["hypersphere S³/S⁴", "tesseract, 5-cube", "Clifford torus", "duocylinder, hypertori",
          "random manifolds"], edge=YELLOW)
    _box(ax, 0.015, 0.06, 0.17, 0.36, "3D primitives",
         ["sphere, ball, torus", "cube, cylinder", "ellipsoid", "trefoil tube"], edge=BLUE)
    _box(ax, 0.245, 0.56, 0.155, 0.36, "Rotate in R^N",
         ["Haar-uniform SO(N)", "(QR of Gaussian)"], edge=YELLOW)
    _box(ax, 0.245, 0.06, 0.155, 0.36, "Rotate in R³",
         ["same treatment", "(fairness rule)"], edge=BLUE)
    _box(ax, 0.46, 0.56, 0.155, 0.36, "Project to R³",
         ["orthographic", "or perspective"], edge=YELLOW)
    _box(ax, 0.675, 0.31, 0.145, 0.38, "Corrupt",
         ["tier 0: clean", "tier 1: + noise", "tier 2: + occlusion", "tier 3: + dropout"],
         edge=VIOLET)
    _box(ax, 0.865, 0.31, 0.125, 0.38, "Normalize",
         ["centre, unit scale", "resample 1024 pts", "label 0 / 1"], edge=AQUA)

    _arrow(ax, 0.185, 0.74, 0.245, 0.74)
    _arrow(ax, 0.185, 0.24, 0.245, 0.24)
    _arrow(ax, 0.40, 0.74, 0.46, 0.74)
    _arrow(ax, 0.615, 0.74, 0.675, 0.60)
    _arrow(ax, 0.40, 0.24, 0.675, 0.42)
    _arrow(ax, 0.82, 0.50, 0.865, 0.50)
    ax.set_title("HyperShadow generation pipeline", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out / "pipeline.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'pipeline.png'}")


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 2.7), facecolor=SURFACE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")

    stages = [
        ("Input cloud", ["1024 × 3"], BLUE),
        ("Shared MLP", ["1×1 conv 64", "BN + ReLU"], AQUA),
        ("Shared MLP", ["1×1 conv 128", "BN + ReLU"], AQUA),
        ("Shared MLP", ["1×1 conv 256", "BN + ReLU"], AQUA),
        ("Pool", ["max ⊕ mean", "512-d"], VIOLET),
        ("Head", ["256 → 64 → 2", "dropout 0.3"], YELLOW),
        ("Output", ["native 3D vs", "N-D shadow"], BLUE),
    ]
    w, gap = 0.102, 0.045
    x = 0.01
    for i, (title, lines, edge) in enumerate(stages):
        _box(ax, x, 0.22, w, 0.56, title, lines, edge=edge)
        if i < len(stages) - 1:
            _arrow(ax, x + w + 0.014, 0.50, x + w + gap - 0.012, 0.50)
        x += w + gap
    ax.set_title("PointNet-lite architecture (190,914 parameters, trains in 4 min on a 4 GB GPU)",
                 fontsize=10.5, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'architecture.png'}")


def rigidity_concept(out: Path) -> None:
    """Residual distributions of the Kabsch test, from the actual data."""
    import json
    res = json.loads(Path("results/rigidity_results.json").read_text())
    blob = np.load("data/temporal.npz")
    x, y = blob["points"].astype(np.float64), blob["labels"]

    from baselines.rigidity import sequence_statistic
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 400, replace=False)
    stat = np.array([sequence_statistic(x[i]) for i in idx])
    lab = y[idx]

    fig, ax = plt.subplots(figsize=(6.8, 3.6), facecolor=SURFACE)
    bins = np.linspace(0, max(stat.max(), 0.25), 48)
    ax.hist(stat[lab == 0], bins=bins, color=BLUE, alpha=0.85,
            label="native 3D (rigid in 3D)", zorder=3)
    ax.hist(stat[lab == 1], bins=bins, color=YELLOW, alpha=0.85,
            label="N-D shadow (rigid in R^N)", zorder=3)
    thr = res["threshold"]
    ax.axvline(thr, color=VIOLET, lw=1.6, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f"decision threshold ({thr:.3f})", (thr + 0.004, ax.get_ylim()[1] * 0.9),
                fontsize=8.5, color=VIOLET)
    ax.set_xlabel("mean Kabsch rigidity residual per sequence", fontsize=9, color=INK)
    ax.set_ylabel("sequences", fontsize=9, color=INK)
    ax.set_title(f"The rigidity witness: AUROC {res['auroc']:.3f}, 0 learned parameters",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(out / "rigidity.png", dpi=200, facecolor=SURFACE)
    print(f"saved {out / 'rigidity.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline(args.out)
    architecture(args.out)
    rigidity_concept(args.out)


if __name__ == "__main__":
    main()
