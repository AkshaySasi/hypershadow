"""Visual sanity check: a gallery of native 3D shapes vs projected N-D shadows.

Usage:  python -m hypershadow.visualize --out figures/gallery.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import primitives3d, primitivesnd
from .generate import make_native, make_projected


def gallery(out_path: Path, seed: int = 0, n_points: int = 2048) -> None:
    rng = np.random.default_rng(seed)
    native = list(primitives3d.SAMPLERS)
    projected = list(primitivesnd.SAMPLERS)
    n_cols = max(len(native), len(projected))

    fig = plt.figure(figsize=(2.2 * n_cols, 6.0))
    for i, name in enumerate(native):
        ax = fig.add_subplot(2, n_cols, i + 1, projection="3d")
        p = make_native(name, n_points, tier=0, rng=rng)
        _plot(ax, p, name, "tab:blue")
    for i, name in enumerate(projected):
        ax = fig.add_subplot(2, n_cols, n_cols + i + 1, projection="3d")
        p = make_projected(name, n_points, tier=0, rng=rng, projection="orthographic")
        _plot(ax, p, name, "tab:red")

    fig.suptitle("HyperShadow: top row, native 3D (label 0); bottom row, N-D projections (label 1)",
                 fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


def _plot(ax, p: np.ndarray, title: str, color: str) -> None:
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.2, alpha=0.5, c=color)
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()
    lim = np.abs(p).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figures/gallery.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gallery(args.out, args.seed)


if __name__ == "__main__":
    main()
