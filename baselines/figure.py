"""Paper figure: method accuracy per corruption tier on the static track.

Reads results/*.json produced by the baseline scripts and renders a grouped
bar chart with the intrinsic-dimension ceiling as a reference line.

Usage:  python -m baselines.figure --out figures/results.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#ffffff"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"Geometric features + GBT": "#2a78d6",
          "Persistent homology + GBT": "#1baf7a",
          "PointNet-lite (learned)": "#eda100"}
ID_COLOR = "#4a3aa7"


def load(results_dir: Path) -> tuple[dict, float]:
    methods = {}
    feats = results_dir / "features_results.json"
    if feats.exists():
        methods["Geometric features + GBT"] = json.loads(feats.read_text())["per_tier"]
    tda = results_dir / "tda_results.json"
    if tda.exists():
        methods["Persistent homology + GBT"] = json.loads(tda.read_text())["per_tier"]
    pn = results_dir / "pointnet_results.json"
    if pn.exists():
        methods["PointNet-lite (learned)"] = json.loads(pn.read_text())["per_tier"]
    idr = results_dir / "id_estimator_results.json"
    id_acc = None
    if idr.exists():
        d = json.loads(idr.read_text())
        id_acc = max(v["held_out_threshold_accuracy"] for v in d.values())
    return methods, id_acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures/results.png"))
    args = ap.parse_args()

    methods, id_acc = load(args.results)
    if not methods:
        raise SystemExit("no results found in " + str(args.results))

    tiers = ["clean", "noise", "occlusion", "sensor"]
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(len(tiers))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (name, per_tier) in enumerate(methods.items()):
        vals = [per_tier.get(str(t), per_tier.get(t, np.nan)) for t in range(4)]
        pos = x + (i - (n_m - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, color=SERIES[name], label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}".lstrip("0"), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4)

    if id_acc is not None:
        ax.axhline(id_acc, color=ID_COLOR, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"intrinsic-dimension ceiling ({id_acc:.2f})",
                    (len(tiers) - 0.52, id_acc + 0.008), ha="right", fontsize=8,
                    color=ID_COLOR)
    ax.axhline(0.5, color=BASELINE, lw=1.0, zorder=2)
    ax.annotate("chance", (-0.42, 0.505), fontsize=8, color=MUTED)

    ax.set_ylim(0.45, 1.02)
    ax.set_xticks(x, [f"tier {i}\n{t}" for i, t in enumerate(tiers)], fontsize=9)
    ax.set_ylabel("held-out accuracy", fontsize=9, color=INK)
    ax.set_title("Detecting higher-dimensional shadows: accuracy by corruption tier",
                 fontsize=10.5, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
