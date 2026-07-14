"""The 4D-rigidity test (temporal track).

Core idea, and the main result of the benchmark:
A rigidly rotating 3D object, projected orthographically, moves rigidly in 3D.
A rigidly rotating 4D+ object projected to 3D deforms: NO rigid 3D transform
maps frame t to frame t+1. We fit the optimal rigid transform per frame pair
(Kabsch algorithm, exact point correspondence) and use the residual as a
dimensionality witness.

Statistic per sequence: mean over frame pairs of the RMS Kabsch residual.
A single threshold on this statistic is the classifier: zero learned
parameters, fully interpretable.

Usage:  python -m baselines.rigidity --data data/temporal.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kabsch_residual(a: np.ndarray, b: np.ndarray) -> float:
    """RMS residual after optimally rigidly aligning a -> b (with scaling).

    Points correspond index-wise. Allows a global scale so that the unit-
    radius normalisation applied per frame cannot create spurious residual.
    """
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    h = ac.T @ bc
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    scale = (s * [1, 1, d]).sum() / (ac ** 2).sum()
    resid = bc - scale * ac @ rot.T
    return float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def sequence_statistic(seq: np.ndarray) -> float:
    """Mean Kabsch residual over consecutive frame pairs. seq: (T, N, 3)."""
    return float(np.mean([kabsch_residual(seq[t], seq[t + 1]) for t in range(len(seq) - 1)]))


def best_threshold(stat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold maximising accuracy (higher stat => label 1)."""
    order = np.argsort(stat)
    s, lab = stat[order], y[order]
    # candidate thresholds between consecutive distinct values
    best_acc, best_thr = 0.0, s[0] - 1
    for i in range(len(s) + 1):
        thr = (s[i - 1] + s[i]) / 2 if 0 < i < len(s) else (s[0] - 1 if i == 0 else s[-1] + 1)
        acc = ((stat > thr) == y).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return float(best_thr), float(best_acc)


def auroc(stat: np.ndarray, y: np.ndarray) -> float:
    pos, neg = stat[y == 1], stat[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/temporal.npz"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    blob = np.load(args.data)
    x, y = blob["points"].astype(np.float64), blob["labels"]
    meta = json.loads((args.data.parent / (args.data.stem + "_meta.json")).read_text())["samples"]

    stat = np.array([sequence_statistic(seq) for seq in x])

    # fit threshold on half, evaluate on the other half
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    half = len(y) // 2
    fit, hold = order[:half], order[half:]
    thr, fit_acc = best_threshold(stat[fit], y[fit])
    hold_acc = float(((stat[hold] > thr) == y[hold]).mean())
    auc = auroc(stat, y)

    print(f"sequences: {len(y)}   AUROC: {auc:.4f}")
    print(f"threshold {thr:.5f}  fit acc {fit_acc:.4f}  held-out acc {hold_acc:.4f}")
    print(f"mean residual  native 3D: {stat[y == 0].mean():.5f}  "
          f"projected N-D: {stat[y == 1].mean():.5f}")

    tiers = np.array([m["tier"] for m in meta])
    per_tier = {}
    for t in sorted(set(tiers)):
        m = tiers[hold] == t
        per_tier[int(t)] = float(((stat[hold][m] > thr) == y[hold][m]).mean())
        print(f"  tier {t} held-out acc: {per_tier[int(t)]:.4f}")

    shapes = np.array([m["shape"] for m in meta])
    per_shape = {s: float(((stat[shapes == s] > thr) == y[shapes == s]).mean())
                 for s in sorted(set(shapes))}
    print("hardest shapes:", sorted(per_shape.items(), key=lambda kv: kv[1])[:5])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rigidity_results.json").write_text(json.dumps({
        "method": "kabsch_rigidity_threshold", "learned_params": 0,
        "auroc": auc, "held_out_accuracy": hold_acc, "threshold": thr,
        "mean_residual_native": float(stat[y == 0].mean()),
        "mean_residual_projected": float(stat[y == 1].mean()),
        "per_tier": per_tier, "per_shape": per_shape,
    }, indent=2))
    print(f"saved {args.out / 'rigidity_results.json'}")


if __name__ == "__main__":
    main()
