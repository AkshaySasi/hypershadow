"""Classical intrinsic-dimension estimators on the static track.

The scientific point: a 3D projection of a 4D object is still at most
3-dimensional data, so intrinsic dimension per se cannot identify it as a
shadow. ID estimators do carry partial signal (many projected surfaces fill
volumes, ID~3, while most native shapes are surfaces, ID~2) -- but the native
solid ball and the projected 2-manifolds break that shortcut. We quantify
exactly how far ID alone gets, as the failure baseline that learned methods
must beat.

Estimators (standard, self-contained implementations):
  * TwoNN (Facco et al., 2017): d = log 2 / mean log(r2/r1)   [we use the
    MLE form d = (n-1) / sum log(mu_i) which is its maximum-likelihood fit]
  * Levina-Bickel MLE (2005) with k neighbours, averaged over points

Usage:  python -m baselines.id_estimators --data data/static.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(p: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=3).fit(p)
    dist, _ = nn.kneighbors(p)
    r1, r2 = dist[:, 1], dist[:, 2]
    good = r1 > 1e-12  # padded clouds contain duplicate points
    mu = np.log(r2[good] / r1[good])
    mu = mu[mu > 1e-12]
    return float(len(mu) / mu.sum()) if len(mu) else 0.0


def levina_bickel(p: np.ndarray, k: int = 10) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(p)
    dist, _ = nn.kneighbors(p)
    dist = np.maximum(dist, 1e-12)
    rk = dist[:, k][:, None]
    inv = np.log(rk / dist[:, 1:k]).mean(axis=1)
    inv = inv[inv > 1e-12]
    return float(1.0 / inv.mean()) if len(inv) else 0.0


def best_threshold_acc(stat: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the best single threshold (either direction), fit on half."""
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    half = len(y) // 2
    fit, hold = order[:half], order[half:]
    cands = np.quantile(stat[fit], np.linspace(0, 1, 201))
    best = max(
        max(((stat[fit] > t) == y[fit]).mean(), ((stat[fit] < t) == y[fit]).mean())
        for t in cands
    )
    # refit the winning (threshold, direction) and score held-out
    best_t, best_dir, best_acc = None, 1, 0.0
    for t in cands:
        for d in (1, -1):
            acc = ((d * stat[fit] > d * t) == y[fit]).mean()
            if acc > best_acc:
                best_t, best_dir, best_acc = t, d, acc
    return float(((best_dir * stat[hold] > best_dir * best_t) == y[hold]).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/static.npz"))
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    blob = np.load(args.data)
    x, y = blob["points"].astype(np.float64), blob["labels"]
    if len(y) > args.max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), args.max_samples, replace=False)
        x, y = x[idx], y[idx]

    ids_twonn = np.array([twonn(c) for c in x])
    ids_lb = np.array([levina_bickel(c) for c in x])

    res = {}
    for name, stat in [("twonn", ids_twonn), ("levina_bickel", ids_lb)]:
        acc = best_threshold_acc(stat, y)
        res[name] = {
            "held_out_threshold_accuracy": acc,
            "mean_id_native": float(stat[y == 0].mean()),
            "mean_id_projected": float(stat[y == 1].mean()),
        }
        print(f"{name:14s}  ID native {stat[y == 0].mean():.2f}  "
              f"ID projected {stat[y == 1].mean():.2f}  threshold acc {acc:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "id_estimator_results.json").write_text(json.dumps(res, indent=2))
    print(f"saved {args.out / 'id_estimator_results.json'}")


if __name__ == "__main__":
    main()
