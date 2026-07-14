"""Hand-crafted geometric features + classical classifier baseline.

Purpose: dataset validation. If this scores ~50% the task may be impossible;
if it scores ~100% on all tiers the task is trivial and the benchmark is
uninteresting. A publish-worthy dataset sits in between, with accuracy
degrading across corruption tiers and (later) learned models beating this.

Features per cloud (all rotation-invariant):
  * histogram of pairwise distances (16 bins)      -- shape distribution (D2)
  * histogram of radii from centroid (16 bins)     -- radial mass profile
  * PCA eigenvalue spectrum (3)                    -- anisotropy
  * local intrinsic-dimension proxy: mean log ratio of kNN distances (2)
  * nearest-neighbour distance stats (2)           -- surface vs volume density

Usage: python -m baselines.features --data data_pilot/static.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors


def cloud_features(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(p)
    # D2 shape distribution on a subsample
    idx = rng.choice(n, min(n, 256), replace=False)
    sub = p[idx]
    d = np.linalg.norm(sub[:, None] - sub[None, :], axis=2)
    d = d[np.triu_indices(len(sub), k=1)]
    d2_hist, _ = np.histogram(d, bins=16, range=(0, 3.0), density=True)

    # radial profile (clouds are unit mean radius)
    r = np.linalg.norm(p, axis=1)
    r_hist, _ = np.histogram(r, bins=16, range=(0, 2.5), density=True)

    # anisotropy
    eig = np.linalg.eigvalsh(np.cov(p.T))
    eig = eig / eig.sum()

    # local dimension proxy: for volume-filling data log(d_k/d_j) ~ smaller slope
    nn = NearestNeighbors(n_neighbors=9).fit(p)
    dist, _ = nn.kneighbors(p)
    dist = np.maximum(dist, 1e-12)  # padded clouds contain duplicate points
    ratio_a = np.log(dist[:, 8] / dist[:, 4]).mean()
    ratio_b = np.log(dist[:, 4] / dist[:, 2]).mean()
    nn_mean = dist[:, 1].mean()
    nn_std = dist[:, 1].std()

    return np.concatenate([d2_hist, r_hist, eig, [ratio_a, ratio_b, nn_mean, nn_std]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data_pilot/static.npz"))
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()

    blob = np.load(args.data)
    x_raw, y = blob["points"], blob["labels"]
    meta_path = args.meta or args.data.parent / (args.data.stem + "_meta.json")
    meta = json.loads(meta_path.read_text())["samples"]

    rng = np.random.default_rng(0)
    feats = np.stack([cloud_features(c.astype(np.float64), rng) for c in x_raw])

    clf = HistGradientBoostingClassifier(random_state=0)
    pred = cross_val_predict(clf, feats, y, cv=5)
    acc = (pred == y).mean()
    print(f"overall 5-fold accuracy: {acc:.3f}  (n={len(y)}, chance=0.5)")

    tiers = np.array([m["tier"] for m in meta])
    per_tier = {}
    for t in sorted(set(tiers)):
        mask = tiers == t
        per_tier[int(t)] = float((pred[mask] == y[mask]).mean())
        print(f"  tier {t}: acc={per_tier[int(t)]:.3f}  (n={mask.sum()})")

    # per-shape breakdown of errors
    shapes = np.array([m["shape"] for m in meta])
    print("hardest shapes (lowest accuracy):")
    per = sorted(
        (( (pred[shapes == s] == y[shapes == s]).mean(), s) for s in set(shapes))
    )[:6]
    for a, s in per:
        print(f"  {s}: {a:.3f}")

    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "features_results.json").write_text(json.dumps({
        "method": "geometric_features_hgb", "accuracy": float(acc),
        "per_tier": per_tier,
        "per_shape": {s: float((pred[shapes == s] == y[shapes == s]).mean())
                      for s in sorted(set(shapes))},
        "n": int(len(y)),
    }, indent=2))
    print(f"saved {out / 'features_results.json'}")


if __name__ == "__main__":
    main()
