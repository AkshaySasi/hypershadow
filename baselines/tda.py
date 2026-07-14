"""Topological baseline: persistent-homology features + gradient boosting.

Projection changes topology: shadows self-intersect, fill voids, and create
density folds, so H1/H2 persistence statistics should differ between native
surfaces and projected N-D objects even when local geometry looks similar.

Features per cloud (from a 256-point subsample for tractability):
  * H0/H1/H2 persistence: count, max, mean, sum of lifetimes
  * top-5 H1 and H2 lifetimes (padded)
Classifier: HistGradientBoosting, 5-fold CV, per-tier breakdown.

Usage:  python -m baselines.tda --data data/static.npz --max-samples 4000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ripser import ripser
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict


def persistence_features(p: np.ndarray, rng: np.random.Generator, n_sub: int = 256) -> np.ndarray:
    idx = rng.choice(len(p), min(len(p), n_sub), replace=False)
    dgms = ripser(p[idx], maxdim=2)["dgms"]
    feats = []
    for dim, dgm in enumerate(dgms):
        life = dgm[:, 1] - dgm[:, 0]
        life = life[np.isfinite(life)]
        feats += [len(life), life.max() if len(life) else 0.0,
                  life.mean() if len(life) else 0.0, life.sum() if len(life) else 0.0]
        if dim >= 1:
            top = np.sort(life)[::-1][:5]
            feats += list(np.pad(top, (0, 5 - len(top))))
    return np.array(feats, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/static.npz"))
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    blob = np.load(args.data)
    x, y = blob["points"].astype(np.float64), blob["labels"]
    meta = json.loads((args.data.parent / (args.data.stem + "_meta.json")).read_text())["samples"]
    if len(y) > args.max_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(y), args.max_samples, replace=False))
        x, y = x[idx], y[idx]
        meta = [meta[i] for i in idx]

    rng = np.random.default_rng(0)
    feats = []
    for i, c in enumerate(x):
        feats.append(persistence_features(c, rng))
        if i % 200 == 0:
            print(f"persistence {i}/{len(x)}", flush=True)
    feats = np.stack(feats)

    clf = HistGradientBoostingClassifier(random_state=0)
    pred = cross_val_predict(clf, feats, y, cv=5)
    acc = float((pred == y).mean())
    print(f"TDA 5-fold accuracy: {acc:.4f}  (n={len(y)})")

    tiers = np.array([m["tier"] for m in meta])
    per_tier = {int(t): float((pred[tiers == t] == y[tiers == t]).mean())
                for t in sorted(set(tiers))}
    for t, a in per_tier.items():
        print(f"  tier {t}: {a:.4f}")

    shapes = np.array([m["shape"] for m in meta])
    per_shape = {s: float((pred[shapes == s] == y[shapes == s]).mean())
                 for s in sorted(set(shapes))}
    print("hardest shapes:", sorted(per_shape.items(), key=lambda kv: kv[1])[:5])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tda_results.json").write_text(json.dumps({
        "method": "persistence_features_hgb", "accuracy": acc,
        "per_tier": per_tier, "per_shape": per_shape, "n": len(y),
    }, indent=2))
    print(f"saved {args.out / 'tda_results.json'}")


if __name__ == "__main__":
    main()
