"""Dataset generation for the HyperShadow benchmark.

Static track:   (n_points, 3) clouds, label 0 = native 3D, 1 = projected N-D.
Temporal track: (n_frames, n_points, 3) sequences of a rigidly rotating
                object; for label 1 the rigid rotation happens in R^N.

Fairness rules (so a classifier cannot cheat):
  * native 3D shapes get the same random SO(3) orientation treatment
  * every cloud is centred and scaled to unit mean radius after corruption
  * every cloud is resampled to exactly n_points

Usage:
  python -m hypershadow.generate --out data/ --per-class 2000 --seed 0
  python -m hypershadow.generate --out data/ --temporal --per-class 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import corruptions, primitives3d, primitivesnd
from .project import PROJECTIONS
from .rotations import random_rotation, random_rotation_path


def normalize(points: np.ndarray) -> np.ndarray:
    """Centre at the origin and scale to unit mean radius."""
    p = points - points.mean(axis=0)
    r = np.linalg.norm(p, axis=1).mean()
    return p / r if r > 0 else p


def resample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return exactly n points (subsample without, or pad with, replacement)."""
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
    else:
        idx = rng.choice(len(points), n, replace=True)
    return points[idx]


def make_native(name: str, n_points: int, tier: int, rng: np.random.Generator) -> np.ndarray:
    raw = primitives3d.SAMPLERS[name](2 * n_points, rng)
    raw = raw @ random_rotation(3, rng).T
    raw = corruptions.apply_tier(raw, tier, rng)
    return normalize(resample(raw, n_points, rng))


def make_projected(
    name: str, n_points: int, tier: int, rng: np.random.Generator, projection: str
) -> np.ndarray:
    fn, kw = primitivesnd.SAMPLERS[name]
    raw = fn(2 * n_points, rng, **kw)
    raw = PROJECTIONS[projection](raw, rng=rng)
    raw = corruptions.apply_tier(raw, tier, rng)
    return normalize(resample(raw, n_points, rng))


def generate_static(
    out_dir: Path,
    per_class: int,
    n_points: int,
    seed: int,
) -> dict:
    """per_class = clouds per shape name (split across tiers and projections)."""
    rng = np.random.default_rng(seed)
    clouds, labels, meta = [], [], []

    for name in primitives3d.SAMPLERS:
        for k in range(per_class):
            tier = k % corruptions.N_TIERS
            clouds.append(make_native(name, n_points, tier, rng))
            labels.append(0)
            meta.append({"shape": name, "ambient_dim": 3, "tier": tier, "projection": "none"})

    proj_names = list(PROJECTIONS)
    for name in primitivesnd.SAMPLERS:
        for k in range(per_class):
            tier = k % corruptions.N_TIERS
            projection = proj_names[(k // corruptions.N_TIERS) % len(proj_names)]
            clouds.append(make_projected(name, n_points, tier, rng, projection))
            labels.append(1)
            meta.append(
                {
                    "shape": name,
                    "ambient_dim": primitivesnd.ambient_dim(name),
                    "tier": tier,
                    "projection": projection,
                }
            )

    x = np.stack(clouds).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    meta = [meta[i] for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "static.npz", points=x, labels=y)
    info = {
        "track": "static",
        "n_samples": int(len(y)),
        "n_points": n_points,
        "seed": seed,
        "class_balance": {"native_3d": int((y == 0).sum()), "projected_nd": int((y == 1).sum())},
        "samples": meta,
    }
    (out_dir / "static_meta.json").write_text(json.dumps(info, indent=1))
    return info


def generate_temporal(
    out_dir: Path,
    per_class: int,
    n_points: int,
    n_frames: int,
    seed: int,
) -> dict:
    """Sequences of one rigidly rotating object, projected each frame.

    Label 0: rigid rotation of a 3D shape (orthographic projection is the
    identity on the first 3 coords, so frames are genuinely rigid in 3D).
    Label 1: rigid rotation in R^N projected to 3D each frame -- the shadow
    deforms non-rigidly.
    """
    rng = np.random.default_rng(seed)
    seqs, labels, meta = [], [], []

    def track(points_nd: np.ndarray, tier: int) -> np.ndarray:
        d = points_nd.shape[1]
        path = random_rotation_path(d, n_frames, rng)
        # fixed point identity across frames: corrupt indices consistently
        base = resample(points_nd, n_points, rng)
        frames = np.empty((n_frames, n_points, 3), dtype=np.float32)
        for t in range(n_frames):
            p3 = (base @ path[t].T)[:, :3]
            p3 = corruptions.add_noise(p3, rng, 0.02) if tier >= 1 else p3
            frames[t] = normalize(p3)
        return frames

    for name in primitives3d.SAMPLERS:
        for k in range(per_class):
            tier = k % 2
            raw = primitives3d.SAMPLERS[name](2 * n_points, rng)
            seqs.append(track(raw, tier))
            labels.append(0)
            meta.append({"shape": name, "ambient_dim": 3, "tier": tier})

    for name in primitivesnd.SAMPLERS:
        for k in range(per_class):
            tier = k % 2
            fn, kw = primitivesnd.SAMPLERS[name]
            raw = fn(2 * n_points, rng, **kw)
            seqs.append(track(raw, tier))
            labels.append(1)
            meta.append({"shape": name, "ambient_dim": primitivesnd.ambient_dim(name), "tier": tier})

    x = np.stack(seqs).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    meta = [meta[i] for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "temporal.npz", points=x, labels=y)
    info = {
        "track": "temporal",
        "n_samples": int(len(y)),
        "n_points": n_points,
        "n_frames": n_frames,
        "seed": seed,
        "class_balance": {"native_3d": int((y == 0).sum()), "projected_nd": int((y == 1).sum())},
        "samples": meta,
    }
    (out_dir / "temporal_meta.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--per-class", type=int, default=200, help="samples per shape name")
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--n-frames", type=int, default=16, help="temporal track only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temporal", action="store_true")
    args = ap.parse_args()

    if args.temporal:
        info = generate_temporal(args.out, args.per_class, args.n_points, args.n_frames, args.seed)
    else:
        info = generate_static(args.out, args.per_class, args.n_points, args.seed)
    print(json.dumps({k: v for k, v in info.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
