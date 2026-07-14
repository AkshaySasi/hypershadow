"""Corruption tiers applied to 3D point clouds after projection.

Tiers (cumulative difficulty):
  0 clean       -- no corruption
  1 noise       -- Gaussian jitter, sigma = 2% of cloud scale
  2 occlusion   -- + a random half-space slab of points removed
  3 sensor      -- + heavy noise, view-dependent density falloff, dropout

Every function preserves determinism given the rng and never changes the
label-relevant structure by more than a real sensor would.
"""

from __future__ import annotations

import numpy as np


def _scale(points: np.ndarray) -> float:
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def add_noise(points: np.ndarray, rng: np.random.Generator, rel_sigma: float = 0.02) -> np.ndarray:
    return points + rng.standard_normal(points.shape) * rel_sigma * _scale(points)


def occlude(points: np.ndarray, rng: np.random.Generator, frac: float = 0.3) -> np.ndarray:
    """Remove roughly `frac` of points behind a random plane (self-occlusion)."""
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    depth = points @ direction
    cutoff = np.quantile(depth, 1.0 - frac)
    kept = points[depth < cutoff]
    return kept if len(kept) >= 32 else points  # never destroy the cloud


def sensor_dropout(points: np.ndarray, rng: np.random.Generator, keep_frac: float = 0.7) -> np.ndarray:
    """Distance-dependent dropout from a random viewpoint (LiDAR-like sparsity)."""
    view = rng.standard_normal(3)
    view = 4.0 * view / np.linalg.norm(view) * _scale(points)
    dist = np.linalg.norm(points - view, axis=1)
    p_keep = keep_frac * (dist.min() / dist) ** 2
    kept = points[rng.random(len(points)) < np.clip(p_keep, 0.05, 1.0)]
    return kept if len(kept) >= 32 else points


def apply_tier(points: np.ndarray, tier: int, rng: np.random.Generator) -> np.ndarray:
    if tier >= 1:
        points = add_noise(points, rng, rel_sigma=0.02)
    if tier >= 2:
        points = occlude(points, rng, frac=float(rng.uniform(0.2, 0.4)))
    if tier >= 3:
        points = add_noise(points, rng, rel_sigma=0.03)
        points = sensor_dropout(points, rng, keep_frac=0.7)
    return points


N_TIERS = 4
