"""Projection of N-dimensional point clouds into R^3.

Two projection models:
  * orthographic: rotate in R^N, keep the first 3 coordinates
    (the mathematical 'shadow')
  * perspective: rotate, then divide by the distance along each extra axis
    from a virtual viewpoint (the 4D analogue of a pinhole camera)
"""

from __future__ import annotations

import numpy as np

from .rotations import random_rotation


def orthographic(points: np.ndarray, rotation: np.ndarray | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Rotate an (n, d) cloud in R^d and keep the first 3 coordinates."""
    d = points.shape[1]
    if rotation is None:
        rotation = random_rotation(d, rng if rng is not None else np.random.default_rng())
    return (points @ rotation.T)[:, :3]


def perspective(points: np.ndarray, rotation: np.ndarray | None = None,
                rng: np.random.Generator | None = None,
                viewer_dist: float = 3.0) -> np.ndarray:
    """Rotate, then perspective-divide through each extra dimension in turn.

    Each coordinate beyond the third acts as depth for a virtual camera at
    distance `viewer_dist` along that axis (points are unit-scaled, so the
    camera is safely outside the object).
    """
    d = points.shape[1]
    if rotation is None:
        rotation = random_rotation(d, rng if rng is not None else np.random.default_rng())
    p = points @ rotation.T
    for axis in range(d - 1, 2, -1):
        w = viewer_dist / (viewer_dist - p[:, axis])
        p = p[:, :axis] * w[:, None]
    return p


PROJECTIONS = {"orthographic": orthographic, "perspective": perspective}
