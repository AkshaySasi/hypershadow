"""Rotations in R^N: random orientations and smooth rotation paths.

Rotation paths are the key to the temporal track of the benchmark: a rigid
rotation in R^4 projected to R^3 produces deformation that no rigid 3D
motion can explain.
"""

from __future__ import annotations

import numpy as np


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix in SO(d) via QR of a Gaussian matrix."""
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))  # make the decomposition unique => Haar measure
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]  # reflect into SO(d)
    return q


def plane_rotation(d: int, i: int, j: int, angle: float) -> np.ndarray:
    """Rotation by `angle` in the (i, j) coordinate plane of R^d."""
    m = np.eye(d)
    c, s = np.cos(angle), np.sin(angle)
    m[i, i] = c
    m[j, j] = c
    m[i, j] = -s
    m[j, i] = s
    return m


def random_rotation_path(
    d: int, n_frames: int, rng: np.random.Generator, max_speed: float = 0.2
) -> np.ndarray:
    """A smooth rotation path: (n_frames, d, d) matrices R_t = expm(t*A) @ R_0.

    A is a random antisymmetric generator (constant angular velocity), so the
    motion is a rigid rotation at fixed speed in d dimensions, starting from
    a Haar-random orientation.
    """
    a = rng.standard_normal((d, d))
    gen = (a - a.T) / 2
    # normalise so the largest rotation rate is max_speed rad/frame
    w = np.abs(np.linalg.eigvals(gen)).max()
    if w > 0:
        gen *= max_speed / w
    r0 = random_rotation(d, rng)
    frames = np.empty((n_frames, d, d))
    step = _expm_antisym(gen)
    r = r0
    for t in range(n_frames):
        frames[t] = r
        r = step @ r
    return frames


def _expm_antisym(a: np.ndarray) -> np.ndarray:
    """Matrix exponential of an antisymmetric matrix via eigendecomposition.

    Antisymmetric real matrices are normal with purely imaginary spectrum, so
    exp(a) computed in the complex eigenbasis is exactly orthogonal.
    """
    vals, vecs = np.linalg.eig(a)
    return np.real(vecs @ np.diag(np.exp(vals)) @ np.linalg.inv(vecs))
