"""Higher-dimensional (4D-6D spatial) object samplers (label 1 after projection).

Every sampler returns an (n, d) float64 array with d >= 4, sampled uniformly
with respect to the natural measure of the object, roughly unit-scaled.
These live in N spatial dimensions; they become dataset items only after
being rotated in R^N and projected to R^3 (see project.py).
"""

from __future__ import annotations

import numpy as np


def hypersphere(n: int, rng: np.random.Generator, d: int = 4) -> np.ndarray:
    """Uniform on the unit sphere S^(d-1) in R^d."""
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def hyperball(n: int, rng: np.random.Generator, d: int = 4) -> np.ndarray:
    """Uniform in the solid unit ball B^d."""
    x = hypersphere(n, rng, d)
    r = rng.random(n) ** (1.0 / d)
    return x * r[:, None]


def tesseract_surface(n: int, rng: np.random.Generator, d: int = 4) -> np.ndarray:
    """Uniform on the boundary of the hypercube [-1, 1]^d.

    The boundary consists of 2*d cells, each a (d-1)-cube of equal measure:
    pick a cell, fix that coordinate to +-1, sample the rest uniformly.
    """
    cell = rng.integers(0, 2 * d, n)
    axis = cell // 2
    sign = np.where(cell % 2 == 0, 1.0, -1.0)
    pts = rng.uniform(-1, 1, (n, d))
    pts[np.arange(n), axis] = sign
    return pts


def clifford_torus(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on the Clifford torus T^2 in R^4 (flat metric => uniform angles)."""
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)], axis=1
    ) / np.sqrt(2)


def duocylinder_ridge(n: int, rng: np.random.Generator, r1: float = 1.0, r2: float = 0.6) -> np.ndarray:
    """Uniform on the duocylinder ridge {x^2+y^2=r1^2, z^2+w^2=r2^2} in R^4.

    A flat torus with unequal radii (the Clifford torus is the r1=r2 case,
    up to scale), so angles are uniform.
    """
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [r1 * np.cos(theta), r1 * np.sin(theta), r2 * np.cos(phi), r2 * np.sin(phi)],
        axis=1,
    )


def hypertorus(n: int, rng: np.random.Generator, d: int = 4, R: float = 1.0, r: float = 0.4) -> np.ndarray:
    """(d-1)-torus embedded in R^d as a tube around S^(d-2) x {0}, area-weighted.

    For d=4 this is the '3-sphere torus': the set of points at distance r
    from a great circle... generalised: take a uniformly sampled point c on
    the (d-2)-sphere of radius R in the first d-1 coordinates, then offset by
    r in the plane spanned by the radial direction and the last axis.
    Rejection-corrected for the (R + r cos phi) area factor.
    """
    pts = np.empty((0, d))
    while len(pts) < n:
        m = 2 * (n - len(pts)) + 16
        c = hypersphere(m, rng, d - 1)  # unit directions in first d-1 coords
        phi = rng.uniform(0, 2 * np.pi, m)
        keep = rng.uniform(0, 1, m) < (R + r * np.cos(phi)) / (R + r)
        c, phi = c[keep], phi[keep]
        radial = (R + r * np.cos(phi))[:, None] * c
        out = np.concatenate([radial, (r * np.sin(phi))[:, None]], axis=1)
        pts = np.concatenate([pts, out])
    return pts[:n]


def random_manifold(
    n: int,
    rng: np.random.Generator,
    d: int = 5,
    intrinsic: int = 2,
    n_freq: int = 4,
) -> np.ndarray:
    """A random smooth compact `intrinsic`-manifold embedded in R^d.

    Embeds the flat torus T^intrinsic via random trigonometric features:
    each ambient coordinate is a random low-frequency trigonometric
    polynomial of the parameters, giving a smooth closed manifold.
    """
    u = rng.uniform(0, 2 * np.pi, (n, intrinsic))
    freqs = rng.integers(1, n_freq + 1, (d, n_freq, intrinsic))
    phases = rng.uniform(0, 2 * np.pi, (d, n_freq))
    amps = rng.standard_normal((d, n_freq)) / np.sqrt(n_freq)
    # coords[i, j] = sum_k amps[j,k] * cos(u[i] . freqs[j,k] + phases[j,k])
    arg = np.einsum("ni,jki->njk", u, freqs) + phases[None, :, :]
    pts = np.einsum("jk,njk->nj", amps, np.cos(arg))
    scale = np.abs(pts).max()
    return pts / scale


SAMPLERS = {
    # name: (fn, kwargs) -- ambient dim recorded at generation time
    "hypersphere_4d": (hypersphere, {"d": 4}),
    "hypersphere_5d": (hypersphere, {"d": 5}),
    "hyperball_4d": (hyperball, {"d": 4}),
    "tesseract_4d": (tesseract_surface, {"d": 4}),
    "hypercube_5d": (tesseract_surface, {"d": 5}),
    "clifford_torus_4d": (clifford_torus, {}),
    "duocylinder_ridge_4d": (duocylinder_ridge, {}),
    "hypertorus_4d": (hypertorus, {"d": 4}),
    "hypertorus_5d": (hypertorus, {"d": 5}),
    "random_manifold_5d": (random_manifold, {"d": 5, "intrinsic": 2}),
    "random_manifold_6d": (random_manifold, {"d": 6, "intrinsic": 3}),
}


def ambient_dim(name: str) -> int:
    fn, kw = SAMPLERS[name]
    if "d" in kw:
        return kw["d"]
    return 4  # clifford torus / duocylinder live in R^4
