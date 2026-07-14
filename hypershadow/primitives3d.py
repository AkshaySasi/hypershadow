"""Native 3D shape samplers (label 0: not a higher-dimensional projection).

Every sampler returns an (n, 3) float64 array of points sampled uniformly
(with respect to surface area or volume, as noted) and roughly unit-scaled.
"""

from __future__ import annotations

import numpy as np


def sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on the unit sphere S^2."""
    x = rng.standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ball(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform in the solid unit ball B^3."""
    x = sphere(n, rng)
    r = rng.random(n) ** (1.0 / 3.0)
    return x * r[:, None]


def torus(n: int, rng: np.random.Generator, R: float = 1.0, r: float = 0.4) -> np.ndarray:
    """Area-uniform on a torus of major radius R, minor radius r.

    Uses rejection sampling on the minor angle: the surface area element is
    proportional to (R + r*cos(phi)), so naive uniform angles oversample the
    inner rim.
    """
    pts = np.empty((0, 3))
    while len(pts) < n:
        m = 2 * (n - len(pts)) + 16
        theta = rng.uniform(0, 2 * np.pi, m)
        phi = rng.uniform(0, 2 * np.pi, m)
        keep = rng.uniform(0, 1, m) < (R + r * np.cos(phi)) / (R + r)
        theta, phi = theta[keep], phi[keep]
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        pts = np.concatenate([pts, np.stack([x, y, z], axis=1)])
    return pts[:n]


def cube_surface(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on the surface of the cube [-1, 1]^3 (6 faces, equal area)."""
    face = rng.integers(0, 6, n)
    axis = face // 2
    sign = np.where(face % 2 == 0, 1.0, -1.0)
    pts = rng.uniform(-1, 1, (n, 3))
    pts[np.arange(n), axis] = sign
    return pts


def cylinder(n: int, rng: np.random.Generator, r: float = 0.5, h: float = 2.0) -> np.ndarray:
    """Area-uniform on a closed cylinder (side + two caps)."""
    side_area = 2 * np.pi * r * h
    cap_area = np.pi * r * r
    total = side_area + 2 * cap_area
    u = rng.random(n)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.empty((n, 3))
    on_side = u < side_area / total
    ns = int(on_side.sum())
    pts[on_side, 0] = r * np.cos(theta[on_side])
    pts[on_side, 1] = r * np.sin(theta[on_side])
    pts[on_side, 2] = rng.uniform(-h / 2, h / 2, ns)
    off = ~on_side
    rad = r * np.sqrt(rng.random(int(off.sum())))
    pts[off, 0] = rad * np.cos(theta[off])
    pts[off, 1] = rad * np.sin(theta[off])
    pts[off, 2] = np.where(rng.random(int(off.sum())) < 0.5, h / 2, -h / 2)
    return pts


def ellipsoid(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximately area-uniform on a random ellipsoid via rejection.

    Points are drawn uniformly on the sphere, mapped to the ellipsoid, and
    accepted proportionally to the local area distortion.
    """
    axes = rng.uniform(0.4, 1.2, 3)
    pts = np.empty((0, 3))
    while len(pts) < n:
        m = 2 * (n - len(pts)) + 16
        s = sphere(m, rng)
        # area distortion factor of the map x -> axes*x on the sphere
        g = np.sqrt(((s * axes[[1, 2, 0]] * axes[[2, 0, 1]]) ** 2).sum(axis=1))
        keep = rng.uniform(0, g.max(), m) < g
        pts = np.concatenate([pts, s[keep] * axes])
    return pts[:n]


def trefoil_tube(n: int, rng: np.random.Generator, tube_r: float = 0.15) -> np.ndarray:
    """Points on a tube around a trefoil knot (a topologically nontrivial 3D shape)."""
    t = rng.uniform(0, 2 * np.pi, n)
    c = np.stack(
        [
            np.sin(t) + 2 * np.sin(2 * t),
            np.cos(t) - 2 * np.cos(2 * t),
            -np.sin(3 * t),
        ],
        axis=1,
    ) / 3.0
    # tangent via analytic derivative
    d = np.stack(
        [
            np.cos(t) + 4 * np.cos(2 * t),
            -np.sin(t) + 4 * np.sin(2 * t),
            -3 * np.cos(3 * t),
        ],
        axis=1,
    )
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # arbitrary normal frame
    ref = np.tile([0.0, 0.0, 1.0], (n, 1))
    n1 = np.cross(d, ref)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(d, n1)
    phi = rng.uniform(0, 2 * np.pi, n)
    return c + tube_r * (np.cos(phi)[:, None] * n1 + np.sin(phi)[:, None] * n2)


SAMPLERS = {
    "sphere": sphere,
    "ball": ball,
    "torus": torus,
    "cube_surface": cube_surface,
    "cylinder": cylinder,
    "ellipsoid": ellipsoid,
    "trefoil_tube": trefoil_tube,
}
