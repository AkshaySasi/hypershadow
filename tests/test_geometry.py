"""Correctness tests for the HyperShadow geometry and generation pipeline."""

import numpy as np
import pytest

from hypershadow import corruptions, primitives3d, primitivesnd
from hypershadow.generate import make_native, make_projected, normalize, resample
from hypershadow.project import orthographic, perspective
from hypershadow.rotations import random_rotation, random_rotation_path

RNG = lambda: np.random.default_rng(42)  # noqa: E731


# ---------- primitives ----------

@pytest.mark.parametrize("name", list(primitives3d.SAMPLERS))
def test_native_shapes_and_finiteness(name):
    p = primitives3d.SAMPLERS[name](500, RNG())
    assert p.shape == (500, 3)
    assert np.isfinite(p).all()


@pytest.mark.parametrize("name", list(primitivesnd.SAMPLERS))
def test_nd_shapes_and_finiteness(name):
    fn, kw = primitivesnd.SAMPLERS[name]
    p = fn(500, RNG(), **kw)
    assert p.shape == (500, primitivesnd.ambient_dim(name))
    assert np.isfinite(p).all()


def test_sphere_on_unit_sphere():
    p = primitives3d.sphere(1000, RNG())
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)


def test_hypersphere_on_unit_sphere():
    p = primitivesnd.hypersphere(1000, RNG(), d=5)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)


def test_clifford_torus_constraints():
    p = primitivesnd.clifford_torus(1000, RNG())
    assert np.allclose(p[:, 0] ** 2 + p[:, 1] ** 2, 0.5)
    assert np.allclose(p[:, 2] ** 2 + p[:, 3] ** 2, 0.5)


def test_tesseract_on_boundary():
    p = primitivesnd.tesseract_surface(1000, RNG(), d=4)
    assert np.allclose(np.abs(p).max(axis=1), 1.0)


def test_torus_area_uniformity():
    """Rejection sampling should not oversample the inner rim."""
    p = primitives3d.torus(20000, RNG(), R=1.0, r=0.4)
    # angle around the tube: outer points have radial distance > R
    radial = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    outer_frac = (radial > 1.0).mean()
    # area-uniform sampling puts MORE than half the points on the outer half
    assert outer_frac > 0.55


# ---------- rotations ----------

@pytest.mark.parametrize("d", [3, 4, 5, 6])
def test_random_rotation_is_special_orthogonal(d):
    r = random_rotation(d, RNG())
    assert np.allclose(r @ r.T, np.eye(d), atol=1e-10)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_rotation_path_stays_orthogonal_and_smooth():
    path = random_rotation_path(4, 32, RNG(), max_speed=0.2)
    for t in range(32):
        assert np.allclose(path[t] @ path[t].T, np.eye(4), atol=1e-8)
    # consecutive frames differ by a small rotation
    step = path[1] @ path[0].T
    angle = np.arccos(np.clip((np.trace(step) - 2) / 2, -1, 1))
    assert 0 < angle < 0.5


# ---------- projection ----------

def test_orthographic_output_shape():
    p = primitivesnd.hypersphere(200, RNG(), d=5)
    q = orthographic(p, rng=RNG())
    assert q.shape == (200, 3)


def test_perspective_output_shape_and_finiteness():
    p = primitivesnd.hypersphere(200, RNG(), d=6)
    q = perspective(p, rng=RNG())
    assert q.shape == (200, 3)
    assert np.isfinite(q).all()


def test_projected_hypersphere_fills_ball():
    """The shadow of S^3 is a solid ball, not a 2-sphere: interior points exist."""
    p = primitivesnd.hypersphere(5000, RNG(), d=4)
    q = orthographic(p, rng=RNG())
    r = np.linalg.norm(q, axis=1)
    assert (r < 0.5).mean() > 0.05  # plenty of interior mass
    assert r.max() <= 1.0 + 1e-9


# ---------- corruption / pipeline ----------

def test_tiers_preserve_validity():
    p = primitives3d.sphere(2000, RNG())
    for tier in range(corruptions.N_TIERS):
        out = corruptions.apply_tier(p.copy(), tier, RNG())
        assert np.isfinite(out).all()
        assert len(out) >= 32


def test_normalize_and_resample():
    p = primitives3d.torus(700, RNG())
    q = normalize(resample(p, 1024, RNG()))
    assert q.shape == (1024, 3)
    assert np.allclose(q.mean(axis=0), 0, atol=1e-9)
    assert np.isclose(np.linalg.norm(q, axis=1).mean(), 1.0)


def test_make_native_and_projected_are_deterministic():
    a = make_native("torus", 256, 2, np.random.default_rng(7))
    b = make_native("torus", 256, 2, np.random.default_rng(7))
    assert np.array_equal(a, b)
    a = make_projected("hypersphere_4d", 256, 2, np.random.default_rng(7), "perspective")
    b = make_projected("hypersphere_4d", 256, 2, np.random.default_rng(7), "perspective")
    assert np.array_equal(a, b)
