"""HyperShadow: a benchmark dataset of 3D projections of higher-dimensional objects.

Generates point clouds of two kinds:
  * native 3D shapes (label 0)
  * 3D projections of 4D-6D spatial objects (label 1)

with controlled corruption tiers, for the task of detecting whether a 3D
point cloud is the "shadow" of a higher-dimensional object.
"""

__version__ = "0.1.0"

from . import primitives3d, primitivesnd, rotations, project, corruptions

__all__ = ["primitives3d", "primitivesnd", "rotations", "project", "corruptions"]
