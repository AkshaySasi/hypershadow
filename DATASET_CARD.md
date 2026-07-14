---
license: mit
task_categories:
  - other
tags:
  - point-cloud
  - geometry
  - topology
  - benchmark
  - higher-dimensions
  - synthetic
pretty_name: HyperShadow
size_categories:
  - 10K<n<100K
---

# HyperShadow

A benchmark for one question: given a 3D point cloud, can you tell whether
it is an ordinary 3D object or the 3D projection (the "shadow") of an
object from a higher spatial dimension?

Most datasets that say "4D" mean 3D plus time. Here the extra dimensions
are spatial: label 1 clouds are projections of objects living in R^4, R^5
or R^6 (hyperspheres, tesseracts, Clifford tori, duocylinders, hypertori,
random smooth manifolds), rotated randomly in their ambient space before
projection. Label 0 clouds are native 3D shapes given the same rotation
and corruption treatment, so nothing in the preprocessing separates the
classes.

## Why this is an interesting task

A shadow is still at most 3-dimensional data, so intrinsic-dimension
estimation cannot solve this. On this benchmark, TwoNN and the
Levina-Bickel MLE reach about 71-73% accuracy. What actually separates
the classes are the traces the projection map leaves behind: density
folds, volumes filled with a particular radial profile, changed topology.
A small PointNet (190k parameters) picks these up at 96.6%, and still
detects shape families it was never trained on (79-91%).

The temporal track has the strongest result, and it needs no learning at
all. A rigid 3D object in rigid motion can be aligned frame to frame by a
rigid 3D transform with near-zero residual. The shadow of a rigidly
rotating 4D object cannot: no rigid 3D motion explains it. The mean
Kabsch alignment residual, one number per sequence, separates the classes
at AUROC 0.982.

## Files

| file | contents |
|---|---|
| `static.npz` | `points`: float32 (10800, 1024, 3); `labels`: int64 (10800,) |
| `static_meta.json` | per-sample shape name, ambient dimension, corruption tier, projection type |
| `temporal.npz` | `points`: float32 (1800, 16, 512, 3), rigid-rotation sequences; `labels` |
| `temporal_meta.json` | per-sequence metadata |

Labels: 0 = native 3D, 1 = projection of a higher-dimensional object.
Classes are balanced across corruption tiers. All clouds are centred,
scaled to unit mean radius, and resampled to a fixed point count.

Corruption tiers (cumulative): 0 clean; 1 Gaussian jitter (2% of scale);
2 plus random half-space occlusion; 3 plus heavier jitter and
distance-dependent dropout.

## Loading

```python
import numpy as np

d = np.load("static.npz")
x, y = d["points"], d["labels"]     # (10800, 1024, 3), (10800,)

t = np.load("temporal.npz")
seq = t["points"]                    # (1800, 16, 512, 3)
```

## Baseline numbers to beat

| Method | Accuracy (static) |
|---|---|
| intrinsic dimension, best threshold | 0.737 +/- 0.004 |
| persistent homology + GBT | 0.904 |
| geometric features + GBT | 0.956 |
| PointNet-lite, 190k params | 0.962 +/- 0.003 |

Uncertainty is the standard deviation over five random splits or training
seeds. Temporal track: Kabsch rigidity residual threshold, AUROC 0.982,
held-out accuracy 0.978 +/- 0.004.

## Reproducing or extending

The generator is plain seeded NumPy, no GPU needed. Code, tests, and all
baseline implementations are at
https://github.com/AkshaySasi/hypershadow. Trained checkpoints are at
https://huggingface.co/AkshaySasi/hypershadow-models. Regenerating with the published
seeds reproduces these exact files. New shape families can be added with
a single sampler function.

## Limitations

Everything here is simulated, under specific choices of sampling measure,
projection model and noise. Results on this data say nothing about
physical reality. The intended uses are: benchmarking point-cloud models
on a task where the ground truth is known and the shortcut routes have
been closed off, studying out-of-distribution detection with a
mathematically defined "out", and comparing machine performance against
the human 4D-rigidity perception results reported by He, Bi and Zaidi
(2023).

## Citation

```bibtex
@misc{hypershadow2026,
  title  = {HyperShadow: A Benchmark for Detecting 3D Projections of
            Higher-Dimensional Spatial Objects},
  author = {Sasi, Akshay},
  year   = {2026},
}
```
