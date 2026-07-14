# HyperShadow: A Benchmark for Detecting 3D Projections of Higher-Dimensional Spatial Objects

**Akshay Sasi**
akshaysasi12.knr@gmail.com

---

## Abstract

Machine-learning datasets labelled "4D" universally denote three spatial dimensions plus time. We introduce **HyperShadow**, the first public benchmark in which the fourth, fifth, and sixth dimensions are *spatial*: the task is to decide whether a 3D point cloud is a native three-dimensional shape or the projection — the "shadow" — of a rigid object living in R^N (N = 4–6). We show this task is fundamentally distinct from intrinsic-dimension estimation: a shadow is still at-most-3-dimensional data, and standard estimators (TwoNN, Levina–Bickel MLE) reach only 71–73% accuracy. Detection instead requires *projection signatures* — density folds, filled volumes with characteristic radial profiles, and topology changes — which a 190k-parameter point network recovers at 96.6% accuracy across four corruption tiers, generalizing at 79–91% to object families never seen in training. On a temporal track of rigidly rotating objects we introduce a zero-parameter **rigidity witness**: the residual of the optimal rigid 3D alignment (Kabsch) between consecutive frames, which must vanish for any rigid 3D motion but cannot vanish for the shadow of a rigid rotation in R^N. This single interpretable statistic separates the classes at AUROC 0.982. All data are generated reproducibly from seeds; the dataset, models, and code are released publicly. HyperShadow makes no claim about physical reality; it is a controlled instrument for studying which observable statistics can *certify incompatibility with a purely three-dimensional explanation*.

---

## 1. Introduction

A cube held before a lamp casts a two-dimensional shadow. The shadow is genuinely 2D — every pixel of it lies in the plane — yet it is not a *generic* 2D shape: as the cube rotates, the shadow deforms in ways no rigid flat object could, and even a single frame carries statistical traces of the projection that produced it. This paper asks the analogous question one dimension up, computationally: **given only a 3D point cloud, or a short sequence of them, can any method decide whether the data is the projection of a higher-dimensional rigid object?**

Three observations motivate the benchmark.

**(1) The task is not intrinsic-dimension estimation.** A rich literature estimates the dimension of the manifold underlying observed data [1, 2, 3]. But a projection *reduces* dimension: the orthographic shadow of the 3-sphere S³ ⊂ R⁴ is a solid 3-ball; the shadow of the 2-dimensional Clifford torus remains 2-dimensional. Intrinsic dimension of the observed data therefore cannot, even in principle, fully identify shadows. What distinguishes them is *how* mass, density, and topology are arranged — the signatures of the projection map itself. Section 6.1 quantifies this gap: classical estimators reach 73% on HyperShadow where a small learned model reaches 96.6%.

**(2) "4D" datasets in machine learning are space + time.** Existing 4D benchmarks — 4D world models, dynamic point-cloud suites such as MSR-Action3D, DeformingThings4D, HOI4D — all model 3D geometry evolving over time [4, 5]. To our knowledge no public dataset contains projections of objects with four or more *spatial* dimensions, despite the pedagogical ubiquity of the tesseract.

**(3) Humans can do a version of this task.** He, Bi & Zaidi [6] showed in virtual reality that human observers discriminate rigid from non-rigid motion of a hypercube's 3D projection at accuracy comparable to the 3D case, suggesting the perceptual system exploits geometric regularities that do not require four-dimensional experience. There is, however, no machine counterpart: no dataset, no baseline, no algorithmic detector. HyperShadow provides all three, and our rigidity witness (Section 5) is the explicit algorithmic analogue of the human judgement studied there.

**Contributions.**
1. **HyperShadow**, a reproducible benchmark of 10,800 static point clouds and 1,800 temporal sequences: native 3D shapes vs. 3D projections of eleven object families in R⁴–R⁶, under two projection models and four cumulative corruption tiers, with fairness rules that remove every shortcut we could identify (Section 3).
2. **A characterization of the failure of intrinsic-dimension estimation** on projected data, establishing that shadow detection is a distinct problem (Section 6.1).
3. **Baselines spanning three methodological families** — hand-crafted geometric features, persistent homology, and a compact learned point network — including leave-one-family-out generalization tests (Sections 4, 6).
4. **The rigidity witness**: a zero-parameter, closed-form statistic for temporal data that certifies incompatibility with rigid 3D motion, achieving AUROC 0.982 (Section 5). We propose *dimensional witnesses* — statistics whose value is provably bounded under any 3D explanation — as a general template, analogous in logical structure to Bell inequalities: one does not observe the hidden structure, one rules out the class of explanations that lack it.

**Scope.** All results hold under the stated simulation assumptions. Nothing in this paper is evidence for or against higher spatial dimensions in physical reality; the benchmark is an instrument for studying detectability, validated where ground truth is known.

---

## 2. Related Work

**Intrinsic-dimension estimation.** Estimators from Levina–Bickel MLE [1] to TwoNN [2] and their many successors, surveyed in [3], infer the dimension of the data manifold, typically benchmarked on manifolds embedded in *higher*-dimensional ambient spaces (scikit-dimension [7]). HyperShadow inverts the setting: the generating object has higher dimension than the observation space. Learned bottleneck approaches such as IDEA [8] estimate dimension with autoencoders; they too target the data manifold, not the generator.

**Topological data analysis.** Persistent homology summarizes multi-scale topology of point clouds [9], with recent work making H₁/H₂ computation robust in high ambient dimension [10] and enriching standard 3D benchmarks with topological features [11]. Projection alters topology (shadows self-intersect and fill voids), which motivates our TDA baseline.

**Point-cloud learning.** PointNet-style set networks [12] and their successors dominate 3D point classification on ModelNet40 and ScanObjectNN [13]. Our PointNet-lite is a deliberately small member of this family, sized to demonstrate that the task does not require large models or compute.

**Four spatial dimensions.** Computer-graphics work renders 4D scenes for visualization [14]; vision science shows humans judge 4D rigidity from 3D projections [6]. Existing "4D" ML datasets are dynamic 3D [4, 5]. None provide a benchmark of higher-spatial-dimension projections.

---

## 3. The HyperShadow Benchmark

Figure 1 (`figures/pipeline.png`) summarizes generation. All code is seeded NumPy; every sample is reproducible from a seed and a metadata record.

### 3.1 Object families

**Native 3D (label 0), 7 families** — sphere, solid ball, torus (area-uniform via rejection), cube surface, capped cylinder, random ellipsoids, and a tube around a trefoil knot (topologically non-trivial). Sampling is uniform with respect to surface area or volume.

**Higher-dimensional (label 1), 11 families** — unit hyperspheres S³ and S⁴; the solid 4-ball; tesseract and 5-cube boundaries (uniform over cells); the Clifford torus and duocylinder ridge in R⁴ (flat tori, uniform in angles); generalized hypertori in R⁴ and R⁵ (rejection-corrected); and random smooth compact manifolds embedded in R⁵/R⁶ by random trigonometric maps, providing unbounded family diversity.

### 3.2 Projection and corruption

Each N-D object receives a Haar-uniform rotation in SO(N) (QR decomposition of a Gaussian matrix with sign correction), then is projected to R³ **orthographically** (drop coordinates — the mathematical shadow) or **perspectively** (successive division through a virtual viewpoint along each extra axis — the 4D pinhole camera). Four cumulative corruption tiers follow: **0** clean; **1** Gaussian jitter (σ = 2% of cloud scale); **2** + removal of a random 20–40% half-space slab (self-occlusion); **3** + heavier jitter and distance-dependent dropout from a random viewpoint (LiDAR-like sparsity).

### 3.3 Fairness rules

Benchmarks of synthetic classes are vulnerable to shortcuts. We remove every one we identified:

- **Identical treatment**: native shapes receive the same Haar-random (3D) rotations and identical corruption tiers.
- **Normalization**: every cloud is centred, scaled to unit mean radius, and resampled to exactly 1,024 points — no information in position, scale, or cardinality.
- **Volume is not the answer**: the solid ball sits in the native class, so "fills a volume" cannot solve the task; conversely the Clifford torus shadow remains a surface, so "is a surface" fails too.
- **Class balance by tier**: tiers are distributed identically across classes.

### 3.4 Tracks

**Static**: 10,800 clouds of 1,024 points (600 per family), split 70/10/20 train/val/test.
**Temporal**: 1,800 sequences of 16 frames × 512 points. A single object rotates rigidly at constant angular velocity along a random one-parameter subgroup of SO(d) (matrix exponential of a random antisymmetric generator); each frame is projected and normalized independently, with point identity preserved across frames. For label 0 the rigid rotation is in R³; for label 1, in R^N.

---

## 4. Baselines

**Intrinsic dimension (0 parameters).** TwoNN [2] and Levina–Bickel MLE [1] computed per cloud; a single decision threshold (direction and value fit on half the data, evaluated on the other half).

**Geometric features + GBT.** 39 rotation-invariant features per cloud — the D2 pairwise-distance histogram, radial mass profile, PCA eigenvalue spectrum, k-NN distance-ratio statistics (a local dimension proxy), and nearest-neighbour density statistics — classified by gradient-boosted trees, 5-fold cross-validation.

**Persistent homology + GBT.** Ripser [9] persistence diagrams up to H₂ on 256-point subsamples; per-dimension lifetime statistics and top-5 H₁/H₂ lifetimes; same classifier protocol.

**PointNet-lite (Figure 2, `figures/architecture.png`).** A 190,914-parameter set network: shared per-point MLP (64–128–256 channels, batch norm), concatenated max- and mean-pooling, and a small dropout-regularized head. Training: AdamW, cosine schedule, 60 epochs, random-rotation + jitter augmentation; ~4 minutes on a GTX 1650 Ti (4 GB). The deliberate smallness is a claim: the signal is strong enough that no scale is needed.

---

## 5. The Rigidity Witness

Let X_t ∈ R^{n×3} be corresponding points across frames. For any rigid 3D motion there exist R ∈ SO(3), t, (and scale s, to absorb per-frame normalization) with X_{t+1} = s X_t R^T + t exactly; the optimal (s, R, t) is closed-form (Kabsch/Procrustes). Define the witness

> **w(X) = mean over t of RMS residual of the optimal rigid alignment X_t → X_{t+1}.**

For native sequences w is bounded by the noise floor. For the shadow of a rigid rotation in R^N, the apparent 3D motion composes visible rotation with a *w-dependent re-weighting of hidden coordinates*; no rigid 3D map explains it, and w stays bounded away from zero. Crucially, w is a **certificate**: a large value does not merely correlate with higher dimensionality — it *rules out* the entire class of rigid-3D explanations, up to noise. This is the logical structure of a Bell inequality, transplanted to kinematics.

Empirically (Figure 3, `figures/rigidity.png`): native sequences concentrate at w ≈ 0 (clean) and w ≈ 0.05 (the noise tier's floor); shadows concentrate at w ≈ 0.10 (4× the clean-motion residual). A single threshold fit on half the sequences achieves **held-out accuracy 0.982, AUROC 0.982** — with zero learned parameters. The hardest families (Clifford torus 0.92, tesseract 0.93) are those whose random rotation plane most often lies almost entirely within the visible subspace, making the shadow *nearly* rigid — a geometric, not statistical, limitation.

---

## 6. Results

Full per-tier results in Figure 4 (`figures/results.png`); per-shape breakdowns ship with the repository (`results/*.json`).

### 6.1 Static track

| Method | Params | Accuracy |
|---|---|---|
| Chance | — | 0.500 |
| TwoNN threshold | 0 | 0.710 |
| Levina–Bickel MLE threshold | 0 | 0.732 |
| Persistent homology + GBT | — | 0.904 |
| Geometric features + GBT | — | 0.956 |
| **PointNet-lite** | 190k | **0.966** |

Mean estimated intrinsic dimension is 2.5 (native) vs. 2.9 (projected) under TwoNN — a real but small gap, confounded exactly as predicted: the solid ball (native, ID 3) and projected 2-manifolds (ID 2) sit on the wrong sides of any threshold. **Dimensionality is the wrong observable; the projection map, not the manifold dimension, is what leaves evidence.**

Learned and feature-based methods degrade gracefully with corruption (PointNet: 0.984 clean → 0.939 sensor tier). The per-shape error profile is theoretically coherent: the residual confusions are S⁴-shadows vs. balls (0.54) and ball vs. everything (0.82) — pairs that differ *only* in radial density profile, since the shadow of a hypersphere *is* a ball with mass pushed toward the rim.

### 6.2 Generalization to unseen families

Training with entire families excluded, then testing detection on the excluded families only:

| Held-out family | Detection accuracy | 4D member | 5D member |
|---|---|---|---|
| Hypertori (R⁴ + R⁵) | 0.910 | 0.82 | 1.00 |
| Hypercubes (tesseract + 5-cube) | 0.795 | 0.64 | 0.95 |

Two consistent patterns: (i) detection transfers well above chance to never-seen generators, so the model learns projection signatures rather than a shape catalog; (ii) 5D objects are uniformly easier than 4D — each collapsed dimension compounds the density-fold signature. The tesseract is the hardest object in the benchmark: its flat cells project to polyhedron-like shadows that mimic native geometry.

### 6.3 Temporal track

The rigidity witness (Section 5) dominates: AUROC 0.982 with zero parameters, against 96.6% for the best learned static method. **Motion reveals dimensionality that still frames hide** — consistent with, and now quantifying, the human results of [6].

---

## 7. Limitations and Ethics

- **Simulation scope.** Results certify detectability under our generative assumptions (uniform sampling, two projection models, our noise models). Real sensors differ; Section 6.2 measures generalization across object families but not across projection physics.
- **The witness needs correspondence.** The Kabsch witness assumes tracked points across frames. Untracked variants (e.g., distributional rigidity via optimal transport) are open.
- **Coverage.** Eleven N-D families cannot exhaust "higher-dimensional objects"; the random-manifold family mitigates but does not eliminate this.
- **Interpretation discipline.** The benchmark will inevitably invite speculative application to anomalous real-world data. We state plainly: a positive detection on real data would establish only that the data is inconsistent with the rigid-3D model class considered, never the existence of extra dimensions. Every mundane explanation (tracking error, non-rigid objects, atmospherics, optics) dominates a priori and must be eliminated independently.

## 8. Conclusion

HyperShadow turns a century-old thought experiment — Plato's cave, Abbott's Flatland, one dimension up — into a measurable machine-learning problem with controlled ground truth. Shadows of higher-dimensional objects are detectably different from native 3D geometry; the difference is invisible to intrinsic-dimension estimation, learnable from single frames, and certifiable from motion by a closed-form witness. We release the dataset, models, and code, and propose the systematic study of dimensional witnesses — observable statistics bounded under low-dimensional explanations — as the path by which this line of work could eventually meet physical data.

---

## References

[1] E. Levina, P. Bickel. *Maximum likelihood estimation of intrinsic dimension.* NeurIPS 2005.
[2] E. Facco, M. d'Errico, A. Rodriguez, A. Laio. *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Scientific Reports, 2017.
[3] J. A. Lee et al. *A survey of dimension estimation methods.* arXiv:2507.13887, 2025.
[4] TesserAct: Learning 4D embodied world models. arXiv:2504.20995, 2025.
[5] OmniWorld: A multi-domain and multi-modal dataset for 4D world modeling. arXiv:2509.12201, 2025.
[6] Z. He, W. Bi, Q. Zaidi. *Perception of rigidity in three- and four-dimensional spaces.* 2023. (PMC10470465)
[7] J. Bac et al. *scikit-dimension: a Python package for intrinsic dimension estimation.* Entropy, 2021.
[8] Intrinsic Dimension Estimating Autoencoder (IDEA). arXiv:2509.10011, 2025.
[9] C. Tralie, N. Saul, R. Bar-On. *Ripser.py: A lean persistent homology library for Python.* JOSS, 2018.
[10] Persistent homology for high-dimensional data based on spectral methods. NeurIPS 2024. arXiv:2311.03087.
[11] Learning significant persistent homology features for 3D shape understanding. arXiv:2602.14228, 2026.
[12] C. Qi, H. Su, K. Mo, L. Guibas. *PointNet: Deep learning on point sets for 3D classification and segmentation.* CVPR 2017.
[13] M. Uy et al. *Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data.* ICCV 2019.
[14] Higher dimensional graphics: conceiving worlds in four spatial dimensions and beyond. arXiv:2103.14627, 2021.

---

## Appendix A. Reproducibility

```bash
python -m hypershadow.generate --out data --per-class 600 --n-points 1024 --seed 0
python -m hypershadow.generate --out data --temporal --per-class 100 --n-frames 16 --seed 1
python -m baselines.features      --data data/static.npz
python -m baselines.id_estimators --data data/static.npz
python -m baselines.tda           --data data/static.npz --max-samples 1200
python -m baselines.pointnet      --data data/static.npz --epochs 60
python -m baselines.pointnet      --data data/static.npz --epochs 40 --holdout-shapes hypertorus_4d hypertorus_5d
python -m baselines.rigidity      --data data/temporal.npz
```

All randomness flows from the two generation seeds and per-script seeds; 34 unit tests verify geometric correctness (Haar orthogonality, measure-uniform sampling, projection identities).

## Appendix B. Figures

| # | File | Content |
|---|---|---|
| 1 | `figures/pipeline.png` | Generation pipeline |
| 2 | `figures/architecture.png` | PointNet-lite architecture |
| 3 | `figures/rigidity.png` | Witness residual distributions |
| 4 | `figures/results.png` | Accuracy by corruption tier |
| 5 | `figures/gallery.png` | Shape gallery (native vs. projected) |
