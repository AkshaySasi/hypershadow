"""Split-seed variance for the intrinsic-dimension threshold baselines."""

import json

import numpy as np

from baselines.id_estimators import levina_bickel, twonn


def main() -> None:
    blob = np.load("data/static.npz")
    x, y = blob["points"].astype(np.float64), blob["labels"]
    rng0 = np.random.default_rng(0)
    idx = rng0.choice(len(y), 4000, replace=False)
    x, y = x[idx], y[idx]
    stats = {
        "twonn": np.array([twonn(c) for c in x]),
        "levina_bickel": np.array([levina_bickel(c) for c in x]),
    }
    del x

    out = {}
    half = len(y) // 2
    for name, stat in stats.items():
        accs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(y))
            fit, hold = order[:half], order[half:]
            cands = np.quantile(stat[fit], np.linspace(0, 1, 201))
            best_t, best_dir, best_acc = None, 1, 0.0
            for t in cands:
                for d in (1, -1):
                    a = ((d * stat[fit] > d * t) == y[fit]).mean()
                    if a > best_acc:
                        best_t, best_dir, best_acc = t, d, a
            accs.append(float(((best_dir * stat[hold] > best_dir * best_t) == y[hold]).mean()))
        out[name] = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(f"{name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    json.dump(out, open("results/id_seeds.json", "w"), indent=2)


if __name__ == "__main__":
    main()
