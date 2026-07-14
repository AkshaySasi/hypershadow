"""PointNet-lite: learned baseline for the HyperShadow static track.

A compact PointNet (~0.9M params) sized for a 4 GB GPU. Reports overall,
per-tier, and per-shape accuracy on a held-out test split, and saves the
best checkpoint + a JSON results file.

Usage:
  python -m baselines.pointnet --data data/static.npz --epochs 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PointNetLite(nn.Module):
    """Shared per-point MLP -> max+mean pool -> classifier head."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, 3)
        f = self.point_mlp(x.transpose(1, 2))           # (B, 256, N)
        pooled = torch.cat([f.max(dim=2).values, f.mean(dim=2)], dim=1)
        return self.head(pooled)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random 3D rotation + slight jitter (labels are rotation-invariant)."""
    b = x.shape[0]
    a = torch.randn(b, 3, 3, device=x.device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=1, dim2=2)).unsqueeze(1)
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det.sign().unsqueeze(1)
    x = torch.bmm(x, q.transpose(1, 2))
    return x + torch.randn_like(x) * 0.005


def split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = n // 5
    n_val = n // 10
    return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str,
             batch: int = 128) -> tuple[float, np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(x[i:i + batch].to(device))
            preds.append(logits.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    return float((preds == y.numpy()).mean()), preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/static.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-shapes", nargs="*", default=None,
                    help="shape names excluded from training and used as the "
                         "entire test set (leave-one-family-out generalization)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    blob = np.load(args.data)
    x = torch.from_numpy(blob["points"]).float()
    y = torch.from_numpy(blob["labels"]).long()
    meta = json.loads((args.data.parent / (args.data.stem + "_meta.json")).read_text())["samples"]

    if args.holdout_shapes:
        shapes_all = np.array([m["shape"] for m in meta])
        held = np.isin(shapes_all, args.holdout_shapes)
        if not held.any():
            raise SystemExit(f"no samples match holdout shapes {args.holdout_shapes}")
        te = np.flatnonzero(held)
        rest = np.flatnonzero(~held)
        rng = np.random.default_rng(args.seed)
        rest = rng.permutation(rest)
        n_val = len(rest) // 10
        va, tr = rest[:n_val], rest[n_val:]
        print(f"holdout shapes: {args.holdout_shapes}")
    else:
        tr, va, te = split_indices(len(y), args.seed)
    model = PointNetLite().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  train/val/test: {len(tr)}/{len(va)}/{len(te)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_va, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total_loss = 0.0
        for i in range(0, len(tr), args.batch):
            idx = tr[perm[i:i + args.batch].numpy()]
            xb = augment(x[idx].to(device))
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()
        va_acc, _ = evaluate(model, x[va], y[va], device)
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {total_loss / len(tr):.4f}  val acc {va_acc:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    model.load_state_dict(best_state)
    te_acc, te_pred = evaluate(model, x[te], y[te], device)
    print(f"\nTEST accuracy: {te_acc:.4f}  (best val {best_va:.4f})")

    tiers = np.array([meta[i]["tier"] for i in te])
    shapes = np.array([meta[i]["shape"] for i in te])
    y_te = y[te].numpy()
    per_tier = {int(t): float((te_pred[tiers == t] == y_te[tiers == t]).mean())
                for t in sorted(set(tiers))}
    per_shape = {s: float((te_pred[shapes == s] == y_te[shapes == s]).mean())
                 for s in sorted(set(shapes))}
    for t, a in per_tier.items():
        print(f"  tier {t}: {a:.4f}")
    print("hardest shapes:", sorted(per_shape.items(), key=lambda kv: kv[1])[:5])

    args.out.mkdir(parents=True, exist_ok=True)
    tag = ("pointnet_holdout_" + "_".join(args.holdout_shapes)) if args.holdout_shapes else "pointnet"
    torch.save(best_state, args.out / f"{tag}_lite.pt")
    (args.out / f"{tag}_results.json").write_text(json.dumps({
        "holdout_shapes": args.holdout_shapes,
        "model": "pointnet_lite", "params": n_params, "device": device,
        "test_accuracy": te_acc, "val_accuracy": best_va,
        "per_tier": per_tier, "per_shape": per_shape,
        "epochs": args.epochs, "seed": args.seed, "n_train": len(tr), "n_test": len(te),
    }, indent=2))
    print(f"saved {args.out / (tag + '_lite.pt')} and results json")


if __name__ == "__main__":
    main()
