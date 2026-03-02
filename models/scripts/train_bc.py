#!/usr/bin/env python3
# ------------------------------------------------------------
# Behavior Cloning Training Script
#
# How to run (inside container, from repo root):
#
#   cd /workspaces/ur5e-checkers-irl
#   python3 scripts/train_bc.py \
#       --dataset_dir datasets \
#       --out models/bc_policy.pt \
#       --epochs 50
#
# Make sure you have collected demonstration episodes first
# and that they exist in the datasets/ folder as episode_*.npz
# ------------------------------------------------------------

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_npz_files(dataset_dir: str):
    paths = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz found in {dataset_dir}")

    obs_list, act_list = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        obs = d["observations"].astype(np.float32)
        act = d["actions"].astype(np.float32)

        if obs.ndim != 2 or act.ndim != 2:
            raise ValueError(f"{p}: expected 2D arrays, got obs {obs.shape}, act {act.shape}")
        if obs.shape[0] != act.shape[0]:
            raise ValueError(f"{p}: obs/actions length mismatch {obs.shape[0]} vs {act.shape[0]}")

        obs_list.append(obs)
        act_list.append(act)

    X = np.concatenate(obs_list, axis=0)
    Y = np.concatenate(act_list, axis=0)
    return X, Y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets")
    ap.add_argument("--out", default="models/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, Y, paths = load_npz_files(args.dataset_dir)
    in_dim = X.shape[1]
    out_dim = Y.shape[1]

    obs_mean = X.mean(axis=0)
    obs_std = X.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0
    Xn = (X - obs_mean) / obs_std

    N = Xn.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = int(N * args.val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = torch.from_numpy(Xn[tr_idx]).float()
    Ytr = torch.from_numpy(Y[tr_idx]).float()
    Xva = torch.from_numpy(Xn[val_idx]).float()
    Yva = torch.from_numpy(Y[val_idx]).float()

    tr_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loaded {len(paths)} episodes")
    print(f"Samples: {N} | obs_dim: {in_dim} | act_dim: {out_dim} | device: {device}")

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.shape[0]
        tr_loss /= max(len(tr_idx), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * xb.shape[0]
        va_loss /= max(len(val_idx), 1)

        print(f"epoch {ep:03d} | train_mse {tr_loss:.6f} | val_mse {va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "obs_mean": obs_mean.astype(np.float32),
                    "obs_std": obs_std.astype(np.float32),
                },
                args.out,
            )

    print(f"Saved best model to: {args.out} (best val_mse={best_val:.6f})")


if __name__ == "__main__":
    main()