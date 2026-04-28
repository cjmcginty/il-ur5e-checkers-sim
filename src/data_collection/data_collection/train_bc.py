#!/usr/bin/env python3
"""
train_bc.py

Behavior cloning trainer for datasets saved by data_collection_node.py.

Expected .npz contents per episode:
    observations: [N, obs_dim]
    actions:      [N, act_dim]
    timestamps:   [N]
    joint_names:  [act_dim] (optional/useful for inspection)

Saves checkpoint format compatible with bc_policy_node.py:
    {
        "in_dim": int,
        "out_dim": int,
        "hidden": int,
        "depth": int,
        "model_state": ...,
        "obs_mean": np.ndarray,
        "obs_std": np.ndarray,
    }
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def load_episode(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one .npz episode and return observations/actions."""
    data = np.load(path, allow_pickle=True)

    if "observations" not in data or "actions" not in data:
        raise ValueError(f"{path} is missing 'observations' or 'actions'")

    obs = data["observations"].astype(np.float32)
    acts = data["actions"].astype(np.float32)

    if obs.ndim != 2:
        raise ValueError(f"{path}: observations must be 2D, got shape {obs.shape}")
    if acts.ndim != 2:
        raise ValueError(f"{path}: actions must be 2D, got shape {acts.shape}")
    if len(obs) != len(acts):
        raise ValueError(
            f"{path}: observations/actions length mismatch: {len(obs)} vs {len(acts)}"
        )

    return obs, acts


def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all episode_*.npz files and concatenate them."""
    pattern = os.path.join(dataset_dir, "episode_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No episode files found in {dataset_dir}")

    obs_list = []
    act_list = []

    obs_dim = None
    act_dim = None

    for path in files:
        obs, acts = load_episode(path)

        if obs_dim is None:
            obs_dim = obs.shape[1]
            act_dim = acts.shape[1]
        else:
            if obs.shape[1] != obs_dim:
                raise ValueError(
                    f"{path}: obs dim mismatch. Expected {obs_dim}, got {obs.shape[1]}"
                )
            if acts.shape[1] != act_dim:
                raise ValueError(
                    f"{path}: act dim mismatch. Expected {act_dim}, got {acts.shape[1]}"
                )

        obs_list.append(obs)
        act_list.append(acts)

    X = np.concatenate(obs_list, axis=0)
    Y = np.concatenate(act_list, axis=0)

    return X, Y, files


def make_splits(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/val split over samples."""
    assert len(X) == len(Y)
    n = len(X)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Guard against pathological tiny datasets
    if len(train_idx) == 0:
        train_idx = val_idx
        val_idx = np.array([], dtype=np.int64)

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx] if len(val_idx) > 0 else np.empty((0, X.shape[1]), dtype=np.float32)
    Y_val = Y[val_idx] if len(val_idx) > 0 else np.empty((0, Y.shape[1]), dtype=np.float32)

    return X_train, Y_train, X_val, Y_val


def normalize_train_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalization stats from training set only."""
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    """Compute mean loss over a loader."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a behavior cloning MLP.")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--model_out", type=str, default="models/bc_policy.pt")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load dataset
    X, Y, files = load_dataset(args.dataset_dir)
    print(f"Loaded {len(files)} episode files")
    print(f"Total samples: {len(X)}")
    print(f"Observation dim: {X.shape[1]}")
    print(f"Action dim: {Y.shape[1]}")

    # Split
    X_train, Y_train, X_val, Y_val = make_splits(X, Y, args.val_ratio, args.seed)
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")

    # Normalize observations using training stats only
    obs_mean, obs_std = normalize_train_stats(X_train)
    X_train_n = ((X_train - obs_mean) / obs_std).astype(np.float32)
    X_val_n = ((X_val - obs_mean) / obs_std).astype(np.float32)

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(Y_train.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    if len(X_val_n) > 0:
        val_ds = TensorDataset(
            torch.from_numpy(X_val_n),
            torch.from_numpy(Y_val.astype(np.float32)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        val_loader = None

    # Model
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, depth=args.depth).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        train_loss = total_train_loss / max(total_train_count, 1)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f}"
            )
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    checkpoint = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden": args.hidden,
        "depth": args.depth,
        "model_state": best_state,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
    }

    torch.save(checkpoint, args.model_out)
    print(f"Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()