#!/usr/bin/env python3
"""
Offline behavior cloning trainer.

Reads demonstration episodes saved by data_collection_node.py and trains
an MLP that matches the checkpoint format expected by bc_policy_node.py.

Notes:
- Uses PyTorch Dataset/DataLoader and state_dict-style checkpointing,
  following the official PyTorch tutorials.
- This trainer learns observation -> action using supervised regression.
"""

import argparse
import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


@dataclass
class DemoArrays:
    observations: np.ndarray
    actions: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray


class BehaviorCloningDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx])
        act = torch.from_numpy(self.actions[idx])
        return obs, act


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_demo_files(dataset_dir: str) -> list[str]:
    pattern = os.path.join(dataset_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .npz files found in dataset_dir={dataset_dir}")
    return files


def load_demos(files: list[str], max_files: int | None = None) -> DemoArrays:
    if max_files is not None:
        files = files[:max_files]

    obs_list = []
    act_list = []

    for path in files:
        data = np.load(path, allow_pickle=True)

        if "observations" not in data or "actions" not in data:
            print(f"[WARN] Skipping {path}: missing observations/actions")
            continue

        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)

        if observations.ndim != 2 or actions.ndim != 2:
            print(f"[WARN] Skipping {path}: observations/actions are not 2D")
            continue

        n = min(len(observations), len(actions))
        if n == 0:
            print(f"[WARN] Skipping {path}: empty episode")
            continue

        if len(observations) != len(actions):
            print(
                f"[WARN] {path}: obs/actions length mismatch "
                f"({len(observations)} vs {len(actions)}), truncating to {n}"
            )

        obs_list.append(observations[:n])
        act_list.append(actions[:n])

    if not obs_list:
        raise RuntimeError("No usable demonstrations found after loading all files.")

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)

    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = obs.std(axis=0).astype(np.float32)
    obs_std[obs_std < 1e-6] = 1.0

    obs_norm = ((obs - obs_mean) / obs_std).astype(np.float32)

    return DemoArrays(
        observations=obs_norm,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(round(n * val_ratio))
    n_val = max(1, n_val) if n > 1 and val_ratio > 0.0 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if len(train_idx) == 0:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    return train_idx, val_idx


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> float:
    if len(loader.dataset) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = loss_fn(pred, act)

            batch_size = obs.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="datasets")
    parser.add_argument("--out", default="models/bc_policy.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    files = find_demo_files(args.dataset_dir)
    print(f"Found {len(files)} dataset files in {args.dataset_dir}")

    demos = load_demos(files, max_files=args.max_files)

    num_samples = demos.observations.shape[0]
    in_dim = demos.observations.shape[1]
    out_dim = demos.actions.shape[1]

    print(f"Loaded {num_samples} samples")
    print(f"in_dim={in_dim}, out_dim={out_dim}")

    train_idx, val_idx = split_indices(num_samples, args.val_ratio, args.seed)

    train_obs = demos.observations[train_idx]
    train_act = demos.actions[train_idx]
    val_obs = demos.observations[val_idx] if len(val_idx) > 0 else np.zeros((0, in_dim), dtype=np.float32)
    val_act = demos.actions[val_idx] if len(val_idx) > 0 else np.zeros((0, out_dim), dtype=np.float32)

    train_ds = BehaviorCloningDataset(train_obs, train_act)
    val_ds = BehaviorCloningDataset(val_obs, val_act)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden, depth=args.depth).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = loss_fn(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = obs.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device, loss_fn)

        if np.isnan(val_loss):
            improved = best_state is None
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "in_dim": int(in_dim),
                "out_dim": int(out_dim),
                "hidden": int(args.hidden),
                "depth": int(args.depth),
                "obs_mean": demos.obs_mean.astype(np.float32),
                "obs_std": demos.obs_std.astype(np.float32),
            }

        if np.isnan(val_loss):
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training ended without producing a checkpoint.")

    torch.save(best_state, args.out)
    print(f"Saved checkpoint to {args.out}")

    # Nice sanity-print for your ROS node expectations
    print("Checkpoint contents:")
    print(f"  in_dim   = {best_state['in_dim']}")
    print(f"  out_dim  = {best_state['out_dim']}")
    print(f"  hidden   = {best_state['hidden']}")
    print(f"  depth    = {best_state['depth']}")
    print(f"  obs_mean = shape {best_state['obs_mean'].shape}")
    print(f"  obs_std  = shape {best_state['obs_std'].shape}")


if __name__ == "__main__":
    main()