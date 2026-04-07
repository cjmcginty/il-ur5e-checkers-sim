import torch
import torch.nn as nn

from ur5e_checkers_bringup.dqn_action_space import num_actions


class DQN(nn.Module):
    """
    Simple DQN for checkers.

    Input:
        board tensor of shape [B, 5, 8, 8]

    Output:
        Q-values of shape [B, num_actions()]
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[1:] != (5, 8, 8):
            raise ValueError(
                f"Expected input shape [B, 5, 8, 8], got {tuple(x.shape)}"
            )

        x = self.conv(x)
        x = self.head(x)
        return x