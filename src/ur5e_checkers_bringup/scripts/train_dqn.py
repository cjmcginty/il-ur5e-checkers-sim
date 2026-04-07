import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ur5e_checkers_bringup.board import CheckersBoard
from ur5e_checkers_bringup.dqn_action_space import (
    index_to_action_key,
    legal_action_indices,
    move_to_action_index,
    num_actions,
)
from ur5e_checkers_bringup.dqn_model import DQN
from ur5e_checkers_bringup.dqn_utils import (
    encode_board,
    epsilon_greedy_index,
    key_to_move,
    reward_for_move,
)


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def clone_board(board: CheckersBoard) -> CheckersBoard:
    return copy.deepcopy(board)


def get_winner(board: CheckersBoard) -> Optional[str]:
    """
    Returns:
        'r' if red wins
        'b' if black wins
        None otherwise

    This assumes board.py has either:
      - winner()
      - get_winner()

    If neither exists, it falls back to:
      if no legal moves remain, opponent wins
    """
    if hasattr(board, "winner") and callable(board.winner):
        winner = board.winner()
        if winner in ("r", "b"):
            return winner
        if winner == "red":
            return "r"
        if winner == "black":
            return "b"
        return None

    if hasattr(board, "get_winner") and callable(board.get_winner):
        winner = board.get_winner()
        if winner in ("r", "b"):
            return winner
        if winner == "red":
            return "r"
        if winner == "black":
            return "b"
        return None

    legal = board.legal_moves()
    if len(legal) == 0:
        return "b" if board.turn == "r" else "r"

    return None


def apply_move(board: CheckersBoard, action_index: int) -> CheckersBoard:
    """
    Applies the chosen action to a copy of the board and returns the new board.

    Assumes board.py supports one of:
      - board.push(move)
      - board.apply_move(move)
      - board.make_move(move)
    """
    next_board = clone_board(board)

    action_key = index_to_action_key(action_index)
    move = key_to_move(action_key)

    if hasattr(next_board, "push") and callable(next_board.push):
        next_board.push(move)
        return next_board

    if hasattr(next_board, "apply_move") and callable(next_board.apply_move):
        next_board.apply_move(move)
        return next_board

    if hasattr(next_board, "make_move") and callable(next_board.make_move):
        next_board.make_move(move)
        return next_board

    raise AttributeError(
        "CheckersBoard is missing a supported move-application method "
        "(expected push, apply_move, or make_move)."
    )


def select_opponent_action(board: CheckersBoard) -> int:
    """
    Very simple opponent: random legal move.
    """
    legal = legal_action_indices(board)
    return random.choice(legal)


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> Optional[float]:
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)

    states = torch.stack([t.state for t in batch]).to(device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.stack([t.next_state for t in batch]).to(device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    q_values = policy_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1).values
        target_values = rewards + gamma * max_next_q * (1.0 - dones)

    loss_fn = nn.MSELoss()
    loss = loss_fn(state_action_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_dqn(
    episodes: int = 500,
    replay_capacity: int = 10000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    target_update_every: int = 20,
    max_steps_per_episode: int = 200,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    save_path: str = "models/dqn_checkers.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = epsilon_start
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Num actions: {num_actions()}")

    for episode in range(1, episodes + 1):
        board = CheckersBoard()
        episode_reward = 0.0
        losses: List[float] = []

        for step in range(max_steps_per_episode):
            state = encode_board(board)

            try:
                legal_indices = legal_action_indices(board)
            except ValueError as e:
                print(f"[episode {episode}] stopped: {e}")
                break

            q_values = policy_net(state.to(device)).squeeze(0).detach().cpu()
            action_index = epsilon_greedy_index(q_values, legal_indices, epsilon)

            board_before = clone_board(board)
            board = apply_move(board, action_index)
            winner = get_winner(board)

            reward = reward_for_move(
                board_before=board_before,
                board_after=board,
                player=board_before.turn,
                winner=winner,
            )

            done = winner is not None
            next_state = encode_board(board)

            replay_buffer.push(
                Transition(
                    state=state,
                    action=action_index,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )
            if loss is not None:
                losses.append(loss)

            episode_reward += reward

            if done:
                break

            # opponent turn
            try:
                opponent_legal = legal_action_indices(board)
            except ValueError as e:
                print(f"[episode {episode}] opponent stopped: {e}")
                break

            opponent_action = select_opponent_action(board)
            board_before_opp = clone_board(board)
            board = apply_move(board, opponent_action)
            winner = get_winner(board)

            opponent_player = board_before_opp.turn
            opponent_reward = reward_for_move(
                board_before=board_before_opp,
                board_after=board,
                player=opponent_player,
                winner=winner,
            )

            done = winner is not None
            next_state_after_opp = encode_board(board)

            replay_buffer.push(
                Transition(
                    state=next_state,
                    action=opponent_action,
                    reward=opponent_reward,
                    next_state=next_state_after_opp,
                    done=done,
                )
            )

            loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )
            if loss is not None:
                losses.append(loss)

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(
            f"Episode {episode:04d} | "
            f"reward={episode_reward:.3f} | "
            f"epsilon={epsilon:.3f} | "
            f"buffer={len(replay_buffer)} | "
            f"avg_loss={avg_loss:.6f}"
        )

        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), save_path_obj)
            print(f"Saved checkpoint to {save_path_obj}")

    torch.save(policy_net.state_dict(), save_path_obj)
    print(f"Training complete. Final model saved to {save_path_obj}")


if __name__ == "__main__":
    train_dqn()