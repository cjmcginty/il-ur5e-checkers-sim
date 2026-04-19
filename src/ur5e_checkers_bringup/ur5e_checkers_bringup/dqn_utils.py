import torch

from typing import Iterable, List, Tuple

from ur5e_checkers_bringup.board import CheckersBoard, Move, MoveLike, MoveSequence

def encode_board(board: CheckersBoard) -> torch.Tensor:
    """
    Encode the board as a 5x8x8 float tensor.

    Channels:
    0: red men     ("r")
    1: red kings   ("R")
    2: black men   ("b")
    3: black kings ("B")
    4: turn plane  (all 1.0 if red to move, all 0.0 if black to move)

    This exactly matches the board.py piece representation.
    """
    state = torch.zeros((5, 8, 8), dtype=torch.float32)

    for r in range(8):
        for c in range(8):
            piece = board.board[r][c]

            if piece == "r":
                state[0, r, c] = 1.0
            elif piece == "R":
                state[1, r, c] = 1.0
            elif piece == "b":
                state[2, r, c] = 1.0
            elif piece == "B":
                state[3, r, c] = 1.0

    if board.turn == "r":
        state[4, :, :] = 1.0

    return state


def move_to_key(move: MoveLike) -> tuple:
    """
    Convert a Move or MoveSequence into a hashable tuple key.

    Examples:
        Move((5, 0), (4, 1)) -> ((5, 0), (4, 1))
        MoveSequence(((5, 0), (3, 2), (1, 4))) -> ((5, 0), (3, 2), (1, 4))
    """
    if isinstance(move, MoveSequence):
        return tuple(move.path)

    if isinstance(move, Move):
        return (move.bgn, move.dst)

    raise TypeError(f"Unsupported move type: {type(move)}")


def key_to_move(key: tuple) -> MoveLike:
    """
    Convert a tuple path key back into Move or MoveSequence.
    """
    if len(key) == 2:
        return Move(key[0], key[1])

    if len(key) >= 2:
        return MoveSequence(tuple(key))

    raise ValueError(f"Invalid move key: {key}")


def legal_move_keys(board: CheckersBoard) -> List[tuple]:
    """
    Return the current legal moves as tuple keys.
    """
    return [move_to_key(move) for move in board.legal_moves()]


def reward_for_move(
    board_before: CheckersBoard,
    board_after: CheckersBoard,
    player: str,
    winner: str | None = None,
    step_penalty: float = -0.01,
    capture_reward: float = 0.20,
    king_reward: float = 0.10,
    win_reward: float = 1.00,
    loss_reward: float = -1.00,
) -> float:
    """
    Compute a simple DQN reward from a transition.

    Assumptions:
    - player is the player who took the action ("r" or "b")
    - board_before is the state before that move
    - board_after is the state after that move
    - winner is usually board_after.winner()

    Reward terms:
    - small step penalty every move
    - reward for capturing opponent pieces
    - reward for increasing own king count
    - terminal win/loss reward
    """
    reward = step_penalty

    if player == "r":
        captured_delta = board_after.black_captured - board_before.black_captured
        own_king_before = count_pieces(board_before, "R")
        own_king_after = count_pieces(board_after, "R")
    elif player == "b":
        captured_delta = board_after.red_captured - board_before.red_captured
        own_king_before = count_pieces(board_before, "B")
        own_king_after = count_pieces(board_after, "B")
    else:
        raise ValueError(f"Invalid player: {player}")

    if captured_delta > 0:
        reward += capture_reward * captured_delta

    king_delta = own_king_after - own_king_before
    if king_delta > 0:
        reward += king_reward * king_delta

    if winner is not None:
        if winner == player:
            reward += win_reward
        else:
            reward += loss_reward

    return reward


def count_pieces(board: CheckersBoard, piece_symbol: str) -> int:
    count = 0
    for r in range(8):
        for c in range(8):
            if board.board[r][c] == piece_symbol:
                count += 1
    return count


def epsilon_greedy_index(
    q_values: torch.Tensor,
    legal_indices: Iterable[int],
    epsilon: float,
) -> int:
    """
    Pick an action index using epsilon-greedy over legal actions only.

    q_values: shape [num_actions]
    legal_indices: iterable of valid action indices for the current board
    """
    legal_indices = list(legal_indices)

    if not legal_indices:
        raise ValueError("No legal indices provided")

    if torch.rand(1).item() < epsilon:
        rand_i = torch.randint(0, len(legal_indices), (1,)).item()
        return legal_indices[rand_i]

    masked_q = torch.full_like(q_values, -1e9)
    masked_q[legal_indices] = q_values[legal_indices]
    return int(torch.argmax(masked_q).item())

def legal_moves(board: CheckersBoard) -> List[MoveLike]:
    """
    Return actual legal move objects from the board.
    """
    return board.legal_moves()


def legal_moves_with_keys(board: CheckersBoard) -> List[Tuple[MoveLike, tuple]]:
    """
    Returns:
        [(move, key), ...]
    """
    moves = board.legal_moves()
    return [(m, move_to_key(m)) for m in moves]


def legal_moves_with_indices(board: CheckersBoard) -> List[Tuple[MoveLike, tuple, int]]:
    """
    Returns:
        [(move, key, index), ...]

    This is the main hybrid helper.
    """
    from ur5e_checkers_bringup.dqn_action_space import action_key_to_index

    result = []
    for move in board.legal_moves():
        key = move_to_key(move)
        idx = action_key_to_index(key)
        result.append((move, key, idx))
    return result