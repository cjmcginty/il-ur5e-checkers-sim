from typing import Dict, List, Tuple

from ur5e_checkers_bringup.board import CheckersBoard, MoveLike
from ur5e_checkers_bringup.dqn_utils import legal_move_keys, move_to_key

Coord = Tuple[int, int]
ActionKey = Tuple[Coord, ...]


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _playable_square(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def _generate_simple_and_single_jump_actions() -> List[ActionKey]:
    """
    Generate a fixed action space containing all:
    - 1-step diagonal moves
    - 1-jump diagonal captures

    Each action is represented as a path-like tuple:
        ((r0, c0), (r1, c1))

    This does NOT enumerate full multi-jump sequences.
    Those will be handled by board legal-move logic later if needed.
    """
    actions = set()

    step_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    jump_deltas = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

    for r in range(8):
        for c in range(8):
            if not _playable_square(r, c):
                continue

            start = (r, c)

            for dr, dc in step_deltas:
                r1 = r + dr
                c1 = c + dc
                if _on_board(r1, c1) and _playable_square(r1, c1):
                    actions.add((start, (r1, c1)))

            for dr, dc in jump_deltas:
                r2 = r + dr
                c2 = c + dc
                if _on_board(r2, c2) and _playable_square(r2, c2):
                    actions.add((start, (r2, c2)))

    return sorted(actions)


ACTION_KEYS: List[ActionKey] = _generate_simple_and_single_jump_actions()
ACTION_TO_INDEX: Dict[ActionKey, int] = {
    action: idx for idx, action in enumerate(ACTION_KEYS)
}
INDEX_TO_ACTION: Dict[int, ActionKey] = {
    idx: action for idx, action in enumerate(ACTION_KEYS)
}


def num_actions() -> int:
    return len(ACTION_KEYS)


def action_key_to_index(action_key: ActionKey) -> int:
    if action_key not in ACTION_TO_INDEX:
        raise KeyError(f"Action key not in fixed action space: {action_key}")
    return ACTION_TO_INDEX[action_key]


def index_to_action_key(index: int) -> ActionKey:
    if index not in INDEX_TO_ACTION:
        raise KeyError(f"Invalid action index: {index}")
    return INDEX_TO_ACTION[index]


def move_to_action_index(move: MoveLike) -> int:
    """
    Convert a legal move object from board.py into a fixed DQN action index.

    For now, only supports 2-point moves:
    - normal moves
    - single captures

    Full multi-jump capture sequences will need an expanded action space later.
    """
    key = move_to_key(move)

    if len(key) != 2:
        raise ValueError(
            f"Multi-step move not supported by current action space: {key}"
        )

    return action_key_to_index(key)


def legal_action_indices(board: CheckersBoard) -> List[int]:
    """
    Return the currently legal action indices for this board.

    For now, this only works when every legal move is a 2-point action.
    If the position contains multi-jump legal moves, this function raises.
    """
    indices: List[int] = []

    for key in legal_move_keys(board):
        if len(key) != 2:
            raise ValueError(
                f"Encountered multi-step legal move not supported yet: {key}"
            )
        indices.append(action_key_to_index(key))

    return indices


def debug_print_action_space_summary() -> None:
    print(f"Total fixed actions: {len(ACTION_KEYS)}")

    normal_count = 0
    jump_count = 0

    for action in ACTION_KEYS:
        (r0, c0), (r1, c1) = action
        if abs(r1 - r0) == 1:
            normal_count += 1
        elif abs(r1 - r0) == 2:
            jump_count += 1

    print(f"Normal diagonal moves: {normal_count}")
    print(f"Single-jump captures: {jump_count}")