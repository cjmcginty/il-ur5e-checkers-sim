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


_FIXED_ACTIONS: List[ActionKey] = _generate_simple_and_single_jump_actions()
_ACTION_TO_INDEX: Dict[ActionKey, int] = {
    action_key: i for i, action_key in enumerate(_FIXED_ACTIONS)
}


def num_actions() -> int:
    return len(_FIXED_ACTIONS)


def action_key_to_index(action_key: ActionKey) -> int:
    try:
        return _ACTION_TO_INDEX[action_key]
    except KeyError as exc:
        raise ValueError(f"Action key not in fixed action space: {action_key}") from exc


def index_to_action_key(index: int) -> ActionKey:
    if index < 0 or index >= len(_FIXED_ACTIONS):
        raise IndexError(f"Action index out of range: {index}")
    return _FIXED_ACTIONS[index]


def move_to_action_key(move: MoveLike) -> ActionKey:
    """
    Convert a legal move object from board.py into a fixed DQN action key.

    For now, only supports 2-point moves:
    - normal moves
    - single captures

    Full multi-jump capture sequences will need an expanded action space later.
    """
    return move_to_key(move)


def move_to_action_index(move: MoveLike) -> int:
    action_key = move_to_action_key(move)
    if len(action_key) != 2:
        raise ValueError(
            "Fixed action space only supports 2-point moves; "
            f"got path of length {len(action_key)}: {action_key}"
        )
    return action_key_to_index(action_key)


def legal_action_keys(board: CheckersBoard) -> List[ActionKey]:
    """
    Return the currently legal action keys for this board.

    For now, this only works when every legal move is a 2-point action.
    If the position contains multi-jump legal moves, this function raises.
    """
    keys = list(legal_move_keys(board))
    for key in keys:
        if len(key) != 2:
            raise ValueError(
                "Encountered legal move outside fixed action space; "
                f"multi-step path: {key}"
            )
    return keys


def legal_action_indices(board: CheckersBoard) -> List[int]:
    return [action_key_to_index(key) for key in legal_action_keys(board)]


def debug_print_action_space_summary() -> None:
    print(f"Fixed action space size: {num_actions()}")