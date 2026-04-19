from typing import Dict, List, Set, Tuple

from ur5e_checkers_bringup.board import CheckersBoard, MoveLike
from ur5e_checkers_bringup.dqn_utils import legal_move_keys, move_to_key

Coord = Tuple[int, int]
ActionKey = Tuple[Coord, ...]


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _playable_square(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def _generate_all_action_keys() -> List[ActionKey]:
    """
    Generate a fixed action space containing all geometrically valid:
    - 1-step diagonal moves
    - 1-jump captures
    - multi-jump capture paths

    Each action is represented as a path-like tuple:
        ((r0, c0), (r1, c1))
        ((r0, c0), (r1, c1), (r2, c2))
        ...

    This is a geometry-only superset of legal moves. The board logic still decides
    which of these actions are legal in a given position.
    """
    actions: Set[ActionKey] = set()

    step_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    jump_deltas = [(-2, -2), (-2, 2), (2, -2), (2, 2)]

    def extend_jump_paths(path: ActionKey) -> None:
        r, c = path[-1]

        for dr, dc in jump_deltas:
            r2 = r + dr
            c2 = c + dc
            landing = (r2, c2)

            if not (_on_board(r2, c2) and _playable_square(r2, c2)):
                continue

            # Keep the fixed action space finite and avoid path loops.
            if landing in path:
                continue

            next_path = path + (landing,)
            actions.add(next_path)
            extend_jump_paths(next_path)

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
                    jump_path = (start, (r2, c2))
                    actions.add(jump_path)
                    extend_jump_paths(jump_path)

    return sorted(actions)


_FIXED_ACTIONS: List[ActionKey] = _generate_all_action_keys()
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

    Supports both single-step moves and multi-step capture sequences.
    """
    return move_to_key(move)


def move_to_action_index(move: MoveLike) -> int:
    action_key = move_to_action_key(move)
    return action_key_to_index(action_key)


def legal_action_keys(board: CheckersBoard) -> List[ActionKey]:
    """
    Return the currently legal action keys for this board.
    """
    return list(legal_move_keys(board))


def legal_action_indices(board: CheckersBoard) -> List[int]:
    return [action_key_to_index(key) for key in legal_action_keys(board)]


def debug_print_action_space_summary() -> None:
    max_path_len = max(len(action) for action in _FIXED_ACTIONS)
    print(f"Fixed action space size: {num_actions()} | max path length: {max_path_len}")