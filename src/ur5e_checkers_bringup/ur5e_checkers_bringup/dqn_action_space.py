from typing import Dict, List, Tuple

from ur5e_checkers_bringup.board import CheckersBoard
from ur5e_checkers_bringup.dqn_utils import move_to_key

Coord = Tuple[int, int]
ActionKey = Tuple[Coord, ...]


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _playable_square(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def _jumped_square(a: Coord, b: Coord) -> Coord:
    """Return the square jumped over between a -> b"""
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _generate_all_action_keys(max_depth: int = 10) -> List[ActionKey]:
    """
    Generate a fixed action space containing:
    - simple diagonal moves
    - all multi-jump capture paths

    IMPORTANT:
    - We DO NOT forbid revisiting landing squares (kings can legally do this)
    - We DO forbid reusing the SAME jumped-over square (prevents infinite loops)
    """

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    actions = set()

    for r in range(8):
        for c in range(8):
            if not _playable_square(r, c):
                continue

            start = (r, c)

            # --- Simple moves ---
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if _on_board(nr, nc) and _playable_square(nr, nc):
                    actions.add((start, (nr, nc)))

            # --- Multi-jump generation ---
            def dfs(path: List[Coord], used_jumps: set):
                last = path[-1]

                if len(path) > 1:
                    actions.add(tuple(path))

                if len(path) >= max_depth:
                    return

                for dr, dc in directions:
                    mid = (last[0] + dr, last[1] + dc)
                    land = (last[0] + 2 * dr, last[1] + 2 * dc)

                    if not _on_board(*land):
                        continue
                    if not _playable_square(*land):
                        continue

                    jump_sq = mid

                    # 🚨 FIX: prevent reusing SAME captured square (not landing square)
                    if jump_sq in used_jumps:
                        continue

                    dfs(
                        path + [land],
                        used_jumps | {jump_sq},
                    )

            dfs([start], set())

    return sorted(actions)


# Build fixed action space
_ALL_ACTION_KEYS: List[ActionKey] = _generate_all_action_keys()
_ACTION_TO_INDEX: Dict[ActionKey, int] = {
    key: i for i, key in enumerate(_ALL_ACTION_KEYS)
}


def num_actions() -> int:
    return len(_ALL_ACTION_KEYS)


def action_key_to_index(key: ActionKey) -> int:
    if key not in _ACTION_TO_INDEX:
        raise KeyError(
            f"Action key not in fixed action space: {key}\n"
            f"This means your action space is STILL incomplete."
        )
    return _ACTION_TO_INDEX[key]


def index_to_action_key(index: int) -> ActionKey:
    return _ALL_ACTION_KEYS[index]


def validate_action_space_on_board(board: CheckersBoard) -> None:
    """
    Debug helper:
    Verifies ALL legal board moves exist in the fixed action space.
    """
    legal = board.legal_moves()
    for move in legal:
        key = move_to_key(move)
        if key not in _ACTION_TO_INDEX:
            raise RuntimeError(
                f"Missing action for legal move: {key}\n"
                f"Action space is incomplete."
            )