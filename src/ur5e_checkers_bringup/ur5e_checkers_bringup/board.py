from dataclasses import dataclass
from typing import List, Tuple, Union
import copy

# (0,0) is the top left, (7,7) is the bottom right
Coord = Tuple[int, int]  # row, column


# represents a single move from one square to another
# example: Move((5,0), (4,1))
@dataclass(frozen=True)
class Move:
    bgn: Coord  # beginning square
    dst: Coord  # destination square


# represents a multi-step move path
# examples:
# MoveSequence(((5,0), (4,1)))
# MoveSequence(((5,0), (3,2), (1,4)))
@dataclass(frozen=True)
class MoveSequence:
    path: Tuple[Coord, ...]

    @property
    def bgn(self) -> Coord:
        return self.path[0]

    @property
    def dst(self) -> Coord:
        return self.path[-1]


MoveLike = Union[Move, MoveSequence]


# r is red, b is black, . is empty
# red moves downward toward row 7
# black moves upward toward row 0
class CheckersBoard:
    def __init__(self) -> None:
        # create empty 8x8 board
        self.board: List[List[str]] = [["." for _ in range(8)] for _ in range(8)]

        # optional red always moves first
        self.turn: str = "r"

        # captured counters
        self.red_captured: int = 0
        self.black_captured: int = 0

        # place starting pieces
        self._setup()

    # place starting pieces on dark squares
    # dark squares are where row + col is odd
    def _setup(self) -> None:
        # top 3 rows are red pieces
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r][c] = "r"

        # bottom 3 rows are black pieces
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r][c] = "b"

    # return a deep copy of the board, will be useful later for search or RL
    def clone(self) -> "CheckersBoard":
        return copy.deepcopy(self)

    # basic board helpers below

    # return the piece at a given position
    def piece_at(self, pos: Coord) -> str:
        r, c = pos
        return self.board[r][c]

    # set a board square to a given value (b, r, .)
    def set_piece(self, pos: Coord, value: str) -> None:
        r, c = pos
        self.board[r][c] = value

    # piece ownership helpers (handles kings too)
    def is_red_piece(self, piece: str) -> bool:
        return piece in ("r", "R")

    def is_black_piece(self, piece: str) -> bool:
        return piece in ("b", "B")

    def belongs_to_player(self, piece: str, player: str) -> bool:
        if player == "r":
            return self.is_red_piece(piece)
        return self.is_black_piece(piece)

    # return a list the coordinates of all pieces belonging to the given player
    def all_pieces(self, player: str) -> List[Coord]:
        pieces = []
        for r in range(8):
            for c in range(8):
                if self.belongs_to_player(self.board[r][c], player):
                    pieces.append((r, c))
        return pieces

    # movement logic helpers below

    # return the opposing player
    def _opponent(self, player: str) -> str:
        return "b" if player == "r" else "r"

    # return the allowed diagonal movement directions for a given player
    def _move_directions_for_piece(self, piece: str):
        if piece == "r":
            return [(1, -1), (1, 1)]  # red moves down
        if piece == "b":
            return [(-1, -1), (-1, 1)]  # black moves up
        if piece in ("R", "B"):  # kings move both directions
            return [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        return []

    def move_is_capture(self, move: MoveLike) -> bool:
        if isinstance(move, MoveSequence):
            if len(move.path) < 2:
                return False
            return abs(move.path[1][0] - move.path[0][0]) == 2
        return abs(move.dst[0] - move.bgn[0]) == 2

    # return all legal single-step moves from a source square
    def legal_moves_from(self, bgn: Coord) -> List[Move]:
        r, c = bgn
        piece = self.piece_at(bgn)

        # makes sure u can only move your own piece
        if not self.belongs_to_player(piece, self.turn):
            return []

        moves: List[Move] = []

        # check diagonal directions
        for dr, dc in self._move_directions_for_piece(piece):
            r2, c2 = r + dr, c + dc

            # stay inside board boundaries
            if 0 <= r2 < 8 and 0 <= c2 < 8:
                # only move if destination is empty
                if self.board[r2][c2] == ".":
                    moves.append(Move(bgn, (r2, c2)))

        return moves

    # return all legal capture moves from a source square
    def legal_captures_from(self, bgn: Coord) -> List[Move]:
        r, c = bgn
        piece = self.piece_at(bgn)

        # makes sure u can only move your own piece
        if not self.belongs_to_player(piece, self.turn):
            return []

        moves: List[Move] = []
        opponent = self._opponent(self.turn)

        for dr, dc in self._move_directions_for_piece(piece):
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc

            if 0 <= r2 < 8 and 0 <= c2 < 8:
                jumped_piece = self.board[r1][c1]
                if self.board[r2][c2] == "." and self.belongs_to_player(jumped_piece, opponent):
                    moves.append(Move(bgn, (r2, c2)))

        return moves

    def _apply_single_step_no_turn_flip(self, move: Move) -> Coord:
        piece = self.piece_at(move.bgn)
        self.set_piece(move.bgn, ".")
        self.set_piece(move.dst, piece)

        if self.move_is_capture(move):
            jumped_r = (move.bgn[0] + move.dst[0]) // 2
            jumped_c = (move.bgn[1] + move.dst[1]) // 2
            jumped_piece = self.piece_at((jumped_r, jumped_c))

            if self.is_red_piece(jumped_piece):
                self.red_captured += 1
            elif self.is_black_piece(jumped_piece):
                self.black_captured += 1

            self.set_piece((jumped_r, jumped_c), ".")

        self.maybe_promote_to_king(move.dst)
        return move.dst

    def _capture_sequences_from(
        self,
        start: Coord,
        path_prefix: List[Coord] | None = None,
    ) -> List["MoveSequence"]:
        if path_prefix is None:
            path_prefix = [start]

        piece = self.piece_at(start)
        if not self.belongs_to_player(piece, self.turn):
            return []

        immediate_captures = self.legal_captures_from(start)
        if not immediate_captures:
            if len(path_prefix) > 1:
                return [MoveSequence(tuple(path_prefix))]
            return []

        sequences: List[MoveSequence] = []

        for capture in immediate_captures:
            board_copy = self.clone()
            landing = board_copy._apply_single_step_no_turn_flip(capture)

            continued = board_copy._capture_sequences_from(
                landing,
                path_prefix + [landing],
            )

            if continued:
                sequences.extend(continued)
            else:
                sequences.append(MoveSequence(tuple(path_prefix + [landing])))

        return sequences

    # return all legal moves for the current player
    def legal_moves(self) -> List[MoveLike]:
        capture_sequences: List[MoveSequence] = []
        moves: List[Move] = []

        for bgn in self.all_pieces(self.turn):
            capture_sequences.extend(self._capture_sequences_from(bgn))
            moves.extend(self.legal_moves_from(bgn))

        # if any capture exists, it must be taken
        return capture_sequences if capture_sequences else moves

    # check if pieces should be promoted to kings
    def maybe_promote_to_king(self, pos: Coord) -> None:
        piece = self.piece_at(pos)
        r, _ = pos

        if piece == "r" and r == 7:
            self.set_piece(pos, "R")
        elif piece == "b" and r == 0:
            self.set_piece(pos, "B")

    def _normalize_move(self, move: MoveLike) -> MoveSequence:
        if isinstance(move, MoveSequence):
            return move
        return MoveSequence((move.bgn, move.dst))

    # apply a move to the board and switch turns, raises error if move is illegal
    def apply_move(self, move: MoveLike) -> None:
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError("Illegal move")

        seq = self._normalize_move(move)

        for i in range(len(seq.path) - 1):
            step = Move(seq.path[i], seq.path[i + 1])
            self._apply_single_step_no_turn_flip(step)

        # switch turn after full move sequence finishes
        self.turn = "b" if self.turn == "r" else "r"

    # return the winner if the game is over, otherwise return None
    def winner(self) -> str | None:
        if len(self.all_pieces("r")) == 0:
            return "b"
        if len(self.all_pieces("b")) == 0:
            return "r"
        if len(self.legal_moves()) == 0:
            return "b" if self.turn == "r" else "r"
        return None

    # return whether the game is over
    def is_game_over(self) -> bool:
        return self.winner() is not None