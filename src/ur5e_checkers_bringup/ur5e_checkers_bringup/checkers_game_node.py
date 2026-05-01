import json
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ur5e_checkers_bringup.board import CheckersBoard, Move, MoveSequence


class CheckersGameNode(Node):
    def __init__(self) -> None:
        super().__init__("checkers_game_node")

        self.declare_parameter("model_states_topic", "/checkers/piece_states")
        self.declare_parameter("board_state_topic", "/checkers/board_state")
        self.declare_parameter("legal_moves_topic", "/checkers/legal_moves")
        self.declare_parameter("game_event_topic", "/checkers/game_event")
        self.declare_parameter("selected_move_topic", "/checkers/selected_move")
        self.declare_parameter("robot_move_done_topic", "/checkers/robot_move_done")

        self.declare_parameter("update_hz", 5.0)
        self.declare_parameter("board_center_x", 0.6)
        self.declare_parameter("board_center_y", 0.0)
        self.declare_parameter("board_size", 0.40)
        self.declare_parameter("piece_diameter", 0.03)
        self.declare_parameter("starting_turn", "red")
        self.declare_parameter("robot_color", "red")

        self.model_states_topic = self.get_parameter("model_states_topic").value
        self.board_state_topic = self.get_parameter("board_state_topic").value
        self.legal_moves_topic = self.get_parameter("legal_moves_topic").value
        self.game_event_topic = self.get_parameter("game_event_topic").value
        self.selected_move_topic = self.get_parameter("selected_move_topic").value
        self.robot_move_done_topic = self.get_parameter("robot_move_done_topic").value

        self.update_hz = self.get_parameter("update_hz").value
        self.board_center_x = self.get_parameter("board_center_x").value
        self.board_center_y = self.get_parameter("board_center_y").value
        self.board_size = self.get_parameter("board_size").value
        self.piece_diameter = self.get_parameter("piece_diameter").value
        self.starting_turn = self.get_parameter("starting_turn").value
        self.robot_color = self.get_parameter("robot_color").value

        self.robot_turn = "b" if self.robot_color == "black" else "r"

        self.square_size = self.board_size / 8.0
        self.board_min_x = self.board_center_x - self.board_size / 2.0
        self.board_max_x = self.board_center_x + self.board_size / 2.0
        self.board_min_y = self.board_center_y - self.board_size / 2.0
        self.board_max_y = self.board_center_y + self.board_size / 2.0

        self.board = CheckersBoard()
        self.board.turn = "b" if self.starting_turn == "black" else "r"

        self.latest_model_states: Optional[List[dict]] = None
        self.prev_board_signature: Optional[Tuple[Tuple[str, ...], ...]] = None
        self.have_seen_initial_board = False

        self.pending_removed_squares: List[Tuple[int, int]] = []

        self.pending_robot_move = None
        self.robot_move_in_progress = False

        self.create_subscription(String, self.model_states_topic, self.model_states_callback, 10)
        self.create_subscription(String, self.selected_move_topic, self.selected_move_callback, 10)
        self.create_subscription(String, self.robot_move_done_topic, self.robot_move_done_callback, 10)

        self.board_state_pub = self.create_publisher(String, self.board_state_topic, 10)
        self.legal_moves_pub = self.create_publisher(String, self.legal_moves_topic, 10)
        self.game_event_pub = self.create_publisher(String, self.game_event_topic, 10)

        self.timer = self.create_timer(1.0 / self.update_hz, self.update_from_sim)

        self.get_logger().info("Checkers game node started.")
        self.get_logger().info(f"Starting turn: {self.board.turn}")
        self.get_logger().info(f"Robot turn: {self.robot_turn}")
        self.get_logger().info(f"Selected move topic: {self.selected_move_topic}")
        self.get_logger().info(f"Robot move done topic: {self.robot_move_done_topic}")

    def model_states_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_model_states = data
        except json.JSONDecodeError as e:
            self.get_logger().warning(f"Failed to decode piece_states JSON: {e}")

    def selected_move_callback(self, msg: String) -> None:
        if not self.have_seen_initial_board:
            return

        if self.robot_move_in_progress:
            return

        if self.board.turn != self.robot_turn:
            return

        try:
            move = self.parse_move_string(msg.data.strip())
        except Exception as e:
            self.get_logger().warning(f"Failed to parse selected robot move: {e}")
            return

        if move not in self.board.legal_moves():
            self.get_logger().warning(
                f"Ignoring selected robot move because it is not legal right now: "
                f"{self.move_to_string(move)}"
            )
            return

        self.pending_robot_move = move
        self.robot_move_in_progress = True

        self.get_logger().info(
            f"Stored pending robot move and paused board inference: "
            f"{self.move_to_string(move)}"
        )

        self.publish_current_board_outputs()

    def robot_move_done_callback(self, msg: String) -> None:
        if not self.robot_move_in_progress:
            self.get_logger().warning(
                "Received robot_move_done, but no robot move is in progress."
            )
            return

        if self.pending_robot_move is None:
            self.get_logger().warning(
                "Received robot_move_done, but pending_robot_move is None."
            )
            self.robot_move_in_progress = False
            return

        move = self.pending_robot_move

        self.get_logger().info(
            f"Robot reported move complete. Applying pending move: "
            f"{self.move_to_string(move)}"
        )

        self.apply_confirmed_move(move)

        self.pending_robot_move = None
        self.robot_move_in_progress = False
        self.prev_board_signature = self.board_signature(self.board.board)

        self.publish_current_board_outputs()

    def update_from_sim(self) -> None:
        if self.latest_model_states is None:
            return

        raw_detected_board = self.build_board_from_model_states(self.latest_model_states)
        raw_detected_board = self.merge_detected_board_with_internal_kings(raw_detected_board)

        previous_pending = set(self.pending_removed_squares)

        still_pending = []
        for row, col in self.pending_removed_squares:
            if raw_detected_board[row][col] != ".":
                still_pending.append((row, col))
        self.pending_removed_squares = still_pending

        pending_changed = previous_pending != set(self.pending_removed_squares)

        detected_board = self.apply_pending_removals(raw_detected_board)
        detected_signature = self.board_signature(detected_board)

        if not self.have_seen_initial_board:
            self.board.board = detected_board
            self.prev_board_signature = detected_signature
            self.have_seen_initial_board = True

            self.get_logger().info("Initialized board from simulation:")
            self.get_logger().info("\n" + self.format_board(self.board.board))
            self.get_logger().info(f"Current turn: {self.board.turn}")

            self.publish_current_board_outputs()
            return

        if self.robot_move_in_progress:
            self.publish_current_board_outputs()
            return

        if detected_signature == self.prev_board_signature and not pending_changed:
            self.publish_current_board_outputs()
            return

        inferred_move = self.infer_move_from_board_change(self.board.board, detected_board)

        if inferred_move is not None and inferred_move in self.board.legal_moves():
            self.apply_confirmed_move(inferred_move)
            self.prev_board_signature = self.board_signature(self.board.board)

            self.get_logger().info("Detected board change:")
            self.get_logger().info("\n" + self.format_board(self.board.board))
            self.get_logger().info(f"Current turn: {self.board.turn}")

            self.publish_current_board_outputs()
            return

        self.get_logger().warning(
            "Could not match detected board change to a legal move. "
            "Ignoring this frame instead of syncing from simulation."
        )

        self.publish_current_board_outputs()

    def apply_confirmed_move(self, move) -> None:
        try:
            was_capture = self.board.move_is_capture(move)
            capturing_player = self.board.turn
            old_board = [row[:] for row in self.board.board]

            self.board.apply_move(move)

            self.get_logger().info(f"Applied move: {self.move_to_string(move)}")
            self.get_logger().info(f"Turn after move: {self.board.turn}")

            if was_capture:
                captured_squares = self.get_captured_squares(move)
                self.add_pending_removed_squares(captured_squares)
                self.publish_capture_event(move, capturing_player)

            self.publish_promotion_events(old_board)

        except ValueError as e:
            self.get_logger().warning(
                f"Failed to apply move {self.move_to_string(move)}: {e}"
            )

    def publish_current_board_outputs(self) -> None:
        board_text = self.format_board(self.board.board)

        if self.robot_move_in_progress:
            legal_move_strings = []
        else:
            legal_moves = self.board.legal_moves()
            legal_move_strings = [self.move_to_string(move) for move in legal_moves]

        self.publish_board_state(board_text)
        self.publish_legal_moves(legal_move_strings)

    def build_board_from_model_states(self, pieces: List[dict]) -> List[List[str]]:
        board = [["." for _ in range(8)] for _ in range(8)]

        for piece in pieces:
            try:
                model_name = piece["name"]
                position = piece["position"]
                x = float(position["x"])
                y = float(position["y"])
            except (KeyError, TypeError, ValueError) as e:
                self.get_logger().warning(f"Skipping malformed piece entry: {piece} ({e})")
                continue

            piece_symbol = self.piece_entry_to_symbol(piece)
            if piece_symbol is None:
                continue

            row_col = self.world_to_square_majority(
                x=x,
                y=y,
                piece_diameter=self.piece_diameter,
            )

            if row_col is None:
                continue

            row, col = row_col

            if board[row][col] != ".":
                self.get_logger().warning(
                    f"Square conflict at ({row}, {col}): "
                    f"'{model_name}' mapped onto occupied square. Keeping first piece."
                )
                continue

            board[row][col] = piece_symbol

        return board

    def piece_entry_to_symbol(self, piece: dict) -> Optional[str]:
        try:
            model_name = piece["name"]
        except KeyError:
            return None

        is_king = piece.get("is_king", None)

        if model_name.startswith("red_"):
            if is_king is True:
                return "R"
            if is_king is False:
                return "r"

        if model_name.startswith("black_"):
            if is_king is True:
                return "B"
            if is_king is False:
                return "b"

        return self.model_name_to_symbol(model_name)

    def model_name_to_symbol(self, model_name: str) -> Optional[str]:
        if model_name.startswith("red_king_"):
            return "R"
        if model_name.startswith("black_king_"):
            return "B"
        if model_name.startswith("red_checker_"):
            return "r"
        if model_name.startswith("black_checker_"):
            return "b"
        return None

    def merge_detected_board_with_internal_kings(
        self,
        detected_board: List[List[str]],
    ) -> List[List[str]]:
        merged_board = [row[:] for row in detected_board]

        for row in range(8):
            for col in range(8):
                detected_piece = detected_board[row][col]
                internal_piece = self.board.board[row][col]

                if detected_piece == "r" and internal_piece == "R":
                    merged_board[row][col] = "R"
                elif detected_piece == "b" and internal_piece == "B":
                    merged_board[row][col] = "B"

        return merged_board

    def world_to_square_majority(
        self,
        x: float,
        y: float,
        piece_diameter: float,
    ) -> Optional[Tuple[int, int]]:
        half = piece_diameter / 2.0
        piece_min_x = x - half
        piece_max_x = x + half
        piece_min_y = y - half
        piece_max_y = y + half

        if (
            piece_max_x < self.board_min_x
            or piece_min_x > self.board_max_x
            or piece_max_y < self.board_min_y
            or piece_min_y > self.board_max_y
        ):
            return None

        best_square: Optional[Tuple[int, int]] = None
        best_overlap = 0.0

        for row in range(8):
            square_top = self.board_max_y - row * self.square_size
            square_bottom = square_top - self.square_size

            for col in range(8):
                square_left = self.board_min_x + col * self.square_size
                square_right = square_left + self.square_size

                overlap_x = max(
                    0.0,
                    min(piece_max_x, square_right) - max(piece_min_x, square_left),
                )
                overlap_y = max(
                    0.0,
                    min(piece_max_y, square_top) - max(piece_min_y, square_bottom),
                )
                overlap_area = overlap_x * overlap_y

                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_square = (row, col)
                elif math.isclose(overlap_area, best_overlap) and overlap_area > 0.0:
                    current_dist = self.distance_to_square_center(x, y, row, col)
                    assert best_square is not None
                    best_dist = self.distance_to_square_center(
                        x,
                        y,
                        best_square[0],
                        best_square[1],
                    )
                    if current_dist < best_dist:
                        best_square = (row, col)

        return best_square

    def distance_to_square_center(
        self,
        x: float,
        y: float,
        row: int,
        col: int,
    ) -> float:
        center_x = self.board_min_x + (col + 0.5) * self.square_size
        center_y = self.board_max_y - (row + 0.5) * self.square_size
        return math.hypot(x - center_x, y - center_y)

    def infer_move_from_board_change(
        self,
        old_board: List[List[str]],
        new_board: List[List[str]],
    ):
        legal_moves = self.board.legal_moves()

        self.get_logger().info(
            f"Trying to match detected board change against {len(legal_moves)} legal move(s)."
        )

        for move in legal_moves:
            board_copy = self.board.clone()
            move_str = self.move_to_string(move)

            try:
                board_copy.apply_move(move)
            except ValueError as e:
                self.get_logger().warning(
                    f"Skipping legal-move candidate that failed during apply: {move_str} ({e})"
                )
                continue

            if self.board_signature(board_copy.board) == self.board_signature(new_board):
                self.get_logger().info(f"Matched detected board change to move: {move_str}")
                return move

        return None

    def parse_move_string(self, move_str: str):
        parts = [part.strip() for part in move_str.split("->")]
        coords = []

        for part in parts:
            row_str, col_str = [x.strip() for x in part.split(",")]
            coords.append((int(row_str), int(col_str)))

        if len(coords) == 2:
            return Move(coords[0], coords[1])

        if len(coords) > 2:
            return MoveSequence(tuple(coords))

        raise ValueError(f"Invalid move string: {move_str}")

    def board_signature(self, board: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(cell for cell in row) for row in board)

    def format_board_grid(self, board: List[List[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def captured_counts_from_board(self, board: List[List[str]]) -> Tuple[int, int]:
        red_remaining = sum(1 for row in board for cell in row if cell in ("r", "R"))
        black_remaining = sum(1 for row in board for cell in row if cell in ("b", "B"))

        red_captured_by_black = 12 - red_remaining
        black_captured_by_red = 12 - black_remaining

        return red_captured_by_black, black_captured_by_red

    def king_counts_from_board(self, board: List[List[str]]) -> Tuple[int, int]:
        red_kings = sum(1 for row in board for cell in row if cell == "R")
        black_kings = sum(1 for row in board for cell in row if cell == "B")
        return red_kings, black_kings

    def format_board(self, board: List[List[str]]) -> str:
        board_grid = self.format_board_grid(board)
        red_captured_by_black, black_captured_by_red = self.captured_counts_from_board(board)
        red_kings, black_kings = self.king_counts_from_board(board)

        return (
            board_grid
            + "\n"
            + f"red_kings={red_kings} black_kings={black_kings}\n"
            + f"captured_red={red_captured_by_black} captured_black={black_captured_by_red}"
        )

    def move_to_string(self, move) -> str:
        try:
            if isinstance(move, MoveSequence):
                return " -> ".join(f"{r},{c}" for r, c in move.path)

            from_row, from_col = move.bgn
            to_row, to_col = move.dst
            return f"{from_row},{from_col} -> {to_row},{to_col}"
        except Exception:
            return str(move)

    def publish_board_state(self, board_text: str) -> None:
        msg = String()
        msg.data = board_text
        self.board_state_pub.publish(msg)

    def publish_legal_moves(self, legal_move_strings: List[str]) -> None:
        msg = String()
        msg.data = json.dumps(legal_move_strings, ensure_ascii=True)
        self.legal_moves_pub.publish(msg)

    def publish_game_event(self, event: dict) -> None:
        msg = String()
        msg.data = json.dumps(event, ensure_ascii=True)
        self.game_event_pub.publish(msg)

    def publish_capture_event(self, move, mover: str) -> None:
        if isinstance(move, MoveSequence):
            path = move.path
        else:
            path = (move.bgn, move.dst)

        captured_squares = self.get_captured_squares(move)

        if not captured_squares:
            return

        event = {
            "type": "capture",
            "by": mover,
            "from": list(path[0]),
            "to": list(path[-1]),
            "captured": [list(pos) for pos in captured_squares],
            "path": [list(p) for p in path],
        }

        self.publish_game_event(event)

    def publish_promotion_events(self, old_board: List[List[str]]) -> None:
        for row in range(8):
            for col in range(8):
                if old_board[row][col] == "r" and self.board.board[row][col] == "R":
                    self.publish_game_event(
                        {"type": "promote", "color": "red", "row": row, "col": col}
                    )
                elif old_board[row][col] == "b" and self.board.board[row][col] == "B":
                    self.publish_game_event(
                        {"type": "promote", "color": "black", "row": row, "col": col}
                    )

    def apply_pending_removals(
        self,
        board: List[List[str]],
    ) -> List[List[str]]:
        adjusted = [row[:] for row in board]

        for row, col in self.pending_removed_squares:
            adjusted[row][col] = "."

        return adjusted

    def get_captured_squares(self, move) -> List[Tuple[int, int]]:
        if isinstance(move, MoveSequence):
            path = move.path
        else:
            path = (move.bgn, move.dst)

        captured_squares = []

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            if abs(r2 - r1) == 2 and abs(c2 - c1) == 2:
                captured_squares.append(((r1 + r2) // 2, (c1 + c2) // 2))

        return captured_squares

    def add_pending_removed_squares(self, squares: List[Tuple[int, int]]) -> None:
        for square in squares:
            if square not in self.pending_removed_squares:
                self.pending_removed_squares.append(square)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CheckersGameNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()