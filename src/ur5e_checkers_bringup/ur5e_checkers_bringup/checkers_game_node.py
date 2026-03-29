import json
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from ur5e_checkers_bringup.board import CheckersBoard, Move


class CheckersGameNode(Node):
    def __init__(self) -> None:
        super().__init__("checkers_game_node")

        # ---------------------------
        # Parameters
        # ---------------------------
        self.declare_parameter("model_states_topic", "/checkers/piece_states")
        self.declare_parameter("board_state_topic", "/checkers/board_state")
        self.declare_parameter("legal_moves_topic", "/checkers/legal_moves")

        self.declare_parameter("update_hz", 5.0)

        self.declare_parameter("board_center_x", 0.6)
        self.declare_parameter("board_center_y", 0.0)
        self.declare_parameter("board_size", 0.40)

        # Approximate checker footprint in XY for majority-square assignment.
        self.declare_parameter("piece_diameter", 0.03)

        # Simple turn handling for now.
        self.declare_parameter("starting_turn", "red")

        self.model_states_topic = (
            self.get_parameter("model_states_topic").get_parameter_value().string_value
        )
        self.board_state_topic = (
            self.get_parameter("board_state_topic").get_parameter_value().string_value
        )
        self.legal_moves_topic = (
            self.get_parameter("legal_moves_topic").get_parameter_value().string_value
        )

        self.update_hz = (
            self.get_parameter("update_hz").get_parameter_value().double_value
        )
        self.board_center_x = (
            self.get_parameter("board_center_x").get_parameter_value().double_value
        )
        self.board_center_y = (
            self.get_parameter("board_center_y").get_parameter_value().double_value
        )
        self.board_size = (
            self.get_parameter("board_size").get_parameter_value().double_value
        )
        self.piece_diameter = (
            self.get_parameter("piece_diameter").get_parameter_value().double_value
        )
        self.starting_turn = (
            self.get_parameter("starting_turn").get_parameter_value().string_value
        )

        self.square_size = self.board_size / 8.0
        self.board_min_x = self.board_center_x - self.board_size / 2.0
        self.board_max_x = self.board_center_x + self.board_size / 2.0
        self.board_min_y = self.board_center_y - self.board_size / 2.0
        self.board_max_y = self.board_center_y + self.board_size / 2.0

        # ---------------------------
        # Internal state
        # ---------------------------
        self.board = CheckersBoard()
        self.board.turn = "r" if self.starting_turn == "red" else "b"

        self.latest_model_states: Optional[List[dict]] = None
        self.prev_board_signature: Optional[Tuple[Tuple[str, ...], ...]] = None
        self.have_seen_initial_board = False

        # ---------------------------
        # ROS interfaces
        # ---------------------------
        self.model_states_sub = self.create_subscription(
            String,
            self.model_states_topic,
            self.model_states_callback,
            10,
        )

        self.board_state_pub = self.create_publisher(
            String,
            self.board_state_topic,
            10,
        )
        self.legal_moves_pub = self.create_publisher(
            String,
            self.legal_moves_topic,
            10,
        )

        timer_period = 1.0 / self.update_hz
        self.timer = self.create_timer(timer_period, self.update_from_sim)

        self.get_logger().info("Checkers game node started.")
        self.get_logger().info(f"Model states topic: {self.model_states_topic}")
        self.get_logger().info(f"Board state topic: {self.board_state_topic}")
        self.get_logger().info(f"Legal moves topic: {self.legal_moves_topic}")
        self.get_logger().info(f"Update rate: {self.update_hz:.2f} Hz")
        self.get_logger().info(
            f"Board geometry: center=({self.board_center_x:.3f}, {self.board_center_y:.3f}), "
            f"size={self.board_size:.3f}, square={self.square_size:.3f}"
        )
        self.get_logger().info(f"Starting turn: {self.board.turn}")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def model_states_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_model_states = data
            else:
                self.get_logger().warning("Received piece_states payload that is not a list.")
        except json.JSONDecodeError as e:
            self.get_logger().warning(f"Failed to decode piece_states JSON: {e}")

    def update_from_sim(self) -> None:
        if self.latest_model_states is None:
            return

        detected_board = self.build_board_from_model_states(self.latest_model_states)
        detected_signature = self.board_signature(detected_board)

        if detected_signature == self.prev_board_signature:
            return

        if not self.have_seen_initial_board:
            self.board.board = detected_board
            self.prev_board_signature = detected_signature
            self.have_seen_initial_board = True

            board_text = self.format_board(self.board.board)
            legal_moves = self.board.legal_moves()
            legal_move_strings = [self.move_to_string(move) for move in legal_moves]

            self.get_logger().info("Initialized board from simulation:")
            self.get_logger().info("\n" + board_text)
            self.get_logger().info(f"Current turn: {self.board.turn}")
            self.get_logger().info(f"Legal moves available: {len(legal_move_strings)}")

            if legal_move_strings:
                self.get_logger().info(
                    f"Legal moves: {json.dumps(legal_move_strings, ensure_ascii=True)}"
                )

            self.publish_board_state(board_text)
            self.publish_legal_moves(legal_move_strings)
            return

        inferred_move = self.infer_move_from_board_change(self.board.board, detected_board)

        if inferred_move is not None and inferred_move in self.board.legal_moves():
            try:
                self.board.apply_move(inferred_move)
                self.get_logger().info(
                    f"Applied detected move: {self.move_to_string(inferred_move)}"
                )
            except ValueError as e:
                self.get_logger().warning(
                    f"Failed to apply inferred move {self.move_to_string(inferred_move)}: {e}"
                )
                self.board.board = detected_board
                self.flip_turn()
        else:
            self.get_logger().warning(
                "Could not match detected board change to a legal move. "
                "Syncing board directly from simulation."
            )
            self.board.board = detected_board
            self.flip_turn()

        self.prev_board_signature = self.board_signature(self.board.board)

        board_text = self.format_board(self.board.board)
        legal_moves = self.board.legal_moves()
        legal_move_strings = [self.move_to_string(move) for move in legal_moves]

        self.get_logger().info("Detected board change:")
        self.get_logger().info("\n" + board_text)
        self.get_logger().info(f"Current turn: {self.board.turn}")
        self.get_logger().info(f"Legal moves available: {len(legal_move_strings)}")

        if legal_move_strings:
            self.get_logger().info(
                f"Legal moves: {json.dumps(legal_move_strings, ensure_ascii=True)}"
            )

        self.publish_board_state(board_text)
        self.publish_legal_moves(legal_move_strings)

    # ------------------------------------------------------------------
    # Board reconstruction
    # ------------------------------------------------------------------

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

            piece_symbol = self.model_name_to_symbol(model_name)
            if piece_symbol is None:
                continue

            row_col = self.world_to_square_majority(
                x=x,
                y=y,
                piece_diameter=self.piece_diameter,
            )

            if row_col is None:
                self.get_logger().warning(
                    f"Piece '{model_name}' is off the board or could not be mapped "
                    f"from pose ({x:.3f}, {y:.3f})."
                )
                continue

            row, col = row_col

            if board[row][col] != ".":
                self.get_logger().warning(
                    f"Square conflict at ({row}, {col}): "
                    f"'{model_name}' mapped onto an occupied square. "
                    f"Keeping first piece '{board[row][col]}' and continuing."
                )
                continue

            if (row + col) % 2 == 0:
                self.get_logger().warning(
                    f"Piece '{model_name}' mapped to light square ({row}, {col}). Continuing anyway."
                )

            board[row][col] = piece_symbol

        return board

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

    def world_to_square_majority(
        self,
        x: float,
        y: float,
        piece_diameter: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Assign a piece to the square containing the majority of its XY footprint.
        We approximate the checker footprint as an axis-aligned square centered at (x, y).
        """
        half = piece_diameter / 2.0
        piece_min_x = x - half
        piece_max_x = x + half
        piece_min_y = y - half
        piece_max_y = y + half

        # Quick reject if no overlap with the board at all.
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
                        x, y, best_square[0], best_square[1]
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
    ) -> Optional[Move]:
        player = self.board.turn

        if player == "r":
            player_pieces_old = {"r", "R"}
            player_pieces_new = {"r", "R"}
            opponent_pieces = {"b", "B"}
        else:
            player_pieces_old = {"b", "B"}
            player_pieces_new = {"b", "B"}
            opponent_pieces = {"r", "R"}

        src = None
        dst = None
        removed_opponents = 0

        for r in range(8):
            for c in range(8):
                old_cell = old_board[r][c]
                new_cell = new_board[r][c]

                if old_cell == new_cell:
                    continue

                if old_cell in player_pieces_old and new_cell == ".":
                    if src is not None:
                        return None
                    src = (r, c)

                elif old_cell == "." and new_cell in player_pieces_new:
                    if dst is not None:
                        return None
                    dst = (r, c)
                elif old_cell in opponent_pieces and new_cell == ".":
                    removed_opponents += 1
                else:
                    return None

        if src is None or dst is None:
            return None

        if removed_opponents > 1:
            return None

        if removed_opponents == 1 and abs(dst[0] - src[0]) != 2:
            return None

        if removed_opponents == 0 and abs(dst[0] - src[0]) != 1:
            return None

        return Move(src, dst)

    # ------------------------------------------------------------------
    # Formatting / publishing
    # ------------------------------------------------------------------

    def board_signature(self, board: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(cell for cell in row) for row in board)

    def format_board(self, board: List[List[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def move_to_string(self, move) -> str:
        try:
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

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    def flip_turn(self) -> None:
        if self.board.turn == "r":
            self.board.turn = "b"
        else:
            self.board.turn = "r"


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