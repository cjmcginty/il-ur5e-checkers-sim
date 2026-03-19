import json
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data

from ur5e_checkers_bringup.board import CheckersBoard


class CheckersGameNode(Node):
    def __init__(self) -> None:
        super().__init__("checkers_game_node")

        # ---------------------------
        # Parameters
        # ---------------------------
        self.declare_parameter("model_states_topic", "/world/checkers_world/pose")
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

        self.latest_model_states: Optional[TFMessage] = None
        self.prev_board_signature: Optional[Tuple[Tuple[str, ...], ...]] = None
        self.have_seen_initial_board = False

        # Track piece identities even though the TF bridge gives empty child_frame_id.
        self.tracked_red_positions: List[Tuple[float, float]] = []
        self.tracked_black_positions: List[Tuple[float, float]] = []

        # ---------------------------
        # ROS interfaces
        # ---------------------------
        self.model_states_sub = self.create_subscription(
            TFMessage,
            self.model_states_topic,
            self.model_states_callback,
            qos_profile_sensor_data,
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

    def model_states_callback(self, msg: TFMessage) -> None:
        self.latest_model_states = msg

    def update_from_sim(self) -> None:
        if self.latest_model_states is None:
            return

        detected_board = self.build_board_from_model_states(self.latest_model_states)
        detected_signature = self.board_signature(detected_board)

        if detected_signature == self.prev_board_signature:
            return

        if self.have_seen_initial_board:
            self.flip_turn()
        else:
            self.have_seen_initial_board = True

        self.board.board = detected_board
        self.prev_board_signature = detected_signature

        board_text = self.format_board(detected_board)
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

    def build_board_from_model_states(self, msg: TFMessage) -> List[List[str]]:
        board = [["." for _ in range(8)] for _ in range(8)]

        # The TF bridge gives empty child_frame_id for this topic, so we identify
        # checker pieces purely from their on-board positions and track color over time.
        detections: List[Tuple[float, float]] = []

        for transform in msg.transforms:
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            if not self.is_likely_checker_piece(x, y, z):
                continue

            detections.append((x, y))

        if not detections:
            return board

        red_positions, black_positions = self.assign_piece_colors(detections)

        for x, y in red_positions:
            row_col = self.world_to_square_majority(
                x=x,
                y=y,
                piece_diameter=self.piece_diameter,
            )
            if row_col is None:
                self.get_logger().warning(
                    f"Red piece is off the board or could not be mapped from pose ({x:.3f}, {y:.3f})."
                )
                continue

            row, col = row_col

            if board[row][col] != ".":
                self.get_logger().warning(
                    f"Square conflict at ({row}, {col}): red piece mapped onto an occupied square. "
                    f"Keeping first piece '{board[row][col]}' and continuing."
                )
                continue

            if (row + col) % 2 == 0:
                self.get_logger().warning(
                    f"Red piece mapped to light square ({row}, {col}). Continuing anyway."
                )

            board[row][col] = "r"

        for x, y in black_positions:
            row_col = self.world_to_square_majority(
                x=x,
                y=y,
                piece_diameter=self.piece_diameter,
            )
            if row_col is None:
                self.get_logger().warning(
                    f"Black piece is off the board or could not be mapped from pose ({x:.3f}, {y:.3f})."
                )
                continue

            row, col = row_col

            if board[row][col] != ".":
                self.get_logger().warning(
                    f"Square conflict at ({row}, {col}): black piece mapped onto an occupied square. "
                    f"Keeping first piece '{board[row][col]}' and continuing."
                )
                continue

            if (row + col) % 2 == 0:
                self.get_logger().warning(
                    f"Black piece mapped to light square ({row}, {col}). Continuing anyway."
                )

            board[row][col] = "b"

        return board

    def is_likely_checker_piece(self, x: float, y: float, z: float) -> bool:
        # Keep only objects physically on the board footprint with checker-like height.
        if (
            x < self.board_min_x - self.square_size
            or x > self.board_max_x + self.square_size
            or y < self.board_min_y - self.square_size
            or y > self.board_max_y + self.square_size
        ):
            return False

        # Checker pieces in your sim are near z ~= 0.021. This excludes robot links / board.
        if not (0.005 <= z <= 0.06):
            return False

        return True

    def assign_piece_colors(
        self, detections: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        # First frame: infer color from starting side of board.
        if not self.tracked_red_positions and not self.tracked_black_positions:
            sorted_by_y = sorted(detections, key=lambda p: p[1], reverse=True)
            self.tracked_red_positions = sorted_by_y[:12]
            self.tracked_black_positions = sorted_by_y[12:24]
            return self.tracked_red_positions, self.tracked_black_positions

        # Later frames: preserve color by nearest-neighbor matching to previous positions.
        red_positions = self.match_positions(self.tracked_red_positions, detections)
        remaining = self.remove_matched(detections, red_positions)
        black_positions = self.match_positions(self.tracked_black_positions, remaining)

        self.tracked_red_positions = red_positions
        self.tracked_black_positions = black_positions

        return red_positions, black_positions

    def match_positions(
        self,
        previous_positions: List[Tuple[float, float]],
        detections: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        remaining = list(detections)
        matched: List[Tuple[float, float]] = []

        for px, py in previous_positions:
            if not remaining:
                break

            best_idx = min(
                range(len(remaining)),
                key=lambda i: math.hypot(remaining[i][0] - px, remaining[i][1] - py),
            )
            matched.append(remaining.pop(best_idx))

        return matched

    def remove_matched(
        self,
        detections: List[Tuple[float, float]],
        matched: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        remaining = list(detections)
        for mx, my in matched:
            for i, (x, y) in enumerate(remaining):
                if math.isclose(x, mx, abs_tol=1e-9) and math.isclose(y, my, abs_tol=1e-9):
                    remaining.pop(i)
                    break
        return remaining

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

    # ------------------------------------------------------------------
    # Formatting / publishing
    # ------------------------------------------------------------------

    def board_signature(self, board: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(cell for cell in row) for row in board)

    def format_board(self, board: List[List[str]]) -> str:
        return "\n".join(" ".join(row) for row in board)

    def move_to_string(self, move) -> str:
        """
        Expected move shape from board.py:
            ((from_row, from_col), (to_row, to_col))
        """
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