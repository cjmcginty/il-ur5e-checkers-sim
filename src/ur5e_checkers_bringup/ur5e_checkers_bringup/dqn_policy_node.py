import json
from pathlib import Path
from typing import List, Optional

import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import String

from ur5e_checkers_bringup.board import CheckersBoard, Move, MoveLike, MoveSequence
from ur5e_checkers_bringup.dqn_model import DQN
from ur5e_checkers_bringup.dqn_utils import (
    canonicalize_move_key_for_player,
    encode_canonical_board,
    move_to_key,
)
from ur5e_checkers_bringup.dqn_action_space import action_key_to_index


class DQNPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("dqn_policy_node")

        self.declare_parameter("board_state_topic", "/checkers/board_state")
        self.declare_parameter("legal_moves_topic", "/checkers/legal_moves")
        self.declare_parameter("selected_move_topic", "/checkers/selected_move")
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("publish_once_per_position", True)
        self.declare_parameter("republish_hz", 2.0)

        self.board_state_topic = (
            self.get_parameter("board_state_topic").get_parameter_value().string_value
        )
        self.legal_moves_topic = (
            self.get_parameter("legal_moves_topic").get_parameter_value().string_value
        )
        self.selected_move_topic = (
            self.get_parameter("selected_move_topic").get_parameter_value().string_value
        )
        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.device_param = (
            self.get_parameter("device").get_parameter_value().string_value
        )
        self.publish_once_per_position = (
            self.get_parameter("publish_once_per_position")
            .get_parameter_value()
            .bool_value
        )
        self.republish_hz = (
            self.get_parameter("republish_hz")
            .get_parameter_value()
            .double_value
        )

        if self.device_param == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_param)

        self.policy_net = DQN().to(self.device)
        self.policy_net.eval()
        self.model_loaded = False

        self.latest_board_text: Optional[str] = None
        self.latest_legal_move_strings: List[str] = []
        self.last_published_position_key: Optional[str] = None
        self.last_selected_move_str: Optional[str] = None
        self.last_selected_position_key: Optional[str] = None

        self.selected_move_pub = self.create_publisher(
            String,
            self.selected_move_topic,
            10,
        )

        self.create_subscription(
            String,
            self.board_state_topic,
            self.board_state_callback,
            10,
        )
        self.create_subscription(
            String,
            self.legal_moves_topic,
            self.legal_moves_callback,
            10,
        )

        self.create_timer(1.0 / max(self.republish_hz, 0.1), self.republish_selected_move)

        self.load_model_if_possible()

        self.get_logger().info("DQN policy node started.")
        self.get_logger().info(f"Board state topic: {self.board_state_topic}")
        self.get_logger().info(f"Legal moves topic: {self.legal_moves_topic}")
        self.get_logger().info(f"Selected move topic: {self.selected_move_topic}")

    def load_model_if_possible(self) -> None:
        if not self.model_path:
            self.get_logger().warning(
                "No model_path provided. DQN policy node will not publish moves "
                "until a valid model path is set."
            )
            return

        path = Path(self.model_path)
        if not path.exists():
            self.get_logger().warning(
                f"Model file does not exist: {self.model_path}"
            )
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["state_dict"])
            else:
                self.policy_net.load_state_dict(checkpoint)

            self.policy_net.eval()
            self.model_loaded = True
            self.get_logger().info(f"Loaded DQN weights from: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model from {self.model_path}: {e}")
            self.model_loaded = False

    def board_state_callback(self, msg: String) -> None:
        self.latest_board_text = msg.data
        self.try_publish_selected_move()

    def legal_moves_callback(self, msg: String) -> None:
        text = msg.data.strip()

        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                self.latest_legal_move_strings = [str(x) for x in data]
            else:
                self.latest_legal_move_strings = []
        except json.JSONDecodeError:
            # Fallback: assume newline-separated plain text
            self.latest_legal_move_strings = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

        self.try_publish_selected_move()

    def try_publish_selected_move(self) -> None:
        if not self.model_loaded:
            return

        if self.latest_board_text is None:
            return

        if not self.latest_legal_move_strings:
            return

        try:
            board = self.parse_board_state_text(self.latest_board_text)
        except Exception as e:
            self.get_logger().warning(f"Failed to parse board state: {e}")
            return

        try:
            legal_moves = [self.parse_move_string(s) for s in self.latest_legal_move_strings]
        except Exception as e:
            self.get_logger().warning(f"Failed to parse legal move strings: {e}")
            return

        if not legal_moves:
            return

        inferred_turn = self.infer_turn_from_legal_moves(board, legal_moves)
        if inferred_turn is None:
            self.get_logger().warning(
                "Could not infer player turn from legal moves. Not publishing."
            )
            return

        board.turn = inferred_turn

        position_key = self.make_position_key(board, self.latest_legal_move_strings)

        if self.publish_once_per_position and position_key == self.last_published_position_key:
            return

        try:
            selected_move = self.select_best_legal_move(board, legal_moves)
        except Exception as e:
            self.get_logger().warning(f"Failed to select move: {e}")
            return

        selected_move_str = self.move_to_string(selected_move)

        self.last_selected_move_str = selected_move_str
        self.last_selected_position_key = position_key

        msg = String()
        msg.data = selected_move_str
        self.selected_move_pub.publish(msg)

        self.last_published_position_key = position_key

        self.get_logger().info(f"Published selected move: {selected_move_str}")

    def republish_selected_move(self) -> None:
        if not self.model_loaded:
            return

        if self.last_selected_move_str is None:
            return

        msg = String()
        msg.data = self.last_selected_move_str
        self.selected_move_pub.publish(msg)

    def parse_board_state_text(self, board_text: str) -> CheckersBoard:
        lines = [line.strip() for line in board_text.strip().splitlines() if line.strip()]

        if len(lines) < 8:
            raise ValueError("Board state text did not contain 8 board rows.")

        board_rows = lines[:8]

        parsed_board: List[List[str]] = []
        for row_text in board_rows:
            row = row_text.split()
            if len(row) != 8:
                raise ValueError(f"Expected 8 columns in row '{row_text}', got {len(row)}")
            parsed_board.append(row)

        board = CheckersBoard()
        board.board = parsed_board

        red_remaining = sum(1 for row in parsed_board for cell in row if cell in ("r", "R"))
        black_remaining = sum(1 for row in parsed_board for cell in row if cell in ("b", "B"))

        board.red_captured = 12 - red_remaining
        board.black_captured = 12 - black_remaining

        return board

    def parse_move_string(self, move_str: str) -> MoveLike:
        parts = [part.strip() for part in move_str.split("->")]
        coords = []

        for part in parts:
            row_str, col_str = [x.strip() for x in part.split(",")]
            coords.append((int(row_str), int(col_str)))

        if len(coords) == 2:
            return Move(coords[0], coords[1])

        if len(coords) >= 2:
            return MoveSequence(tuple(coords))

        raise ValueError(f"Invalid move string: {move_str}")

    def infer_turn_from_legal_moves(
        self,
        board: CheckersBoard,
        legal_moves: List[MoveLike],
    ) -> Optional[str]:
        if not legal_moves:
            return None

        first_move = legal_moves[0]
        start = first_move.bgn
        piece = board.piece_at(start)

        if piece in ("r", "R"):
            return "r"
        if piece in ("b", "B"):
            return "b"

        return None

    def select_best_legal_move(
        self,
        board: CheckersBoard,
        legal_moves: List[MoveLike],
    ) -> MoveLike:
        state = encode_canonical_board(board).unsqueeze(0).to(self.device)

        legal_indices = []
        indexed_moves = []

        for move in legal_moves:
            raw_key = move_to_key(move)
            canonical_key = canonicalize_move_key_for_player(raw_key, board.turn)
            idx = action_key_to_index(canonical_key)
            legal_indices.append(idx)
            indexed_moves.append((move, idx))

        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).detach().cpu()

        masked_q = torch.full_like(q_values, -1e9)
        masked_q[legal_indices] = q_values[legal_indices]
        best_idx = int(torch.argmax(masked_q).item())

        for move, idx in indexed_moves:
            if idx == best_idx:
                return move

        raise RuntimeError("Best legal action index did not map back to a move.")

    def make_position_key(self, board: CheckersBoard, legal_move_strings: List[str]) -> str:
        board_signature = "\n".join(" ".join(row) for row in board.board)
        legal_signature = json.dumps(legal_move_strings, sort_keys=False)
        return f"{board.turn}||{board_signature}||{legal_signature}"

    def move_to_string(self, move: MoveLike) -> str:
        if isinstance(move, MoveSequence):
            return " -> ".join(f"{r},{c}" for r, c in move.path)

        return f"{move.bgn[0]},{move.bgn[1]} -> {move.dst[0]},{move.dst[1]}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DQNPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()