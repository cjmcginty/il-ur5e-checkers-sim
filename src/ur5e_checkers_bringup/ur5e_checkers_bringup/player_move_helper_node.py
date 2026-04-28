import json
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

Square = Tuple[int, int]

MOVE_PATTERN = re.compile(
    r"^\s*(\d+)\s*,\s*(\d+)\s*->\s*(.+?)\s*$"
)
SQUARE_PATTERN = re.compile(r"(\d+)\s*,\s*(\d+)")


class PlayerMoveHelperNode(Node):
    """Help the human black-side player see legal checker moves.

    Subscribes to the game's legal move list and an optional selected piece topic.
    Publishes a readable help message that can be echoed in the terminal now and
    later reused by a UI or Gazebo marker/highlight node.
    """

    def __init__(self) -> None:
        super().__init__("player_move_helper_node")

        self.declare_parameter("legal_moves_topic", "/checkers/legal_moves")
        self.declare_parameter("selected_piece_topic", "/checkers/selected_player_piece")
        self.declare_parameter("help_topic", "/checkers/player_move_help")
        self.declare_parameter("player_color", "black")

        self.legal_moves_topic = (
            self.get_parameter("legal_moves_topic").get_parameter_value().string_value
        )
        self.selected_piece_topic = (
            self.get_parameter("selected_piece_topic").get_parameter_value().string_value
        )
        self.help_topic = (
            self.get_parameter("help_topic").get_parameter_value().string_value
        )
        self.player_color = (
            self.get_parameter("player_color").get_parameter_value().string_value
        )

        self.latest_moves: List[str] = []
        self.selected_piece: Optional[Square] = None
        self.last_help_text = ""

        self.help_pub = self.create_publisher(String, self.help_topic, 10)

        self.create_subscription(
            String,
            self.legal_moves_topic,
            self.legal_moves_callback,
            10,
        )
        self.create_subscription(
            String,
            self.selected_piece_topic,
            self.selected_piece_callback,
            10,
        )

        self.get_logger().info(
            f"Player move helper started. Listening to {self.legal_moves_topic} "
            f"and {self.selected_piece_topic}; publishing {self.help_topic}."
        )

    def legal_moves_callback(self, msg: String) -> None:
        self.latest_moves = self.parse_legal_moves_message(msg.data)

        # If a player already selected a piece, refresh the answer only when it changes.
        # This avoids spam but still handles turn changes / robot moves.
        if self.selected_piece is not None:
            self.publish_help()

    def selected_piece_callback(self, msg: String) -> None:
        selected = self.parse_square(msg.data)
        if selected is None:
            self.selected_piece = None
            self.publish_text(
                "Invalid selected piece. Use format 'row,col', for example '5,2'."
            )
            return

        self.selected_piece = selected
        self.publish_help()

    def parse_legal_moves_message(self, data: str) -> List[str]:
        """Parse /checkers/legal_moves.

        Expected current format is a JSON list of strings, for example:
        ["5,0 -> 4,1", "5,2 -> 4,1", "5,2 -> 4,3"]

        The fallback accepts one move per line in case the publisher changes later.
        """
        try:
            decoded = json.loads(data)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except json.JSONDecodeError:
            pass

        return [line.strip() for line in data.splitlines() if "->" in line]

    def parse_square(self, text: str) -> Optional[Square]:
        match = SQUARE_PATTERN.search(text)
        if match is None:
            return None

        row = int(match.group(1))
        col = int(match.group(2))

        if not (0 <= row <= 7 and 0 <= col <= 7):
            return None

        return row, col

    def parse_move(self, move_text: str) -> Optional[Tuple[Square, List[Square]]]:
        match = MOVE_PATTERN.match(move_text)
        if match is None:
            return None

        start = (int(match.group(1)), int(match.group(2)))
        destination_text = match.group(3)
        destinations = [
            (int(row), int(col))
            for row, col in SQUARE_PATTERN.findall(destination_text)
        ]

        if not destinations:
            return None

        return start, destinations

    def group_moves_by_start(self) -> Dict[Square, List[List[Square]]]:
        grouped: DefaultDict[Square, List[List[Square]]] = defaultdict(list)

        for move_text in self.latest_moves:
            parsed = self.parse_move(move_text)
            if parsed is None:
                continue

            start, destinations = parsed
            grouped[start].append(destinations)

        return dict(grouped)

    def publish_help(self) -> None:
        grouped = self.group_moves_by_start()

        if not grouped:
            self.publish_text(
                "No legal player moves available right now. It may be the robot/red turn, "
                "or the board state may not be initialized yet."
            )
            return

        lines: List[str] = []
        lines.append(f"Player side: {self.player_color}")

        if self.selected_piece is not None:
            lines.append(
                f"Selected piece: {self.square_to_text(self.selected_piece)}"
            )
            selected_moves = grouped.get(self.selected_piece, [])

            if not selected_moves:
                lines.append("This selected piece has no legal moves right now.")
                lines.append("Legal pieces you can move:")
                for start in sorted(grouped.keys()):
                    lines.append(f"- {self.square_to_text(start)}")
            else:
                lines.append("Legal destinations:")
                for path in selected_moves:
                    lines.append(f"- {self.path_to_text(path)}")
        else:
            lines.append("Legal moves:")
            for start in sorted(grouped.keys()):
                destination_texts = [self.path_to_text(path) for path in grouped[start]]
                lines.append(
                    f"- piece at {self.square_to_text(start)} can move to "
                    f"{', '.join(destination_texts)}"
                )

            lines.append("")
            lines.append(
                "To select a piece, publish for example: "
                "ros2 topic pub --once /checkers/selected_player_piece "
                "std_msgs/msg/String \"{data: '5,2'}\""
            )

        self.publish_text("\n".join(lines))

    def publish_text(self, text: str) -> None:
        if not hasattr(self, "last_help_text"):
            self.last_help_text = ""

        if text == self.last_help_text:
            return

        msg = String()
        msg.data = text
        self.help_pub.publish(msg)
        self.get_logger().info(text)

        self.last_help_text = text

    def square_to_text(self, square: Square) -> str:
        return f"{square[0]},{square[1]}"

    def path_to_text(self, path: List[Square]) -> str:
        return " -> ".join(self.square_to_text(square) for square in path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlayerMoveHelperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
