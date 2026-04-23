import json
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

"""
MoveTargetNode
==============

This node converts a high-level checkers move string into structured data that
includes both board coordinates and corresponding world coordinates.

INPUT
-----
Subscribed topics:
    /checkers/selected_move   (std_msgs/msg/String)
    /checkers/piece_states    (std_msgs/msg/String)

Format of incoming selected move:
    "row,col -> row,col"
    or for multi-jump:
    "row,col -> row,col -> row,col ..."

Examples:
    Single move:
        "2,5 -> 3,6"

    Multi-jump move:
        "2,3 -> 4,5 -> 6,7"


PROCESSING
----------
1. The move string is parsed into a list of (row, col) tuples.
2. The first coordinate is treated as the starting square (pickup location).
3. All remaining coordinates are treated as the path (destination(s)).
4. The start square is matched to the actual checker pose from /checkers/piece_states.
5. The destination square(s) are converted into square-center world coordinates using:
       - board_center_x
       - board_center_y
       - board_size
       - piece_z
6. The result is packaged into a JSON string and published.


OUTPUT
------
Published topic:
    /checkers/move_targets   (std_msgs/msg/String)

The output is a JSON-formatted string with the following structure:

Common fields:
    start_square : [row, col]
    path_squares : list of [row, col]
    start_world  : [x, y, z]   <- actual checker pose
    path_world   : list of [x, y, z]   <- destination square center(s)


Example OUTPUT (Single Move)
---------------------------
Input:
    "2,5 -> 3,6"

Output:
{
  "start_square": [2, 5],
  "path_squares": [[3, 6]],
  "start_world": [x_piece, y_piece, z_piece],
  "path_world": [[x_square, y_square, z_square]]
}


Example OUTPUT (Multi-Jump)
--------------------------
Input:
    "2,3 -> 4,5 -> 6,7"

Output:
{
  "start_square": [2, 3],
  "path_squares": [[4, 5], [6, 7]],
  "start_world": [x_piece, y_piece, z_piece],
  "path_world": [
    [x2, y2, z],
    [x3, y3, z]
  ]
}


NOTES
-----
- Board coordinates are 0-indexed:
      (0,0) = top-left of board
      (7,7) = bottom-right

- The path always includes ALL landing squares after the start.
  This makes multi-jump handling straightforward:
      - pick at start_world
      - move through each point in path_world
      - place at final position

- This node does NOT perform motion planning. It only translates
  symbolic moves into spatial targets for downstream IL or control nodes.
"""


class MoveTargetNode(Node):
    def __init__(self) -> None:
        super().__init__("move_target_node")

        self.declare_parameter("selected_move_topic", "/checkers/selected_move")
        self.declare_parameter("piece_states_topic", "/checkers/piece_states")
        self.declare_parameter("move_target_topic", "/checkers/move_targets")
        self.declare_parameter("board_center_x", 0.6)
        self.declare_parameter("board_center_y", 0.0)
        self.declare_parameter("board_size", 0.40)
        self.declare_parameter("piece_z", 0.03)

        self.selected_move_topic = (
            self.get_parameter("selected_move_topic").get_parameter_value().string_value
        )
        self.piece_states_topic = (
            self.get_parameter("piece_states_topic").get_parameter_value().string_value
        )
        self.move_target_topic = (
            self.get_parameter("move_target_topic").get_parameter_value().string_value
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
        self.piece_z = (
            self.get_parameter("piece_z").get_parameter_value().double_value
        )

        self.square_size = self.board_size / 8.0
        self.latest_piece_states: Optional[List[dict]] = None

        self.subscription = self.create_subscription(
            String,
            self.selected_move_topic,
            self.selected_move_callback,
            10,
        )

        self.piece_states_subscription = self.create_subscription(
            String,
            self.piece_states_topic,
            self.piece_states_callback,
            10,
        )

        self.publisher = self.create_publisher(
            String,
            self.move_target_topic,
            10,
        )

        self.get_logger().info("Move target node started.")

    def piece_states_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_piece_states = data
        except json.JSONDecodeError:
            self.get_logger().warning("Failed to parse /checkers/piece_states JSON.")

    def selected_move_callback(self, msg: String) -> None:
        move_str = msg.data.strip()

        try:
            squares = self.parse_move_string(move_str)
        except Exception as e:
            self.get_logger().warning(f"Failed to parse selected move '{move_str}': {e}")
            return

        if len(squares) < 2:
            self.get_logger().warning(f"Move has fewer than 2 points: {move_str}")
            return

        if self.latest_piece_states is None:
            self.get_logger().warning("No piece states received yet.")
            return

        start_square = squares[0]
        path_squares = squares[1:]

        try:
            start_world = self.find_piece_world_at_square(*start_square)
        except Exception as e:
            self.get_logger().warning(
                f"Failed to find checker pose for start square {start_square}: {e}"
            )
            return

        path_world = [self.square_to_world(r, c) for r, c in path_squares]

        payload = {
            "start_square": list(start_square),
            "path_squares": [list(s) for s in path_squares],
            "start_world": list(start_world),
            "path_world": [list(p) for p in path_world],
        }

        out = String()
        out.data = json.dumps(payload)
        self.publisher.publish(out)

    def parse_move_string(self, move_str: str) -> List[Tuple[int, int]]:
        parts = [part.strip() for part in move_str.split("->")]
        coords: List[Tuple[int, int]] = []

        for part in parts:
            row_str, col_str = [x.strip() for x in part.split(",")]
            coords.append((int(row_str), int(col_str)))

        return coords

    def square_to_world(self, row: int, col: int) -> Tuple[float, float, float]:
        x = self.board_center_x - (self.board_size / 2.0) + (col + 0.5) * self.square_size
        y = self.board_center_y + (self.board_size / 2.0) - (row + 0.5) * self.square_size
        z = self.piece_z
        return (x, y, z)

    def world_to_square(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        col = int((x - (self.board_center_x - self.board_size / 2.0)) / self.square_size)
        row = int(((self.board_center_y + self.board_size / 2.0) - y) / self.square_size)

        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)

        return None

    def find_piece_world_at_square(self, row: int, col: int) -> Tuple[float, float, float]:
        for piece in self.latest_piece_states or []:
            try:
                position = piece["position"]
                x = float(position["x"])
                y = float(position["y"])
                z = float(position["z"])
            except (KeyError, TypeError, ValueError):
                continue

            square = self.world_to_square(x, y)
            if square == (row, col):
                return (x, y, z)

        raise RuntimeError(f"No checker found at square ({row}, {col}).")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveTargetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()