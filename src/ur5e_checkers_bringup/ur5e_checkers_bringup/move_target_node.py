import json
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MoveTargetNode(Node):
    def __init__(self) -> None:
        super().__init__("move_target_node")

        self.declare_parameter("robot_move_topic", "/checkers/robot_move_command")
        self.declare_parameter("player_move_topic", "/checkers/selected_player_move")
        self.declare_parameter("piece_states_topic", "/checkers/piece_states")
        self.declare_parameter("move_target_topic", "/checkers/move_targets")
        self.declare_parameter("player_move_target_topic", "/checkers/player_move_targets")
        self.declare_parameter("board_center_x", 0.6)
        self.declare_parameter("board_center_y", 0.0)
        self.declare_parameter("board_size", 0.40)
        self.declare_parameter("piece_z", 0.03)

        self.robot_move_topic = self.get_parameter("robot_move_topic").value
        self.player_move_topic = self.get_parameter("player_move_topic").value
        self.piece_states_topic = self.get_parameter("piece_states_topic").value
        self.move_target_topic = self.get_parameter("move_target_topic").value
        self.player_move_target_topic = self.get_parameter("player_move_target_topic").value

        self.board_center_x = self.get_parameter("board_center_x").value
        self.board_center_y = self.get_parameter("board_center_y").value
        self.board_size = self.get_parameter("board_size").value
        self.piece_z = self.get_parameter("piece_z").value

        self.square_size = self.board_size / 8.0
        self.latest_piece_states: Optional[List[dict]] = None

        self.create_subscription(
            String,
            self.robot_move_topic,
            self.robot_move_callback,
            10,
        )

        self.create_subscription(
            String,
            self.player_move_topic,
            self.player_move_callback,
            10,
        )

        self.create_subscription(
            String,
            self.piece_states_topic,
            self.piece_states_callback,
            10,
        )

        self.robot_target_pub = self.create_publisher(
            String,
            self.move_target_topic,
            10,
        )

        self.player_target_pub = self.create_publisher(
            String,
            self.player_move_target_topic,
            10,
        )

        self.get_logger().info("Move target node started.")
        self.get_logger().info(f"Robot move input: {self.robot_move_topic}")
        self.get_logger().info(f"Robot target output: {self.move_target_topic}")
        self.get_logger().info(f"Player move input: {self.player_move_topic}")
        self.get_logger().info(f"Player target output: {self.player_move_target_topic}")

    def piece_states_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_piece_states = data
        except json.JSONDecodeError:
            self.get_logger().warning("Failed to parse /checkers/piece_states JSON.")

    def robot_move_callback(self, msg: String) -> None:
        self.publish_move_target(msg, self.robot_target_pub, source_name="robot")

    def player_move_callback(self, msg: String) -> None:
        self.publish_move_target(msg, self.player_target_pub, source_name="player")

    def publish_move_target(self, msg: String, publisher, source_name: str) -> None:
        move_str = msg.data.strip()

        try:
            squares = self.parse_move_string(move_str)
        except Exception as e:
            self.get_logger().warning(
                f"Failed to parse {source_name} move '{move_str}': {e}"
            )
            return

        if len(squares) < 2:
            self.get_logger().warning(f"{source_name} move has fewer than 2 points: {move_str}")
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
                f"Failed to find checker pose for {source_name} start square "
                f"{start_square}: {e}"
            )
            return

        path_world = [self.square_to_world(r, c) for r, c in path_squares]

        payload = {
            "start_square": list(start_square),
            "path_squares": [list(square) for square in path_squares],
            "start_world": list(start_world),
            "path_world": [list(point) for point in path_world],
            "source": source_name,
            "move": move_str,
        }

        out = String()
        out.data = json.dumps(payload)
        publisher.publish(out)

        self.get_logger().info(
            f"Published {source_name} move target for: {move_str}"
        )

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