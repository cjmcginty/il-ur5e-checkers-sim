import json
import math
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String


class CheckersPieceManager(Node):
    def __init__(self) -> None:
        super().__init__("checkers_piece_manager")

        self.declare_parameter("game_event_topic", "/checkers/game_event")
        self.declare_parameter("piece_states_topic", "/checkers/piece_states")
        self.declare_parameter("world_name", "checkers_world")
        self.declare_parameter("board_center_x", 0.6)
        self.declare_parameter("board_center_y", 0.0)
        self.declare_parameter("board_size", 0.40)
        self.declare_parameter("piece_z", 0.03)

        self.game_event_topic = (
            self.get_parameter("game_event_topic").get_parameter_value().string_value
        )
        self.piece_states_topic = (
            self.get_parameter("piece_states_topic").get_parameter_value().string_value
        )
        self.world_name = (
            self.get_parameter("world_name").get_parameter_value().string_value
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
        self.latest_model_states: Optional[List[dict]] = None

        package_share = Path(get_package_share_directory("ur5e_checkers_bringup"))
        self.red_king_sdf = str(package_share / "models" / "red_king" / "model.sdf")
        self.black_king_sdf = str(package_share / "models" / "black_king" / "model.sdf")

        self.create_subscription(
            String,
            self.piece_states_topic,
            self.model_states_callback,
            10,
        )
        self.create_subscription(
            String,
            self.game_event_topic,
            self.game_event_callback,
            10,
        )

        self.get_logger().info("Checkers piece manager started.")

    def model_states_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_model_states = data
        except json.JSONDecodeError:
            pass

    def game_event_callback(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not isinstance(event, dict):
            return

        if event.get("type") == "capture":
            if self.latest_model_states is None:
                self.get_logger().warning("No piece states available yet.")
                return

            captured = event.get("captured")
            by = event.get("by")

            if not isinstance(captured, list):
                return
            if by not in ("r", "b"):
                return

            captured_color = "black" if by == "r" else "red"

            for square in captured:
                if (
                    not isinstance(square, list)
                    or len(square) != 2
                    or not isinstance(square[0], int)
                    or not isinstance(square[1], int)
                ):
                    continue

                row, col = square

                piece = self.find_piece_at_square(row, col, captured_color)
                if piece is None:
                    self.get_logger().warning(
                        f"Could not find captured {captured_color} piece at ({row}, {col})."
                    )
                    continue

                piece_name = piece["name"]

                if not self.delete_entity(piece_name):
                    self.get_logger().warning(f"Failed to delete captured piece {piece_name}")
                    continue

                self.get_logger().info(
                    f"Removed captured piece {piece_name} at ({row}, {col})"
                )

            return

        if event.get("type") != "promote":
            return

        if self.latest_model_states is None:
            self.get_logger().warning("No piece states available yet.")
            return

        color = event.get("color")
        row = event.get("row")
        col = event.get("col")

        if color not in ("red", "black"):
            return
        if not isinstance(row, int) or not isinstance(col, int):
            return

        piece = self.find_piece_at_square(row, col, color)
        if piece is None:
            self.get_logger().warning(
                f"Could not find {color} checker to promote at ({row}, {col})."
            )
            return

        old_name = piece["name"]
        suffix = old_name.split("_")[-1]
        new_name = f"{color}_king_{suffix}"

        if old_name.startswith(f"{color}_king_"):
            return

        x, y = self.square_to_world(row, col)

        if color == "red":
            sdf_file = self.red_king_sdf
        else:
            sdf_file = self.black_king_sdf

        if not self.delete_entity(old_name):
            self.get_logger().warning(f"Failed to delete {old_name}")
            return

        if not self.spawn_entity(new_name, sdf_file, x, y, self.piece_z):
            self.get_logger().warning(f"Failed to spawn {new_name}")
            return

        self.get_logger().info(f"Promoted {old_name} -> {new_name}")

    def find_piece_at_square(self, row: int, col: int, color: str) -> Optional[dict]:
        best_piece = None
        best_dist = float("inf")
        target_x, target_y = self.square_to_world(row, col)

        for piece in self.latest_model_states:
            try:
                name = piece["name"]
                position = piece["position"]
                x = float(position["x"])
                y = float(position["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if color == "red":
                if not name.startswith("red_checker_"):
                    continue
            else:
                if not name.startswith("black_checker_"):
                    continue

            dist = math.hypot(x - target_x, y - target_y)
            if dist < best_dist:
                best_dist = dist
                best_piece = piece

        return best_piece

    def square_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.board_center_x - (self.board_size / 2.0) + (col + 0.5) * self.square_size
        y = self.board_center_y + (self.board_size / 2.0) - (row + 0.5) * self.square_size
        return x, y

    def delete_entity(self, name: str) -> bool:
        cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "delete_entity",
            "--name",
            name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.get_logger().warning(result.stderr.strip() or result.stdout.strip())
            return False
        return True

    def spawn_entity(self, name: str, sdf_file: str, x: float, y: float, z: float) -> bool:
        cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            self.world_name,
            "-name",
            name,
            "-file",
            sdf_file,
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.get_logger().warning(result.stderr.strip() or result.stdout.strip())
            return False
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CheckersPieceManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()