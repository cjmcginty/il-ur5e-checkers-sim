import json
import math
import subprocess

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MagicPieceMoverNode(Node):
    def __init__(self):
        super().__init__("magic_piece_mover_node")

        self.declare_parameter("world_name", "checkers_world")
        self.declare_parameter("move_targets_topic", "/checkers/player_move_targets")
        self.declare_parameter("piece_states_topic", "/checkers/piece_states")
        self.declare_parameter("piece_z", 0.03)

        self.world_name = self.get_parameter("world_name").value
        self.piece_z = self.get_parameter("piece_z").value

        self.latest_piece_states = []

        self.create_subscription(
            String,
            self.get_parameter("piece_states_topic").value,
            self.piece_states_callback,
            10,
        )

        self.create_subscription(
            String,
            self.get_parameter("move_targets_topic").value,
            self.move_targets_callback,
            10,
        )

        self.get_logger().info("Magic piece mover started.")

    def piece_states_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_piece_states = data
        except json.JSONDecodeError:
            pass

    def move_targets_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Bad move target JSON.")
            return

        start_world = data.get("start_world")
        path_world = data.get("path_world")

        if not start_world or not path_world:
            self.get_logger().warning("Move target missing start_world or path_world.")
            return

        piece = self.find_piece_near(start_world[0], start_world[1])
        if piece is None:
            self.get_logger().warning("Could not find piece to move.")
            return

        name = piece["name"]
        final_x, final_y, _ = path_world[-1]

        if self.set_entity_pose(name, final_x, final_y, self.piece_z):
            self.get_logger().info(
                f"Moved existing {name} to ({final_x:.3f}, {final_y:.3f}, {self.piece_z:.3f})"
            )

    def find_piece_near(self, x, y):
        best = None
        best_dist = float("inf")

        for piece in self.latest_piece_states:
            try:
                px = float(piece["position"]["x"])
                py = float(piece["position"]["y"])
                name = piece["name"]
            except Exception:
                continue

            if not (
                name.startswith("red_checker_")
                or name.startswith("black_checker_")
                or name.startswith("red_king_")
                or name.startswith("black_king_")
            ):
                continue

            dist = math.hypot(px - x, py - y)
            if dist < best_dist:
                best_dist = dist
                best = piece

        if best_dist > 0.08:
            self.get_logger().warning(
                f"Nearest checker is too far from requested start square: {best_dist:.3f} m"
            )
            return None

        return best

    def set_entity_pose(self, name, x, y, z):
        req = (
            f'name: "{name}" '
            f'position {{ x: {x} y: {y} z: {z} }} '
            f'orientation {{ x: 0 y: 0 z: 0 w: 1 }}'
        )

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            req,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.get_logger().warning(
                result.stderr.strip() or result.stdout.strip() or "Failed to set entity pose."
            )
            return False

        return True


def main(args=None):
    rclpy.init(args=args)
    node = MagicPieceMoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()