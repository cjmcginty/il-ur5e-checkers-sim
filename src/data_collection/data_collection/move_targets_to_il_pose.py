#!/usr/bin/env python3

import json

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


class MoveTargetsToILPose(Node):
    def __init__(self):
        super().__init__("move_targets_to_il_pose")

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("piece_z_offset", 0.0)
        self.declare_parameter("goal_z_offset", 0.0)

        self.frame_id = self.get_parameter("frame_id").value
        self.piece_z_offset = float(self.get_parameter("piece_z_offset").value)
        self.goal_z_offset = float(self.get_parameter("goal_z_offset").value)

        self.sub = self.create_subscription(
            String,
            "/checkers/move_targets",
            self.move_targets_cb,
            10,
        )

        self.piece_pub = self.create_publisher(
            PoseStamped,
            "/il_piece_pose",
            10,
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/il_goal_pose",
            10,
        )

        self.get_logger().info("move_targets_to_il_pose node ready.")

    def make_pose(self, xyz, z_offset=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2]) + z_offset

        # Neutral orientation
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        return msg

    def move_targets_cb(self, msg):
        try:
            data = json.loads(msg.data)

            start_world = data["start_world"]
            path_world = data["path_world"]

            if not path_world:
                self.get_logger().warn("Received move target with empty path_world.")
                return

            goal_world = path_world[-1]

            piece_pose = self.make_pose(start_world, self.piece_z_offset)
            goal_pose = self.make_pose(goal_world, self.goal_z_offset)

            self.piece_pub.publish(piece_pose)
            self.goal_pub.publish(goal_pose)

        except Exception as e:
            self.get_logger().warn(f"Failed to parse /checkers/move_targets: {e}")


def main():
    rclpy.init()
    node = MoveTargetsToILPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()