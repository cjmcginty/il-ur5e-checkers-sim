#!/usr/bin/env python3

import math
import copy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

import tf2_ros


class PickPlaceExecutorNode(Node):
    def __init__(self):
        super().__init__("pick_place_executor_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("pose_cmd_topic", "/servo_node/pose_target_cmds")
        self.declare_parameter("gripper_cmd_topic", "/gripper_position_controller/commands")

        self.declare_parameter("lift_height", 0.12)
        self.declare_parameter("place_height_offset", 0.0)
        self.declare_parameter("position_tolerance", 0.015)
        self.declare_parameter("publish_hz", 20.0)

        self.declare_parameter("open_gripper_value", 0.0)
        self.declare_parameter("closed_gripper_value", 0.75)
        self.declare_parameter("gripper_wait_ticks", 20)

        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.pose_cmd_topic = self.get_parameter("pose_cmd_topic").value
        self.gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").value

        self.lift_height = float(self.get_parameter("lift_height").value)
        self.place_height_offset = float(self.get_parameter("place_height_offset").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.open_gripper_value = float(self.get_parameter("open_gripper_value").value)
        self.closed_gripper_value = float(self.get_parameter("closed_gripper_value").value)
        self.gripper_wait_ticks = int(self.get_parameter("gripper_wait_ticks").value)

        self.latest_piece_pose = None
        self.latest_goal_pose = None

        self.active = False
        self.phase_index = 0
        self.wait_count = 0
        self.phases = []

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.piece_sub = self.create_subscription(
            PoseStamped,
            "/il_piece_pose",
            self.piece_pose_cb,
            10,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/il_goal_pose",
            self.goal_pose_cb,
            10,
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.pose_cmd_topic,
            10,
        )

        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            self.gripper_cmd_topic,
            10,
        )

        period = 1.0 / max(self.publish_hz, 1e-3)
        self.create_timer(period, self.tick)

        self.get_logger().info("Pick-place executor node ready.")
        self.get_logger().info(f"Publishing pose targets to: {self.pose_cmd_topic}")
        self.get_logger().info(f"Publishing gripper commands to: {self.gripper_cmd_topic}")

    def piece_pose_cb(self, msg: PoseStamped):
        self.latest_piece_pose = msg
        self.try_start_new_move()

    def goal_pose_cb(self, msg: PoseStamped):
        self.latest_goal_pose = msg
        self.try_start_new_move()

    def try_start_new_move(self):
        if self.active:
            return

        if self.latest_piece_pose is None or self.latest_goal_pose is None:
            return

        self.build_phases()
        self.active = True
        self.phase_index = 0
        self.wait_count = 0

        self.get_logger().info("Starting new pick-place move.")

    def build_phases(self):
        piece = copy.deepcopy(self.latest_piece_pose)
        goal = copy.deepcopy(self.latest_goal_pose)

        piece.header.frame_id = self.base_frame
        goal.header.frame_id = self.base_frame

        above_piece = self.offset_z(piece, self.lift_height)
        at_piece = self.offset_z(piece, 0.0)

        above_goal = self.offset_z(goal, self.lift_height)
        at_goal = self.offset_z(goal, self.place_height_offset)

        self.phases = [
            ("move", above_piece, "Move above piece"),
            ("move", at_piece, "Lower to piece"),
            ("gripper", self.closed_gripper_value, "Close gripper"),
            ("wait", None, "Wait after grasp"),
            ("move", above_piece, "Lift piece"),
            ("move", above_goal, "Move above goal"),
            ("move", at_goal, "Lower to goal"),
            ("gripper", self.open_gripper_value, "Open gripper"),
            ("wait", None, "Wait after release"),
            ("move", above_goal, "Lift away"),
        ]

    def offset_z(self, pose_msg: PoseStamped, dz: float):
        out = copy.deepcopy(pose_msg)
        out.pose.position.z += dz

        # Keep a neutral end-effector orientation for now.
        # Later you may want to replace this with your real grasp orientation.
        out.pose.orientation.x = 0.0
        out.pose.orientation.y = 0.0
        out.pose.orientation.z = 0.0
        out.pose.orientation.w = 1.0

        return out

    def publish_gripper(self, value: float):
        msg = Float64MultiArray()
        msg.data = [float(value)]
        self.gripper_pub.publish(msg)

    def publish_pose_target(self, pose_msg: PoseStamped):
        msg = copy.deepcopy(pose_msg)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        self.pose_pub.publish(msg)

    def lookup_ee_position(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )

            t = tf_msg.transform.translation
            return (t.x, t.y, t.z)

        except Exception:
            return None

    def distance_to_pose(self, pose_msg: PoseStamped):
        ee = self.lookup_ee_position()
        if ee is None:
            return None

        dx = ee[0] - pose_msg.pose.position.x
        dy = ee[1] - pose_msg.pose.position.y
        dz = ee[2] - pose_msg.pose.position.z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def advance_phase(self):
        self.phase_index += 1
        self.wait_count = 0

        if self.phase_index >= len(self.phases):
            self.get_logger().info("Pick-place move complete.")
            self.active = False
            self.latest_piece_pose = None
            self.latest_goal_pose = None
            return

        desc = self.phases[self.phase_index][2]
        self.get_logger().info(f"Next phase: {desc}")

    def tick(self):
        if not self.active:
            return

        if self.phase_index >= len(self.phases):
            self.active = False
            return

        phase_type, target, description = self.phases[self.phase_index]

        if phase_type == "move":
            self.publish_pose_target(target)

            dist = self.distance_to_pose(target)
            if dist is None:
                return

            if dist < self.position_tolerance:
                self.get_logger().info(
                    f"Reached phase: {description}, distance={dist:.4f}"
                )
                self.advance_phase()

        elif phase_type == "gripper":
            self.publish_gripper(target)
            self.get_logger().info(f"Gripper command: {description}")
            self.advance_phase()

        elif phase_type == "wait":
            self.wait_count += 1
            if self.wait_count >= self.gripper_wait_ticks:
                self.advance_phase()


def main():
    rclpy.init()
    node = PickPlaceExecutorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()