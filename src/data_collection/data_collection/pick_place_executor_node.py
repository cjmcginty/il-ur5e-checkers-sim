#!/usr/bin/env python3

import math
import copy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand
from std_msgs.msg import String

import tf2_ros


class PickPlaceExecutorNode(Node):
    def __init__(self):
        super().__init__("pick_place_executor_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("pose_cmd_topic", "/servo_node/pose_target_cmds")
        self.declare_parameter("robot_move_done_topic", "/checkers/robot_move_done")
        self.declare_parameter(
            "gripper_action_name",
            "/gripper_position_controller/gripper_cmd",
        )

        self.declare_parameter("position_tolerance", 0.015)
        self.declare_parameter("publish_hz", 20.0)

        self.declare_parameter("open_gripper_value", 0.0)
        self.declare_parameter("closed_gripper_value", 0.79)
        self.declare_parameter("gripper_max_effort", 50.0)
        self.declare_parameter("gripper_wait_ticks", 10)

        self.declare_parameter("above_height", 0.25)
        self.declare_parameter("grasp_height", 0.18)
        self.declare_parameter("place_height", 0.18)
        self.declare_parameter("max_move_ticks", 100)

        self.max_move_ticks = int(self.get_parameter("max_move_ticks").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.pose_cmd_topic = self.get_parameter("pose_cmd_topic").value
        self.robot_move_done_topic = self.get_parameter("robot_move_done_topic").value
        self.gripper_action_name = self.get_parameter("gripper_action_name").value

        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.open_gripper_value = float(self.get_parameter("open_gripper_value").value)
        self.closed_gripper_value = float(self.get_parameter("closed_gripper_value").value)
        self.gripper_max_effort = float(self.get_parameter("gripper_max_effort").value)
        self.gripper_wait_ticks = int(self.get_parameter("gripper_wait_ticks").value)

        self.above_height = float(self.get_parameter("above_height").value)
        self.grasp_height = float(self.get_parameter("grasp_height").value)
        self.place_height = float(self.get_parameter("place_height").value)

        self.latest_piece_pose = None
        self.latest_goal_pose = None

        self.active = False
        self.phase_index = 0
        self.wait_count = 0
        self.phases = []
        self.move_count = 0

        self.gripper_goal_active = False
        self.gripper_goal_done = False
        self.last_pose_target = None

        self.locked_orientation = None

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

        self.robot_move_done_pub = self.create_publisher(
            String,
            self.robot_move_done_topic,
            10,
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            self.gripper_action_name,
        )

        period = 1.0 / max(self.publish_hz, 1e-3)
        self.create_timer(period, self.tick)

        self.get_logger().info("Pick-place executor node ready.")
        self.get_logger().info(f"Publishing pose targets to: {self.pose_cmd_topic}")
        self.get_logger().info(f"Publishing robot move done to: {self.robot_move_done_topic}")
        self.get_logger().info(f"Sending gripper actions to: {self.gripper_action_name}")

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

        if self.locked_orientation is None:
            self.locked_orientation = self.lookup_ee_orientation()

            if self.locked_orientation is None:
                self.get_logger().warn(
                    "Could not lock orientation yet. Waiting for TF..."
                )
                return

            self.get_logger().info(
                "Locked persistent end-effector orientation. "
                "This orientation will be reused for all pick-place moves."
            )

        self.build_phases()
        self.active = True
        self.phase_index = 0
        self.wait_count = 0
        self.move_count = 0
        self.gripper_goal_active = False
        self.gripper_goal_done = False
        self.last_pose_target = self.make_current_hold_pose()

        self.get_logger().info("Starting new pick-place move.")

    def build_phases(self):
        piece = copy.deepcopy(self.latest_piece_pose)
        goal = copy.deepcopy(self.latest_goal_pose)

        piece.header.frame_id = self.base_frame
        goal.header.frame_id = self.base_frame

        piece.pose.orientation = copy.deepcopy(self.locked_orientation)
        goal.pose.orientation = copy.deepcopy(self.locked_orientation)

        above_piece = self.set_z(piece, self.above_height)
        at_piece = self.set_z(piece, self.grasp_height)

        above_goal = self.set_z(goal, self.above_height)
        at_goal = self.set_z(goal, self.place_height)

        self.phases = [
            ("gripper", self.open_gripper_value, "Open gripper before starting"),
            ("wait", None, "Extra wait after opening gripper"),

            ("move", above_piece, "Move above piece"),
            ("move", at_piece, "Lower to piece"),

            ("gripper", self.closed_gripper_value, "Close gripper"),
            ("wait", None, "Extra wait after grasp"),

            ("move", above_piece, "Lift piece"),
            ("move", above_goal, "Move above goal"),
            ("move", at_goal, "Lower to goal"),

            ("gripper", self.open_gripper_value, "Open gripper"),
            ("wait", None, "Extra wait after release"),

            ("move", above_goal, "Lift away"),
        ]

    def set_z(self, pose_msg: PoseStamped, z_value: float):
        out = copy.deepcopy(pose_msg)
        out.pose.position.z = z_value
        out.pose.orientation = copy.deepcopy(self.locked_orientation)
        return out

    def make_current_hold_pose(self):
        ee_position = self.lookup_ee_position()

        if ee_position is None or self.locked_orientation is None:
            return None

        hold_pose = PoseStamped()
        hold_pose.header.frame_id = self.base_frame
        hold_pose.header.stamp = self.get_clock().now().to_msg()

        hold_pose.pose.position.x = ee_position[0]
        hold_pose.pose.position.y = ee_position[1]
        hold_pose.pose.position.z = ee_position[2]
        hold_pose.pose.orientation = copy.deepcopy(self.locked_orientation)

        return hold_pose

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

    def lookup_ee_orientation(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )

            return copy.deepcopy(tf_msg.transform.rotation)

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

    def publish_pose_target(self, pose_msg: PoseStamped):
        msg = copy.deepcopy(pose_msg)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.orientation = copy.deepcopy(self.locked_orientation)

        self.pose_pub.publish(msg)
        self.last_pose_target = copy.deepcopy(msg)

    def hold_last_pose_target(self):
        if self.last_pose_target is None:
            self.last_pose_target = self.make_current_hold_pose()

        if self.last_pose_target is not None:
            msg = copy.deepcopy(self.last_pose_target)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            msg.pose.orientation = copy.deepcopy(self.locked_orientation)
            self.pose_pub.publish(msg)

    def send_gripper_goal(self, position: float):
        if not self.gripper_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn(
                f"Gripper action server not available: {self.gripper_action_name}"
            )
            return False

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)
        goal_msg.command.max_effort = self.gripper_max_effort

        self.get_logger().info(
            f"Sending gripper goal: position={position:.3f}, "
            f"max_effort={self.gripper_max_effort:.1f}"
        )

        self.gripper_goal_active = True
        self.gripper_goal_done = False

        send_future = self.gripper_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.gripper_goal_response_cb)

        return True

    def gripper_goal_response_cb(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Gripper goal was rejected.")
            self.gripper_goal_active = False
            self.gripper_goal_done = True
            return

        self.get_logger().info("Gripper goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.gripper_result_cb)

    def gripper_result_cb(self, future):
        try:
            result = future.result().result
            self.get_logger().info(
                f"Gripper result: position={result.position:.3f}, "
                f"effort={result.effort:.3f}, "
                f"stalled={result.stalled}, "
                f"reached_goal={result.reached_goal}"
            )
        except Exception as e:
            self.get_logger().warn(f"Could not read gripper result: {e}")

        self.gripper_goal_active = False
        self.gripper_goal_done = True

    def publish_robot_move_done(self):
        msg = String()
        msg.data = "done"
        self.robot_move_done_pub.publish(msg)
        self.get_logger().info("Published robot move done.")

    def advance_phase(self):
        self.phase_index += 1
        self.wait_count = 0
        self.move_count = 0
        self.gripper_goal_active = False
        self.gripper_goal_done = False

        if self.phase_index >= len(self.phases):
            self.get_logger().info("Pick-place move complete.")

            self.publish_robot_move_done()

            self.active = False
            self.latest_piece_pose = None
            self.latest_goal_pose = None
            self.last_pose_target = None
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
            self.move_count += 1

            dist = self.distance_to_pose(target)
            if dist is None:
                return

            if dist < self.position_tolerance:
                self.get_logger().info(
                    f"Reached phase: {description}, distance={dist:.4f}"
                )
                self.advance_phase()
                return

            if self.move_count >= self.max_move_ticks:
                self.get_logger().warn(
                    f"Timed out on phase '{description}', distance={dist:.4f}. "
                    "Advancing anyway."
                )
                self.advance_phase()

        elif phase_type == "gripper":
            self.hold_last_pose_target()

            if not self.gripper_goal_active and not self.gripper_goal_done:
                sent = self.send_gripper_goal(target)

                if sent:
                    self.get_logger().info(f"Gripper command sent: {description}")

            if self.gripper_goal_done:
                self.get_logger().info(f"Gripper phase complete: {description}")
                self.advance_phase()

        elif phase_type == "wait":
            self.hold_last_pose_target()

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