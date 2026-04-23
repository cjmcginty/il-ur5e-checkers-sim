#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float64MultiArray

import tf2_ros
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCPolicyNode(Node):
    def __init__(self):
        super().__init__("bc_policy_node")

        self.declare_parameter("model_path", "models/bc_policy.pt")
        self.declare_parameter("cmd_topic", "/forward_position_controller/commands")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter(
            "controlled_joints",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )

        self.model_path = self.get_parameter("model_path").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.controlled_joints = list(
            self.get_parameter("controlled_joints").value
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_joint_state = None
        self.latest_goal_pose = None
        self.latest_gripper_state = 0.0
        self.joint_names = []
        self.name_to_index = {}

        self.sub_js = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 50
        )
        self.sub_goal = self.create_subscription(
            PoseStamped, "il_goal_pose", self.goal_pose_cb, 10
        )
        self.sub_gripper = self.create_subscription(
            Float32, "/gripper/state", self.gripper_state_cb, 10
        )

        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.obs_mean, self.obs_std, self.out_dim = self.load_model(self.model_path)

        period = 1.0 / max(self.rate_hz, 1e-3)
        self.create_timer(period, self.tick)

        self.get_logger().info(f"Loaded model: {self.model_path} on {self.device}")
        self.get_logger().info(f"Publishing to: {self.cmd_topic}")

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"model_path not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        in_dim = int(ckpt["in_dim"])
        out_dim = int(ckpt["out_dim"])
        hidden = int(ckpt["hidden"])
        depth = int(ckpt["depth"])

        model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden, depth=depth)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(self.device)

        obs_mean = ckpt["obs_mean"].astype(np.float32)
        obs_std = ckpt["obs_std"].astype(np.float32)
        obs_std[obs_std < 1e-6] = 1.0

        return model, obs_mean, obs_std, out_dim

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg
        if msg.name and not self.joint_names:
            self.joint_names = list(msg.name)
            self.name_to_index = {n: i for i, n in enumerate(self.joint_names)}
            self.get_logger().info("Joint order captured from /joint_states.")

    def goal_pose_cb(self, msg: PoseStamped):
        self.latest_goal_pose = msg

    def gripper_state_cb(self, msg: Float32):
        self.latest_gripper_state = msg.data

    def get_controlled_joint_positions(self):
        if self.latest_joint_state is None:
            return None
        if not self.name_to_index:
            return None

        q = []
        for name in self.controlled_joints:
            if name not in self.name_to_index:
                self.get_logger().warn(
                    f"Controlled joint '{name}' not found in /joint_states."
                )
                return None
            idx = self.name_to_index[name]
            q.append(self.latest_joint_state.position[idx])

        return np.array(q, dtype=np.float32)

    def lookup_ee_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)

    def build_observation(self):
        if self.latest_joint_state is None:
            return None

        q = self.get_controlled_joint_positions()
        if q is None:
            return None
            
        ee = self.lookup_ee_pose()

        if self.latest_goal_pose is not None:
            p = self.latest_goal_pose.pose
            goal = np.array([
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ], dtype=np.float32)
        else:
            goal = np.zeros(7, dtype=np.float32)

        gripper = np.array([self.latest_gripper_state], dtype=np.float32)

        return np.concatenate([q, ee, goal, gripper])

    def tick(self):
        obs = self.build_observation()
        if obs is None:
            return
        if not self.joint_names:
            return

        obs_n = (obs - self.obs_mean) / self.obs_std
        x = torch.from_numpy(obs_n).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            y = self.model(x).squeeze(0).cpu().numpy().astype(np.float32)

        if len(y) != len(self.controlled_joints):
            self.get_logger().error(
                f"Model output dim {len(y)} does not match joint count {len(self.joint_names)}"
            )
            return

        y = np.clip(y, -6.28, 6.28)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in y.tolist()]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = BCPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()