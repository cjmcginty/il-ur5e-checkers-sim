#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
        self.declare_parameter("cmd_topic", "/ur5e_arm_controller/joint_trajectory")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("traj_time", 0.25)

        self.model_path = self.get_parameter("model_path").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.traj_time = float(self.get_parameter("traj_time").value)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_joint_state = None
        self.joint_names = []
        self.name_to_index = {}

        self.sub_js = self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 50)
        self.pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)

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
        q = np.array(self.latest_joint_state.position, dtype=np.float32)
        ee = self.lookup_ee_pose()
        goal = np.zeros(7, dtype=np.float32)
        gripper = np.zeros(1, dtype=np.float32)
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

        # y is in joint_states order already (because actions were saved that way)
        msg = JointTrajectory()
        msg.joint_names = list(self.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in y.tolist()]
        pt.time_from_start.sec = int(self.traj_time)
        pt.time_from_start.nanosec = int((self.traj_time - int(self.traj_time)) * 1e9)

        msg.points = [pt]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = BCPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()