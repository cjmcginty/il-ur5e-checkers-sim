#!/usr/bin/env python3

# data_collection_node.py
# Records (observation, action) pairs for imitation learning
# Action comes from JointTrajectory controller commands

import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

import tf2_ros

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__("data_collection_node")
        
        # Parameters
        self.declare_parameter("output_dir", "datasets")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter(
            "joint_traj_cmd_topic",
            "/joint_trajectory_controller/joint_trajectory"
            )
        
        self.output_dir = self.get_parameter("output_dir").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.cmd_topic = self.get_parameter("joint_traj_cmd_topic").value
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State Storage
        self.latest_joint_state = None
        self.latest_goal_pose = None
        self.latest_gripper_state = 0.0
        
        self.joint_names = []
        self.name_to_index = {}
        
        self.recording = False
        self.obs_buffer = []
        self.act_buffer = []
        self.time_buffer = []
        
        # Subscribers
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            50
        )
        
        self.create_subscription(
            JointTrajectory,
            self.cmd_topic,
            self.joint_traj_callback,
            50
        )
        
        self.create_subscription(
            PoseStamped,
            "il_goal_pose",
            self.goal_pose_callback,
            10
        )
        
        self.create_subscription(
            Float32,
            "/gripper/state",
            self.gripper_state_callback,
            10
        )
        
        # Services
        self.create_service(Trigger, "/il/start_recording", self.start_recording)
        self.create_service(Trigger, "/il/stop_recording", self.stop_recording)
        
        self.get_logger().info("Data collection node ready.")
        self.get_logger().info(f"Listening to trajectory topic: {self.cmd_topic}")
        
    # Callbacks
    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        
        # Save joint ordering once
        if msg.name and not self.joint_names:
            self.joint_names = list(msg.name)
            self.name_to_index = {
                name: i for i, name in enumerate(self.joint_names)
            }
            self.get_logger().info("Joint order captured from /joint_states.")
            
    def goal_pose_callback(self, msg):
        self.latest_goal_pose = msg
    
    def gripper_state_callback(self, msg):
        self.latest_gripper_state = msg.data
        
    def joint_traj_callback(self, msg):
        if not self.recording:
            return
        
        if len(msg.points) == 0:
            return
        
        obs = self.build_observation()
        act = self.extract_action(msg)
        
        if obs is None or act is None:
            return
        
        t = self.get_clock().now().nanoseconds * 1e-9
        
        self.obs_buffer.append(obs)
        self.act_buffer.append(act)
        self.time_buffer.append(t)
        
    # Observation and Action Construction
    def build_observation(self):
        if self.latest_joint_state is None:
            return None
        
        # Joint positions
        q = np.array(self.latest_joint_state.position, dtype=np.float32)
        
        # End effector pose from TF
        ee_pose = self.lookup_ee_pose()
        
        # Goal pose (optional)
        if self.latest_goal_pose is not None:
            p = self.latest_goal_pose.pose
            goal = np.array([
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w
            ], dtype=np.float32)
        else:
            goal = np.zeros(7, dtype=np.float32)
            
        gripper = np.array([self.latest_gripper_state], dtype=np.float32)
        
        obs = np.concatenate([q, ee_pose, goal, gripper])
        return obs
    
    def extract_action(self, traj_msg):
        point = traj_msg.points[0]
        
        if not point.positions:
            return None
        
        cmd_names = traj_msg.joint_names
        cmd_positions = point.positions
        
        # Reorder command to match joint_states order
        if not self.joint_names:
            return np.array(cmd_positions, dtype=np.float32)
        
        action = np.zeros(len(self.joint_names), dtype=np.float32)
        
        for name, pos in zip(cmd_names, cmd_positions):
            if name in self.name_to_index:
                idx = self.name_to_index[name]
                action[idx] = pos
                
        return action
    
    def lookup_ee_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)
            )
            
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            
            return np.array([
                t.x, t.y, t.z,
                q.x, q.y, q.z, q.w
            ], dtype=np.float32)
        
        except Exception:
            # If TF not ready yet
            return np.zeros(7, dtype=np.float32)
        
    # Services
    def start_recording(self, request, response):
        if self.recording:
            response.success = False
            response.message = "Already recording."
            return response
        
        self.recording = True
        self.obs_buffer.clear()
        self.act_buffer.clear()
        self.time_buffer.clear()
        
        response.success = True
        response.message = "Recording started."
        self.get_logger().info("Recording started.")
        return response
    
    def stop_recording(self, request, response):
        if not self.recording:
            response.success = False
            response.message = "Not recording."
            return response
        
        self.recording = False
    
        if len(self.obs_buffer) == 0:
            response.success = False
            response.message = "No data recorded."
            return response
        
        observations = np.stack(self.obs_buffer)
        actions = np.stack(self.act_buffer)
        timestamps = np.array(self.time_buffer)
        
        filename = f"episode_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        path = os.path.join(self.output_dir, filename)
        
        np.savez_compressed(
            path,
            observations=observations,
            actions=actions,
            timestamps=timestamps,
            joint_names=np.array(self.joint_names, dtype=object)
        )
        
        response.success = True
        response.message = f"Saved episode to {path}"
        self.get_logger().info(response.message)
        
        return response
    
def main():
    rclpy.init()
    node = DataCollectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
