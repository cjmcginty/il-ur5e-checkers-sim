#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


class RobotDescriptionPublisher(Node):
    def __init__(self):
        super().__init__("robot_description_publisher")


        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(String, "/robot_description", qos)


        # Publish once immediately, then a few more times to avoid startup races.
        self.desc = self.declare_parameter("robot_description", "").value
        if not self.desc:
            raise RuntimeError("robot_description parameter is empty")

        self.count = 0
        self.timer = self.create_timer(0.2, self._tick)

    def _tick(self):
        msg = String()
        msg.data = self.desc
        self.pub.publish(msg)
        self.count += 1
        if self.count >= 10:
            self.timer.cancel()
            self.get_logger().info("Published /robot_description (x10); done.")

def main():
    rclpy.init()
    node = RobotDescriptionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
