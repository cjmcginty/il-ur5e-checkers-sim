import rclpy
from rclpy.node import Node


class DQNPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("dqn_policy_node")
        self.get_logger().info("DQN policy node placeholder started.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DQNPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()