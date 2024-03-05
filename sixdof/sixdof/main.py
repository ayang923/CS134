import rclpy

from sixdof.TransformHelpers import *

from sixdof.trajnode import TrajectoryNode
from sixdof.detnode import DetectorNode
from sixdof.game import GameNode

def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the Trajectory node.
    traj_node = TrajectoryNode('traj')
    detect_node = DetectorNode('detect')
    game_node = GameNode('game')

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(traj_node)
    executor.add_node(detect_node)
    executor.add_node(game_node)
    
    # Spin the node until interrupted.
    executor.spin()

    # Shutdown the node and ROS.
    rclpy.shutdown()
    traj_node.destroy_node()
    detect_node.destroy_node()
    game_node.destroy_node()
    executor.shutdown()

if __name__ == "__main__":
    main()