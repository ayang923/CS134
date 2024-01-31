import numpy as np
import rclpy

import cv2, cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import JointState, Image
from geometry_msgs.msg  import Point, Pose, Quaternion

from sixdof.TrajectoryUtils import goto, goto5
from sixdof.TransformHelpers import *

from sixdof.states import Tasks, TaskHandler
from sixdof.nodes import TrajectoryNode, DetectorNode

from enum import Enum

def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the Trajectory node.
    traj_node = TrajectoryNode('traj')
    detect_node = DetectorNode("detect")

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(traj_node)
    executor.add_node(detect_node)
    # Spin the node until interrupted.
    executor.spin()

    # Shutdown the node and ROS.
    rclpy.shutdown()
    traj_node.destroy_node()
    detect_node.destroy_node()
    executor.shutdown()

if __name__ == "__main__":
    main()