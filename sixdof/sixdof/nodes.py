import numpy as np
import rclpy

import cv2, cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import JointState, Image
from geometry_msgs.msg  import Point, Pose, Quaternion, PoseArray
from std_msgs.msg       import UInt8MultiArray

from sixdof.TrajectoryUtils import goto, goto5
from sixdof.TransformHelpers import *

from sixdof.states import Tasks, GamePiece, TaskHandler, JOINT_NAMES
from sixdof.game import GameDriver

from enum import Enum

RATE = 100.0            # Hertz

nan = float("nan")

GREEN_CHECKER_LIMITS = np.array(([10, 70], [30, 90], [10, 70]))
BROWN_CHECKER_LIMITS = np.array(([80, 120], [140, 200], [180, 255]))
YELLOW_BOARD_LIMITS = np.array([[80, 120], [100, 220], [140, 180]])
RED_BOARD_LIMITS = np.array([[100, 140], [160, 240], [100, 180]])

class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.jointnames = JOINT_NAMES

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.actpos = self.position0
        self.get_logger().info("Initial positions: %r" % self.position0)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 100) # /joint_commands publisher

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Subscribers:
        # /joint_states from Hebi node
        self.fbksub = self.create_subscription(JointState, '/joint_states',
                                               self.recvfbk, 100)
        # /boardpose from Detector
        self.sub_board = self.create_subscription(PoseArray, '/boardpose',
                                                  self.recvboard, 3)

        # Poses of all detected green checkers, /green from Detector
        self.sub_green = self.create_subscription(PoseArray, '/green',
                                                  self.recvgreen, 3)

        # Poses of all detected brown checkers, /brown from Detector
        self.sub_brown = self.create_publisher(PoseArray, '/brown',
                                               self.recvbrown, 3)

        # /dice Unsigned int array with value of two detected dice from Detector
        self.sub_dice = self.create_publisher(UInt8MultiArray, '/dice',
                                              self.recvdice, 3)

        # creates task handler for robot
        self.task_handler = TaskHandler(self, np.array(self.position0).reshape(-1, 1))

        self.task_handler.add_state(Tasks.INIT)

        # game driver for trajectory node
        self.game_driver = GameDriver(self, self.task_handler)

        # Stored gamestate info
        self.board1pose = 0
        self.board2pose = 0
        # Initial gamestate area assumes setup for beginning of game
        # each element indicates [num_green, num_brown]
        # beginning to end of array progresses ccw from robot upper right
        self.gamestate = np.array([[2,0], [0,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0], [0,3], [0,0], [0,0], [0,0], [5,0],
                                   [0,5], [0,0], [0,0], [0,0], [3,0], [0,0],
                                   [5,0], [0,0], [0,0], [0,0], [0,0], [0,2]])
        # self.recvgreen populates these arrays with detected green checker pos
        self.greenpos = np.array([[]])
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = np.array([[]])
        # self.recvdice populates this with detected [die1_int, die2_int]
        self.dice = np.array([])

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds



    # Called repeatedly by incoming messages - do nothing for now
    def recvfbk(self, fbkmsg):
        self.actpos = list(fbkmsg.position)

    def recvboard(self, msg):
        pass

    def recvgreen(self, msg):
        pass

    def recvbrown(self, msg):
        pass

    def recvdice(self, msg):
        pass    

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.timer.destroy()
        self.destroy_node()
        
    # Grab a single feedback - do not call this repeatedly.
    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        # Return the values.
        return self.grabpos

    def get_time(self):
        return 1e-9 * self.get_clock().now().nanoseconds - self.start_time

    # Receive new command update from trajectory - called repeatedly by incoming messages.
    def update(self):
        self.t = self.get_time()
         # Compute the desired joint positions and velocities for this time.
        desired = self.task_handler.evaluate_task(self.t, 1 / RATE)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired
        
        # Check the results.
        if not (isinstance(q, list) and isinstance(qdot, list)):
            self.get_logger().warn("(q) and (qdot) must be python lists!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(q) must be same length as jointnames!")
            return
        if not (len(q) == len(qdot)):
            self.get_logger().warn("(qdot) must be same length as (q)!")
            return
        if not (isinstance(q[0], float) and isinstance(qdot[0], float)):
            self.get_logger().warn("Flatten NumPy arrays before making lists!")
            return
            
        return (q, qdot)
    
    def gravitycomp(self, q):
        tau_elbow = -5.75 * np.sin(-q[1] + q[2])
        tau_shoulder = -tau_elbow + 9.5 * np.sin(-q[1])
        return (tau_shoulder, tau_elbow)

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        (q, qdot) = self.update()
        (tau_shoulder, tau_elbow) = self.gravitycomp(self.actpos)
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = self.jointnames
        #self.cmdmsg.position     = [nan, nan, nan, nan, nan, nan] # uncomment for gravity comp test
        #self.cmdmsg.velocity     = [nan, nan, nan, nan, nan, nan]
        self.cmdmsg.position     = q # comment for gravity comp
        self.cmdmsg.velocity     = qdot
        self.cmdmsg.effort       = [0.0, tau_shoulder, tau_elbow, 0.0, 0.0, 0.0]
        self.cmdpub.publish(self.cmdmsg)


class DetectorNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.green_checker_limits = GREEN_CHECKER_LIMITS
        self.brown_checker_limits = BROWN_CHECKER_LIMITS

        #subscribers for raw images
        self.rcvtopimg = self.create_subscription(Image, '/usb_cam/image_raw',
                                                 self.process_top_images, 1)
        
        self.rcvtipimg = self.create_subscription(Image, 'tip_cam/image_raw',
                                                  self.process_tip_images, 1)
        
        # Publishers for detected features:
        # Poses of two halves of game board
        self.pubboard = self.create_publisher(PoseArray, '/boardpose', 3)

        # Poses of all detected green checkers
        self.pub_green = self.create_publisher(PoseArray, '/green', 3)

        # Poses of all detected brown checkers
        self.pub_brown = self.create_publisher(PoseArray, '/brown', 3)

        self.pub_dice = self.create_publisher(UInt8MultiArray, '/dice', 3)
        
        #publisher for debugging images
        self.pubmask = self.create_publisher(Image, '/usb_cam/binary',    3)


        self.bridge = cv_bridge.CvBridge()

'''
def process_top(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        #hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(hsv, self.checker_limits[:, 0], self.checker_limits[:, 1])

        # Erode and Dilate. Definitely adjust the iterations!
        iter = 4
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode( binary, None, iterations=iter)

        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        if len(contours) > 0:
            # Pick the largest contour.
            contour = max(contours, key=cv2.contourArea)
            x0, y0 = 0.0, 0.387

            ((ur, vr), (w, h), theta) = cv2.minAreaRect(contour)
            xy = self.pixelToWorld(frame, ur, vr, x0, y0)

            if xy is not None:
                (x, y) = xy
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = 0.006

                self.pubpoints.publish(point_msg)
'''

    def process_top_images(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")


        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        #hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue


        (H, W, D) = blurred.shape
        uc = W//2
        vc = H//2

        blurred = cv2.line(blurred, (uc,0), (uc,H-1), (255, 255, 255), 5)
        blurred = cv2.line(blurred, (0,vc), (W-1,vc), (255, 255, 255), 5)

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary_yellow = cv2.inRange(hsv, YELLOW_BOARD_LIMITS[:, 0], YELLOW_BOARD_LIMITS[:, 1])
        binary_red = cv2.inRange(hsv, RED_BOARD_LIMITS[:, 0], RED_BOARD_LIMITS[:, 1])

        binary = cv2.bitwise_or(binary_yellow, binary_red)
        
        # trying to find board, dilating a lot to fill in boundary
        binary_board = cv2.erode(binary, None, iterations=2)
        binary_board = cv2.dilate(binary_board, None, iterations=8)

        contours_board, _ = cv2.findContours(binary_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # finding board contour
        if len(contours_board) > 0:
            board_msk = np.zeros(binary.shape)
            # Pick the largest contour.
            contour_board = max(contours_board, key=cv2.contourArea)
            cv2.drawContours(board_msk,[contour_board], 0, 1, -1)

            # filters out everythign but board
            binary = cv2.bitwise_and(binary, binary, mask=board_msk.astype('uint8'))

            # detecting checkers
            binary_checkers = cv2.dilate(binary, None, iterations=2)

        # binary = cv2.erode( binary, None, iterations=iter)
        self.get_logger().info(
            "Center pixel HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        self.pubmask.publish(self.bridge.cv2_to_imgmsg(binary_checkers))

        # Erode and Dilate. Definitely adjust the iterations!


        # # Find contours in the mask and initialize the current
        # # (x, y) center of the ball
        # (contours, hierarchy) = cv2.findContours(
        #     binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def process_tip_images(self, msg):
        pass


# helper functions
def pixelToWorld(image, u, v, x0, y0, annotateImage=True):
    '''
    Convert the (u,v) pixel position into (x,y) world coordinates
    Inputs:
        image: The image as seen by the camera
        u:     The horizontal (column) pixel coordinate
        v:     The vertical (row) pixel coordinate
        x0:    The x world coordinate in the center of the marker paper
        y0:    The y world coordinate in the center of the marker paper
        annotateImage: Annotate the image with the marker information

    Outputs:
        point: The (x,y) world coordinates matching (u,v), or None

    Return None for the point if not all the Aruco markers are detected
    '''

    # Detect the Aruco markers (using the 4X4 dictionary).
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
        image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    if annotateImage:
        cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

    # Abort if not all markers are detected.
    if (markerIds is None or len(markerIds) != 4 or
        set(markerIds.flatten()) != set([1,2,3,4])):
        return None


    # Determine the center of the marker pixel coordinates.
    uvMarkers = np.zeros((4,2), dtype='float32')
    for i in range(4):
        uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

    # Calculate the matching World coordinates of the 4 Aruco markers.
    DX = 0.1016
    DY = 0.06985
    xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                            [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])


    # Create the perspective transform.
    M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

    # Map the object in question.
    uvObj = np.float32([u, v])
    xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

    return xyObj