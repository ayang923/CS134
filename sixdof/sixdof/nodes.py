import numpy as np
import rclpy

import cv2, cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import JointState, Image
from geometry_msgs.msg  import Point, Pose, Quaternion

from sixdof.TrajectoryUtils import goto, goto5
from sixdof.TransformHelpers import *

from sixdof.states import Tasks, TaskHandler, JOINT_NAMES

from enum import Enum

RATE = 100.0            # Hertz

nan = float("nan")


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

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(JointState, '/joint_states',
                                               self.recvfbk, 100)

        # creates task handler for robot
        self.task_handler = TaskHandler(self, np.array(self.position0).reshape(-1, 1))
        self.task_handler.add_state(Tasks.INIT)
        self.task_handler.add_state(Tasks.TASK_SPLINE, x_final=np.array([-0.5482, 0.265, 0.000]).reshape(-1, 1))

        # game driver for trajectory node
        self.game_driver = GameDriver(self, self.task_handler)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds



    # Called repeatedly by incoming messages - do nothing for now
    def recvfbk(self, fbkmsg):
        self.actpos = list(fbkmsg.position)

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
        A = -5.9
        B = -0.5
        C = 10.0
        D = 0.0
        tau_elbow =  A * np.sin(-q[1] + q[2]) + B * np.cos(-q[1] + q[2])
        tau_shoulder = -tau_elbow + C * np.sin(-q[1]) + D * np.cos(-q[1])
        return (tau_shoulder, tau_elbow)

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        (q, qdot) = self.update()
        (tau_shoulder, tau_elbow) = self.gravitycomp(self.actpos)
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = self.jointnames
        #self.cmdmsg.position     = [nan, nan, nan, nan, nan] # uncomment for gravity comp test
        #self.cmdmsg.velocity     = [nan, nan, nan, nan, nan]
        self.cmdmsg.position     = q # comment for gravity comp
        self.cmdmsg.velocity     = qdot
        self.cmdmsg.effort       = [0.0, tau_shoulder, tau_elbow, 0.0, 0.0]
        self.cmdpub.publish(self.cmdmsg)

# class passed into trajectory node to handle game logic
class GameDriver():
    def __init__(self, trajectory_node, task_handler):
        self.trajectory_node = trajectory_node
        self.task_handler = task_handler

        self.rcvpoints = self.trajectory_node.create_subscription(Point, '/point',
                                                  self.recvpoint, 3)
        
    def recvpoint(self, pointmsg):
        if not self.task_handler.curr_task_object.done:
            self.trajectory_node.get_logger().info("in motion")
            return
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        
        origin = np.array([-0.3, 0.03, 0.15]).reshape(-1,1)
        point = np.array([x, y, z]).reshape(-1,1)
        
        # print warning
        if np.linalg.norm(point - origin) > 0.75:
            self.get_logger().info("Input near / outside workspace!")
        
        # Report.
        self.task_handler.add_state(Tasks.TASK_SPLINE, x_final=point, T=3)


class DetectorNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.checker_limits = np.array(([80, 120], [175, 255], [175, 255]))
        self.strip_limits = np.array(([80, 120], [110, 150], [175, 255]))
        
        self.bridge = cv_bridge.CvBridge()

        # subscriber for raw images
        self.rcvimages = self.create_subscription(Image, '/usb_cam/image_raw',
                                                  self.process_raw_images, 1)
        
        # publisher for detections
        self.pubpoints = self.create_publisher(Point, '/point', 3)
        self.pubposes = self.create_publisher(Pose, '/pose', 3)
        
        # publisher for debugging images
        self.pub_strip = self.create_publisher(Image, '/goals4/binary',    3)



    def process_raw_images(self, msg):
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

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 0    # the graylevel of images
        params.maxThreshold = 75

        binary_msk_die = cv2.inRange(blurred, 0, 75)

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(grayscale)
        print(keypoints)

        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary_strip = cv2.inRange(hsv, self.checker_limits[:, 0], self.checker_limits[:, 1])
        binary_checker = cv2.inRange(hsv, self.strip_limits[:,0], self.strip_limits[:,1])
        binary = cv2.bitwise_or(binary_checker, binary_strip)

        self.pub_strip.publish(self.bridge.cv2_to_imgmsg(im_with_keypoints, "rgb8"))

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
            x0, y0 = 0.007, 0.477

            ((ur, vr), (w, h), theta) = cv2.minAreaRect(contour)
            xy = pixelToWorld(frame, ur, vr, x0, y0)

            #self.get_logger().info(f"{theta}")

            if xy is not None:
                (x, y) = xy
                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                # Naci Note here: I was checking rviz while running this and also sending node commands.
                # I think we have a small error with our z height recognition, thus I am changing the 
                # z value here. 
                point_msg.z = 0.005
                
                if 0.5 <= w/h <= 2:
                    self.pubpoints.publish(point_msg)
                else:
                    pose_msg = Pose()
                    pose_msg.position = point_msg

                    quart_msg = Quaternion()
                    quart_msg.x = 0.0
                    quart_msg.y = 0.0
                    quart_msg.z = np.sin(theta/2)
                    quart_msg.w = np.cos(theta/2)
                    pose_msg.orientation = quart_msg

                    self.pubpoints.publish(point_msg)
                    self.pubposes.publish(pose_msg)

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