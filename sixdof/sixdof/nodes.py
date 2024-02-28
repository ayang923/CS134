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
from sixdof.game import GameDriver, Color

from enum import Enum

RATE = 100.0            # Hertz

nan = float("nan")

GREEN_CHECKER_LIMITS = np.array(([10, 70], [30, 90], [10, 70]))
BROWN_CHECKER_LIMITS = np.array(([80, 120], [90, 140], [60, 120]))
YELLOW_BOARD_LIMITS = np.array([[80, 120], [100, 220], [150, 180]])
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

        # creates task handler for robot
        self.task_handler = TaskHandler(self, np.array(self.position0).reshape(-1, 1))

        #self.task_handler.add_state(Tasks.INIT)

        # game driver for trajectory node
        self.game_driver = GameDriver(self, self.task_handler)
        self.test_bucket_pub = self.create_publisher(PoseArray, '/buckets', 3)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        #self.timer = self.create_timer(1/rate, self.sendcmd)
        #self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
        #                       (self.timer.timer_period_ns * 1e-9, rate))
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
        
        GameDriver.determine_action()

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
                                                 self.process_top_images, 3)
        
        self.rcvtipimg = self.create_subscription(Image, '/tip_cam/image_raw',
                                                  self.process_tip_images, 3)
        
        self.rcvbuckets = self.create_subscription(PoseArray, '/buckets',
                                                   self.draw_buckets, 3)
        
        # Publishers for detected features:
        # Poses of two halves of game board
        self.pub_board = self.create_publisher(Pose, '/boardpose', 3)

        # Poses of all detected green checkers
        self.pub_green = self.create_publisher(PoseArray, '/green', 3)

        # Poses of all detected brown checkers
        self.pub_brown = self.create_publisher(PoseArray, '/brown', 3)

        self.pub_dice = self.create_publisher(UInt8MultiArray, '/dice', 3)
        
        #publishers for debugging images
        self.pub_board_mask = self.create_publisher(Image, 
                                                    '/usb_cam/board_binary', 3)

        self.pub_green_mask = self.create_publisher(Image, 
                                                    '/usb_cam/green_checker_binary', 3)

        self.pub_brown_mask = self.create_publisher(Image, 
                                                    '/usb_cam/brown_checker_binary', 3)

        self.bridge = cv_bridge.CvBridge()

        self.M = None
        self.Minv = None
        self.last_frame = None

        self.rgb = None

        self.M_timer = self.create_timer(5, self.set_perspective_transform_mat)
        self.Minv_timer = self.create_timer(5, self.set_inv_perspective_transform_mat)
        self.board_timer = self.create_timer(1, self.detect_board)

        self.x0 = 0.0
        self.y0 = 0.387

        self.best_board_rect = None
        
        self.board_u_bounds = None
        self.board_v_bounds = None

    def process_top_images(self, msg):     
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        self.last_frame = frame

        if self.M is None:
            self.set_perspective_transform_mat()

        if self.Minv is None:
            self.set_inv_perspective_transform_mat()

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        (H, W, D) = blurred.shape
        uc = W//2
        vc = H//2

        self.detect_checkers(frame, Color.GREEN)
        #self.detect_checkers(frame, Color.BROWN)

        # binary = cv2.erode( binary, None, iterations=iter)
        #self.get_logger().info(
        #    "Center pixel HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))
        
        self.rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    def detect_board(self):
        if self.last_frame is None:
            return None

        blurred = cv2.GaussianBlur(self.last_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary_yellow = cv2.inRange(hsv, YELLOW_BOARD_LIMITS[:, 0],
                                    YELLOW_BOARD_LIMITS[:, 1])
        binary_red = cv2.inRange(hsv, RED_BOARD_LIMITS[:, 0],
                                 RED_BOARD_LIMITS[:, 1])

        binary = cv2.bitwise_or(binary_yellow, binary_red)
        
        # trying to find board, dilating a lot to fill in boundary
        #binary_board = cv2.erode(binary, None, iterations=1)
        binary_board = cv2.dilate(binary, None, iterations=12)
        binary_board = cv2.erode(binary_board, None, iterations=16)

        contours_board, _ = cv2.findContours(binary_board, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

        # finding board contour
        if len(contours_board) > 0 and self.M is not None and self.Minv is not None:
            board_msk = np.zeros(binary.shape)
            # Pick the largest contour.
            contour_board = max(contours_board, key=cv2.contourArea)
            cv2.drawContours(board_msk,[contour_board], 0, 1, -1)
            # filters out everythign but board
            binary = cv2.bitwise_and(binary, binary, 
                                     mask=board_msk.astype('uint8'))

            # convert largest contour to xy
            contour_xy = []
            for point in contour_board:
                u = point[0][0]
                v = point[0][1]
                xy = uvToXY(self.M,u,v)
                if xy is not None:
                    [x, y] = xy
                    contour_xy.append([x, y])
            contour_xy = np.array(contour_xy, dtype=np.float32)
            
            # get min area rect in xy
            bound = cv2.minAreaRect(contour_xy)

            # filter
            if self.best_board_rect is None:
                self.best_board_rect = bound
            else:
                alpha = 0.01
                self.best_board_rect = (
                alpha * np.array(bound[0]) + (1 - alpha) * np.array(self.best_board_rect[0]),
                alpha * np.array(bound[1]) + (1 - alpha) * np.array(self.best_board_rect[1]),
                alpha * bound[2] + (1 - alpha) * self.best_board_rect[2])
            smoothed = cv2.boxPoints(self.best_board_rect)
            
            # draw red dot in center of rectangle
            centerxy = np.mean(smoothed, axis=0)
            centeruv = None
            x = centerxy[0]
            y = centerxy[1]
            uv = xyToUV(self.Minv,x,y)
            if uv is not None:
                [u, v] = uv
                centeruv = np.int0(np.array([u,v]))
            if centeruv is not None:
                cv2.circle(self.rgb,centeruv,radius=7,color=(0,0,255),thickness=-1)


            # draw filtered fit rect from xy in uv space
            rect_uv = []
            for xy in smoothed:
                x = xy[0]
                y = xy[1]
                uv = xyToUV(self.Minv,x,y)
                if uv is not None:
                    [u, v] = uv
                    rect_uv.append([u,v])
            if rect_uv:
                rect_uv = np.int0(np.array(rect_uv, dtype=np.float32))
                cv2.drawContours(self.rgb,[rect_uv],-1,(0,255,0),2)
                self.board_u_bounds = [np.min(rect_uv[:,0]),np.max(rect_uv[:,0])]
                self.board_v_bounds = [np.min(rect_uv[:,1]),np.max(rect_uv[:,1])]

        self.pub_board_mask.publish(self.bridge.cv2_to_imgmsg(binary_board))
        self.publish_board_pose()

    def detect_checkers(self, img, color:Color):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        if color == Color.GREEN:
            limits = self.green_checker_limits
            rgb = (0,255,0)
        else:
            limits = self.brown_checker_limits
            rgb = (255,0,0)

        binary = cv2.inRange(hsv, limits[:, 0], limits[:, 1])
        if self.board_u_bounds and self.board_v_bounds:    
            u, v = np.meshgrid(np.arange(binary.shape[1]), np.arange(binary.shape[0]))
            umin, umax = self.board_u_bounds
            vmin, vmax = self.board_v_bounds
            board_msk = ((u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)).astype(np.uint8)
            binary = cv2.bitwise_and(binary, binary, mask=board_msk)

        # Erode and Dilate. Definitely adjust the iterations!
        # probably need to erode a lot to make it recognize neighboring circles
        binary = cv2.dilate(binary, None, iterations=2)
        binary = cv2.erode(binary, None, iterations=2)
        binary = cv2.dilate(binary, None, iterations=1)
        binary = cv2.erode(binary, None, iterations=6)
        binary = cv2.dilate(binary, None, iterations=1)
        
        # Find contours in the mask
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        checkers = []
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            i = 0
            for contour in contours:
                if i > 14: # only consider the 15 largest contours
                    break
                (u,v), radius = cv2.minEnclosingCircle(contour)
                if self.rgb is not None: # draw circles on the imshow frame
                    self.rgb = cv2.circle(self.rgb, (int(u),int(v)), int(radius), rgb, 2)
                    #cv2.imshow('test', self.rgb)
                    #cv2.waitKey(1)
                xy = uvToXY(self.M, int(u), int(v))
                if xy is not None:
                    [x, y] = xy
                    checkers.append([x, y])
            checkers = np.array(checkers)
            self.publish_checkers(checkers, color)
        
        if color == Color.GREEN:
            self.pub_green_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        else:
            self.pub_brown_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        
    def draw_buckets(self, msg:PoseArray):
        # Cannot seem to get this to draw correctly, no idea where the scaling
        # is coming into play. Is self.Minv wrong?
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            uv = xyToUV(self.Minv,x,y)
            if uv is not None:
                [u, v] = uv
                centeruv = np.int0(np.array([u,v]))
            if centeruv is not None:
                cv2.circle(self.rgb,centeruv,radius=10,color=(255,50,50),thickness=2)
                cv2.imshow('buckets', self.rgb)
                cv2.waitKey(2)


    def publish_checkers(self, checkers, color:Color):
        checkerarray = PoseArray()
        if len(checkers > 0):
            for checker in checkers:
                p = pxyz(checker[0], checker[1], 0.005)
                R = Reye()
                checkerpose = Pose_from_T(T_from_Rp(R,p))
                checkerarray.poses.append(checkerpose)
            if color == Color.GREEN:
                self.pub_green.publish(checkerarray)
            else:
                self.pub_brown.publish(checkerarray)

    def process_tip_images(self, msg):
        # TODO
        pass

    def publish_board_pose(self):
        '''
        
        '''
        if self.last_frame is None:
            return None
        
        if self.best_board_rect is None:
            return None
        
        x,y = self.best_board_rect[0]
        theta = np.radians(self.best_board_rect[2])

        p1 = pxyz(x, y, 0.005)
        R1 = Rotz(theta)
        boardpose = Pose_from_T(T_from_Rp(R1,p1))

        self.pub_board.publish(boardpose)

    def set_perspective_transform_mat(self):
        if self.last_frame is None:
            print('no saved frame')
            return None
        
        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            self.last_frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        # Abort if not all markers are detected.
        if (markerIds is None or len(markerIds) != 4 or
            set(markerIds.flatten()) != set([1,2,3,4])):
            print('not all aruco seen')
            return None

        # Determine the center of the marker pixel coordinates.
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        # Calculate the matching World coordinates of the 4 Aruco markers.
        DX = 1.184 # horizontal center-center of table aruco markers
        DY = 0.4985 # vertical center-center of table aruco markers
        xyMarkers = np.float32([[self.x0+dx/2, self.y0+dy/2] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

        # return the perspective transform.
        self.M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)
    
    def set_inv_perspective_transform_mat(self):
        if self.last_frame is None:
            print('no last frame')
            return None
        
        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            self.last_frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        # Abort if not all markers are detected.
        if (markerIds is None or len(markerIds) != 4 or
            set(markerIds.flatten()) != set([1,2,3,4])):
            print('not all aruco seen')
            return None

        # Determine the center of the marker pixel coordinates.
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        # Calculate the matching World coordinates of the 4 Aruco markers.
        DX = 1.184 # horizontal center-center of table aruco markers
        DY = 0.4985 # vertical center-center of table aruco markers
        xyMarkers = np.float32([[self.x0+dx/2, self.y0+dy/2] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

        # return the perspective transform.
        self.Minv = cv2.getPerspectiveTransform(xyMarkers, uvMarkers)

# helper functions

def uvToXY(M,u,v):
    if M is None:
        return None
    # Map the object in question.
    uvObj = np.float32([u, v])
    xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

    return xyObj

def xyToUV(M,x,y):
    if M is None:
        return None
    # Map the object in question.
    xyObj = np.float32([x, y])
    uvObj = cv2.perspectiveTransform(xyObj.reshape(1,1,2), M).reshape(2)

    return uvObj

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

    print('markerids',markerIds)

    # Abort if not all markers are detected.
    if (markerIds is None or len(markerIds) != 4 or
        set(markerIds.flatten()) != set([1,2,3,4])):
        return None


    # Determine the center of the marker pixel coordinates.
    uvMarkers = np.zeros((4,2), dtype='float32')
    for i in range(4):
        uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

    # Calculate the matching World coordinates of the 4 Aruco markers.
    DX = 1.184 # horizontal center-center of table aruco markers
    DY = 0.4985 # vertical center-center of table aruco markers
    xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                            [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])


    # Create the perspective transform.
    M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

    # Map the object in question.
    uvObj = np.float32([u, v])
    xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)

    return xyObj