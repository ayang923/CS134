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

import matplotlib.pyplot as plt

RATE = 100.0            # Hertz

nan = float("nan")

# 45 55 60
# 110 130 90

GREEN_CHECKER_LIMITS = np.array(([30, 70], [25, 80], [30, 90]))
BROWN_CHECKER_LIMITS = np.array(([100, 150], [20, 75], [90, 160]))
YELLOW_BOARD_LIMITS = np.array([[80, 120], [100, 220], [150, 180]])
RED_BOARD_LIMITS = np.array([[100, 140], [160, 240], [100, 180]])
BLUE_BOARD_LIMITS = np.array([[0, 40],[90, 140],[100, 190]])

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
        self.test_bucket_pub = self.create_publisher(PoseArray, '/buckets', 10)
        self.game_state_timer = self.create_timer(1, self.game_driver.update_gamestate)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds

        self.last_process = None

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
        if self.last_process is None or self.last_process - self.t > 0.5:
            self.last_process = self.get_time()
            self.game_driver.determine_action()

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
            self.get_logger().info(str(len(q)))
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
        
        # Publishers for detected features:
        # Pose of game board
        self.pub_board = self.create_publisher(Pose, '/boardpose', 3)

        # Poses of all detected green checkers
        self.pub_green = self.create_publisher(PoseArray, '/green', 3)

        # Poses of all detected brown checkers
        self.pub_brown = self.create_publisher(PoseArray, '/brown', 3)

        self.pub_dice = self.create_publisher(UInt8MultiArray, '/dice', 3)
        
        #publishers for debugging images
        self.pub_board_mask = self.create_publisher(Image, 
                                                    '/usb_cam/board_mask', 3)

        self.pub_green_mask = self.create_publisher(Image, 
                                                    '/usb_cam/green_checker_binary', 3)

        self.pub_brown_mask = self.create_publisher(Image, 
                                                    '/usb_cam/brown_checker_binary', 3)
        
        self.pub_markup = self.create_publisher(Image,'/usb_cam/markup',3)

        self.bridge = cv_bridge.CvBridge()

        self.M = None
        self.Minv = None

        self.rgb = None

        self.x0 = 0.0
        self.y0 = 0.387

        self.best_board_xy = (None,[1.06,0.535],0)
        self.board_buckets = None # nparray of bucket centers
        self.occupancy = np.array([[5,0], [0,0], [0,0], [0,0], [0,3], [0,0],
                                   [0,5], [0,0], [0,0], [0,0], [0,0], [2,0],
                                   [0,2], [0,0], [0,0], [0,0], [0,0], [5,0],
                                   [0,0], [3,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0]]) # basically game representation
        self.best_board_uv = None
        self.best_board_center_uv = None

        self.board_mask_uv = None

        self.start_time = 1e-9 * self.get_clock().now().nanoseconds
        self.last_update = None

        self.green_beliefs = None
        self.brown_beliefs = None
        
    def get_time(self):
        return 1e-9 * self.get_clock().now().nanoseconds - self.start_time
    
    def process_top_images(self, msg):     
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        if self.last_update is None or self.get_time() - self.last_update > 1:
            self.set_M(frame)
            self.set_Minv(frame)
            self.detect_board(frame)    
            self.publish_board_pose()
            self.last_update = self.get_time()

        self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.detect_checkers(frame, Color.GREEN)
        self.detect_checkers(frame, Color.BROWN)

        ####### Comment out for better performance without viz #######
        self.draw_best_board() # draws current best board in uv on self.rgb
        self.draw_checkers() # draws filtered checker locations
        self.draw_buckets() # draws current bucket knowledge on self.rgb 

        self.publish_rgb()
        ###############################################################

    def process_tip_images(self, msg):
        # TODO
        pass

    def detect_board(self,frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary_yellow = cv2.inRange(hsv, YELLOW_BOARD_LIMITS[:, 0],
                                    YELLOW_BOARD_LIMITS[:, 1])
        binary_red = cv2.inRange(hsv, RED_BOARD_LIMITS[:, 0],
                                 RED_BOARD_LIMITS[:, 1])
        binary_blue = cv2.inRange(hsv, BLUE_BOARD_LIMITS[:, 0],
                                 BLUE_BOARD_LIMITS[:, 1])

        binary_board = cv2.bitwise_or(binary_yellow, binary_red)
        #binary_board = cv2.bitwise_or(binary_board, binary_blue)
        
        # trying to find board, dilating a lot to fill in boundary
        binary_board = cv2.erode(binary_board, None, iterations=2)
        binary_board = cv2.dilate(binary_board, None, iterations=12)
        binary_board = cv2.erode(binary_board, None, iterations=7)

        #cv2.imshow('board', binary_board)
        #cv2.waitKey(1)

        contours_board, _ = cv2.findContours(binary_board, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

        # finding board contour
        if len(contours_board) > 0:
            # Pick the largest contour.
            contour_board = max(contours_board, key=cv2.contourArea)

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
            if (abs(bound[2] - self.best_board_xy[2]) < 25 and # not a bad angle
                abs(self.best_board_xy[1][0]*self.best_board_xy[1][1] - 
                    bound[1][0]*bound[1][1]) < 0.05): # not covered -> bad area
                alpha = 0.1
                self.best_board_xy = (
                bound[0] if self.best_board_xy[0] is None else alpha * np.array(bound[0]) + (1 - alpha) * np.array(self.best_board_xy[0]),
                alpha * np.array(bound[1]) + (1 - alpha) * np.array(self.best_board_xy[1]),
                alpha * bound[2] + (1 - alpha) * self.best_board_xy[2])
          
            # get filtered fit rect from xy in uv space
            rect_uv = []
            for xy in cv2.boxPoints(self.best_board_xy):
                x = xy[0]
                y = xy[1]
                uv = xyToUV(self.Minv,x,y)
                if uv is not None:
                    [u, v] = uv
                    rect_uv.append([int(u),int(v)])
            if rect_uv is not [] and len(rect_uv) == 4:
                self.best_board_uv = np.array(rect_uv)
                board_msk = np.zeros(binary_board.shape)
                rect_uv = np.int0(np.array(rect_uv, dtype=np.float32))
                self.board_mask_uv = np.uint8(cv2.drawContours(board_msk,[rect_uv],0,200,-1))

        self.update_centers()

        if self.board_mask_uv is not None:
            self.pub_board_mask.publish(self.bridge.cv2_to_imgmsg(self.board_mask_uv))

    def detect_checkers(self, frame, color:Color):
        if self.board_mask_uv is None:
            return None

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        if color == Color.GREEN:
            limits = self.green_checker_limits
            draw_color = (0,255,0)
        else:
            limits = self.brown_checker_limits
            draw_color = (255,0,0)

        binary = cv2.inRange(hsv, limits[:, 0], limits[:, 1])    
        binary = cv2.bitwise_and(binary, binary, mask=self.board_mask_uv)           

        # Erode and Dilate. Definitely adjust the iterations!
        # probably need to erode a lot to make it recognize neighboring circles
        binary = cv2.dilate(binary, None, iterations=2)
        binary = cv2.erode(binary, None, iterations=3)
        binary = cv2.dilate(binary, None, iterations=3)
        #binary = cv2.erode(binary, None, iterations=4)
        #binary = cv2.dilate(binary, None, iterations=1)

        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=27,
                                   param1=5, param2=7, minRadius=12, maxRadius=17)

        # Ensure circles were found
        checkers = []   
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
    
            # Draw the circles on the original image
            for (u, v, r) in circles:
                xy = uvToXY(self.M, int(u), int(v))
                if xy is not None:
                    [x, y] = xy
                    checkers.append([x, y])
            checkers = np.array(checkers)

            # Correspondence
            if color == Color.GREEN:
                if self.green_beliefs is None:
                    self.green_beliefs = [[pos,0] for pos in checkers]
                else:
                    self.green_beliefs = correspondence(checkers, self.green_beliefs)
                    positions = np.array([group[0] for group in self.green_beliefs if group[1] > 2])
                    self.publish_checkers(positions, color)
            else:
                if self.brown_beliefs is None:
                    self.brown_beliefs = [[pos,0] for pos in checkers]
                else:
                    self.brown_beliefs = correspondence(checkers, self.brown_beliefs)
                    positions = np.array([group[0] for group in self.brown_beliefs if group[1] > 2])
                    self.publish_checkers(positions, color)

        self.update_occupancy()

        # Checker Mask Troubleshooting
        ###################################################################
        if color == Color.GREEN:
            self.pub_green_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        else:
            self.pub_brown_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        ###################################################################
        
    def update_centers(self):
        # Cannot seem to get this to draw correctly, no idea where the scaling
        # is coming into play. Is self.Minv wrong?
        if self.best_board_xy[0] is None: # make sure we have detected the board
            return None
        
        centers = np.zeros((25,2))
        
        # board dimensions (all in m):

        cx = self.best_board_xy[0][0]
        cy = self.best_board_xy[0][1]

        L = 1.061 # board length
        H = 0.536 # board width
        dL = 0.067 # triangle to triangle dist
        dH = 0.040 # checker to checker stack dist

        dL0 = 0.247 # gap from blue side to first triangle center
        dL1 = 0.117 - dL # gap between two sections of triangles (minus dL)

        for i in np.arange(6):
            x = cx + L/2 - dL0 - i*dL
            y = cy + H/2 - dH/2 - 3.5*dH
            centers[i] = [x,y]

        for i in np.arange(6,12):
            x = cx + L/2 - dL0 - dL1 - i*dL
            y = cy + H/2 - dH/2 - 3.5*dH
            
        for i in np.arange(12,18):
            x = cx + L/2 - dL0 - dL1 - (23-i)*dL
            y = cy - H/2 + dH/2 + 3.5*dH
            centers[i] = [x,y]
            
        for i in np.arange(18,24):
            x = cx + L/2 - dL0 - (23-i)*dL
            y = cy - H/2 + dH/2 + 3.5*dH
            centers[i][j] = [x,y]
                
        x = cx + L/2 - dL0 - 5*dL - (dL1+dL)/2
        y = cy
        centers[24] = [x,y] # bar
        
        rotated_centers = np.zeros((25,2))
        theta = np.radians(self.best_board_xy[2])
        for i in np.arange(25):
            x = centers[i][0]*np.cos(theta) - centers[i][1]*np.sin(theta)
            y = centers[i][0]*np.sin(theta) + centers[i][1]*np.cos(theta)
            rotated_centers[i] = [x,y]

        self.board_buckets = rotated_centers

    def update_occupancy(self):
        pass
    
    def draw_best_board(self):
        if self.best_board_xy is not None and self.best_board_uv is not None:
            # draw center
            centerxy = np.mean(cv2.boxPoints(self.best_board_xy), axis=0)
            x = centerxy[0]
            y = centerxy[1]
            uv = xyToUV(self.Minv,x,y)
            if uv is not None:
                [u, v] = uv
                centeruv = np.int0(np.array([u,v]))
            cv2.circle(self.rgb,centeruv,radius=7,color=(0,0,255),thickness=-1)
            
            # draw outside of board
            cv2.drawContours(self.rgb,[self.best_board_uv],-1,(0,255,255),4)

    def draw_checkers(self):
        if self.brown_beliefs is not None and self.green_beliefs is not None:
            for pair in self.brown_beliefs:
                if pair[1] > 1.5: # check if we're confident its there
                    [x,y] = pair[0]
                    uv = xyToUV(self.Minv,x,y)
                    if uv is not None:
                        [u,v] = uv
                        cv2.circle(self.rgb, np.int0(np.array([u,v])), radius=15, color=(0,0,255), thickness=3)
            for pair in self.green_beliefs:
                if pair[1] > 1.5: # check if we're confident its there
                    [x,y] = pair[0]
                    uv = xyToUV(self.Minv,x,y)
                    if uv is not None:
                        [u,v] = uv
                        cv2.circle(self.rgb, np.int0(np.array([u,v])), radius=15, color=(0,255,0), thickness=3)

    def draw_buckets(self):
        if self.board_buckets is None:
            return None
        
        L = 100 # pixels
        W = 20 # pixels
        theta = self.board_buckets[2]

        for row in self.board_buckets:              
            x = row[0]
            y = row[1]
            uv = xyToUV(self.Minv,x,y)
            if uv is not None:
                [u, v] = uv
                centeruv = np.int0(np.array([u,v]))
                rect_points = cv2.boxPoints(((centeruv[0], centeruv[1]), (W, L), theta))
                rect_points = np.int0(rect_points)
                cv2.polylines(self.rgb, [rect_points], isClosed=True, color=(255, 50, 50), thickness=2)
                
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

    def publish_rgb(self):
            (H, W, D) = self.rgb.shape
            uc = W//2
            vc = H//2

            #self.get_logger().info(
            #    "Center pixel HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))
            
            cv2.line(self.rgb, (uc - 10, vc), (uc + 10, vc), (0, 0, 0), 2)
            cv2.line(self.rgb, (uc, vc - 7), (uc, vc + 8), (0, 0, 0), 2)
            
            self.pub_markup.publish(self.bridge.cv2_to_imgmsg(self.rgb, "bgr8"))

    def publish_board_pose(self):
        '''
        
        '''        
        if self.best_board_xy[0] is None:
            return None
        
        x,y = self.best_board_xy[0]
        theta = np.radians(self.best_board_xy[2])

        p1 = pxyz(x, y, 0.005)
        R1 = Rotz(theta)
        boardpose = Pose_from_T(T_from_Rp(R1,p1))

        self.pub_board.publish(boardpose)

    def set_M(self, frame):
        if frame is None:
            return None
        
        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

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
        xyMarkers = np.float32([[self.x0+dx/2, self.y0+dy/2] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

        # return the perspective transform.
        self.M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)
    
    def set_Minv(self,frame):
        if frame is None:
            return None
        
        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

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
        xyMarkers = np.float32([[self.x0+dx/2, self.y0+dy/2] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

        # return the perspective transform.
        self.Minv = cv2.getPerspectiveTransform(xyMarkers, uvMarkers)

# helper functions
        
EXIST = 0.3
CLEAR = -0.15

def correspondence(new, old):
    '''
    new are the freshly detected positions
    old is a list of [[position, log-odds]]
    '''
    alpha = 0.2
    updated = old.copy()
    persisting = np.zeros(len(updated)) # flag whether object in old was detected again
    corresp_ids = np.zeros(len(updated))
    i = 0
    oldpositions = [group[0] for group in old]
    for pos in new: # for each detected position
        found = False
        j = 0
        for oldpos in oldpositions: # check an existing object's position
            if np.sqrt((pos[0]-oldpos[0])**2 + (pos[1]-oldpos[1])**2) < 0.02 and not found: # if less than 2cm from it
                corresp_ids[j] = i
                persisting[j] = 1 # mark that old object was detected again
                found = True
            j += 1
        if not found: # if the detected position is not close to any existing objects
            updated.append([pos,0]) # add to updated with 50/50 chance of existence
        i += 1
    for i in range(len(persisting)):
        if persisting[i] == 0: # if not detected in the most recent frame
            updated[i][1] += CLEAR # decrease log odds
        else: # was detected in most recent frame
            if updated[i][1] < 4:
                updated[i][1] += EXIST # increase log odds
            updated[i][0] = [alpha * new[int(corresp_ids[i])][0] + (1-alpha) * updated[i][0][0],
                             alpha * new[int(corresp_ids[i])][1] + (1-alpha) * updated[i][0][1]] # filter position
    final = []
    for i in range(len(persisting)):
        if old[i][1] > -0.5: # get rid of anything that has a low chance of existing
            final.append(updated[i])

    return updated
            


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