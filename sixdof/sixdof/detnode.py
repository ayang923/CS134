# TODO: tweak hsv bounds so it can detect in shadows (especially for brown/purple)
# TODO: verify turn signal detect / publish functionality (DetectorNode.detect_turn_signal()
# and DetectorNode.publish_turn_signal())

import numpy as np
import rclpy

import cv2, cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Pose, PoseArray
from std_msgs.msg       import UInt8MultiArray, Bool

from sixdof.utils.TransformHelpers import *

from sixdof.gamenode import Color

GREEN_CHECKER_LIMITS = np.array(([30, 70], [25, 80], [30, 90]))
BROWN_CHECKER_LIMITS = np.array(([100, 150], [20, 75], [70, 180]))
YELLOW_BOARD_LIMITS = np.array([[80, 120], [100, 220], [150, 180]])
RED_BOARD_LIMITS = np.array([[100, 140], [160, 240], [100, 180]])
BLUE_BOARD_LIMITS = np.array([[0, 40],[90, 140],[100, 190]])

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

        self.pub_turn_signal = self.create_publisher(Pose, '/turn', 10)
        
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

        # x,y of aruco tag rectangle's center
        self.x0 = 0.77875
        self.y0 = 0.3685

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
        self.turn_signal_belief = None
        
    def get_time(self):
        return 1e-9 * self.get_clock().now().nanoseconds - self.start_time
    
    def process_top_images(self, msg):     
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        if (self.last_update is None or self.get_time() - self.last_update > 1 or
            self.best_board_xy[0] is None):
            self.set_M(frame)
            self.set_Minv(frame)
            self.detect_board(frame)    
            self.publish_board_pose()
            self.last_update = self.get_time()

        self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.detect_checkers(frame, Color.GREEN)
        self.detect_checkers(frame, Color.BROWN)
        #self.detect_turn_signal(frame)
        self.update_occupancy()

        ####### Comment out for better performance without viz #######
        self.draw_best_board() # draws current best board in uv on self.rgb
        self.draw_checkers() # draws filtered checker locations
        self.draw_buckets() # draws current bucket knowledge on self.rgb 
        #self.draw_turn_signal()

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
            try: 
                bound = cv2.minAreaRect(contour_xy)
            except:
                #print('convexhull error')
                return None

            # filter
            if (abs(bound[2] - self.best_board_xy[2]) < 25 and # not a bad angle
                abs(self.best_board_xy[1][0]*self.best_board_xy[1][1] - 
                    bound[1][0]*bound[1][1]) < 0.05 and 
                    abs(bound[0][0] - self.x0) < 0.4): # not covered -> bad area
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
                self.board_mask_uv = np.uint8(cv2.drawContours(board_msk,[rect_uv],0,255,-1))

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


        # Checker Mask Troubleshooting
        ###################################################################
        if color == Color.GREEN:
            self.pub_green_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        else:
            self.pub_brown_mask.publish(self.bridge.cv2_to_imgmsg(binary))
        ###################################################################

    def detect_turn_signal(self,frame):
        # TODO
        if self.board_mask_uv is None:
            return None

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        limits = self.brown_checker_limits

        binary = cv2.inRange(hsv, limits[:, 0], limits[:, 1])

        binary = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(self.board_mask_uv))
        
        #binary = cv2.dilate(binary, None, iterations=2)
        binary = cv2.erode(binary, None, iterations=3)
        binary = cv2.dilate(binary, None, iterations=3)
        #binary = cv2.erode(binary, None, iterations=3)
        #binary = cv2.erode(binary, None, iterations=3)

        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=27,
                                   param1=7, param2=10, minRadius=12, maxRadius=17)

        turnsignal = []
        if circles is not None and len(circles[0]) == 1:
            circles = np.round(circles[0, :]).astype("int")
            for (u, v, r) in circles:
                xy = uvToXY(self.M, int(u), int(v))
                if xy is not None:
                    [x, y] = xy
                    cv2.circle(self.rgb, np.int0(np.array([u,v])), r, color=(0,0,255), thickness=3)
                    turnsignal.append([x, y])
            turnsignal = np.array(turnsignal)

        if self.turn_signal_belief is None:
                    self.turn_signal_belief = [[pos,0] for pos in turnsignal]
        else:
            self.turn_signal_belief = correspondence(turnsignal, self.turn_signal_belief)
            positions = np.array([group[0] for group in self.turn_signal_belief if group[1] > 2])
            if len(positions) == 1:
                self.publish_turn_signal(positions[0])

        cv2.imshow('turnsignal', binary)
        cv2.waitKey(1)
        return None

    def update_centers(self):
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

        dL0 = 0.235 # gap from blue side to first triangle center
        dL1 = 0.117 - dL # gap between two sections of triangles (minus dL)

        for i in np.arange(6):
            x = cx + L/2 - dL0 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH
            centers[i] = [x,y]

        for i in np.arange(6,12):
            x = cx + L/2 - dL0 - dL1 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH
            centers[i] = [x,y]
            
        for i in np.arange(12,18):
            x = cx + L/2 - dL0 - dL1 - (23-i)*dL
            y = cy - H/2 + dH/2 + 2.5*dH
            centers[i] = [x,y]
            
        for i in np.arange(18,24):
            x = cx + L/2 - dL0 - (23-i)*dL
            y = cy - H/2 + dH/2 + 2.5*dH
            centers[i] = [x,y]
                
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
        self.occupancy = np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                                   [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                                   [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                                   [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                                   [0,0]])
        if self.green_beliefs is None or self.brown_beliefs is None or self.board_buckets is None:
            return None
        for green in self.green_beliefs:
            for bucket in self.board_buckets:
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                xg = green[0][0]
                yg = green[0][1]
                if ((xg >= xmin and xg <= xmax) and (yg >= ymin and yg <= ymax) and 
                    green[1]>= 1.5):
                    bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                    self.occupancy[bucket_ind][0] += 1
        for brown in self.brown_beliefs:
            for bucket in self.board_buckets:
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                xb = green[0][0]
                yb = green[0][1]
                xb = brown[0][0]
                yb = brown[0][1]
                if ((xb >= xmin and xb <= xmax) and (yb >= ymin and yb <= ymax) and 
                    brown[1]>= 1.5):
                    bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                    self.occupancy[bucket_ind][1] += 1
    
    def draw_best_board(self):
        if self.best_board_xy[0] is not None and self.best_board_uv is not None:
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

    def draw_turn_signal(self):
        if self.turn_signal_belief is not None:
            if len(self.turn_signal_belief) == 1:
                if self.turn_signal_belief[0][1] > 1.5:
                    [x,y] = self.turn_signal_belief[0][0]
                    uv = xyToUV(self.Minv,x,y)
                    if uv is not None:
                        [u,v] = uv
                        cv2.circle(self.rgb, np.int0(np.array([u,v])), radius=15, color=(156,87,255), thickness=3)

    def draw_buckets(self):
        if self.best_board_xy[0] is None:
            return None
        
        L = 150.0 # pixels
        W = 40.0 # pixels
        theta = -self.best_board_xy[2]

        for row in self.board_buckets:              
            x = row[0]
            y = row[1]
            # same min/max used for grouping checkers into buckets
            xmin = x - 0.03
            xmax = x + 0.03
            ymin = y - 0.13
            ymax = y + 0.13
            uvtopleft = xyToUV(self.Minv,xmin,ymax)
            uvbottomright = xyToUV(self.Minv,xmax,ymin)
            #uv = xyToUV(self.Minv,x,y)
            [uTL, vTL] = uvtopleft
            [uBR, vBR] = uvbottomright
            topleft = np.int0(np.array([uTL,vTL]))
            bottomright = np.int0(np.array([uBR,vBR]))
            cv2.rectangle(self.rgb, topleft, bottomright, color=(255,50,50),thickness=2)
            # Writing the number of elements in the buckets with color identification
            checker_nums = self.check_bucket(row)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            xg = x - 0.03
            xb = x 
            if np.where(self.board_buckets == row)[0][0] < 12:
                ygb = y + 0.13
                uvg = xyToUV(self.Minv,xg,ygb)
                uvb = xyToUV(self.Minv,xb,ygb)
                centergreen = tuple(np.int0(np.array([float(uvg[0]),float(uvg[1])])))
                centerbrown = tuple(np.int0(np.array([float(uvb[0]),float(uvb[1])])))
            elif np.where(self.board_buckets == row)[0][0] < 24:
                ygb = y - 0.13
                uvg = xyToUV(self.Minv,xg,ygb)
                uvb = xyToUV(self.Minv,xb,ygb)
                centergreen = tuple(np.int0(np.array([float(uvg[0]),float(uvg[1])])))
                centerbrown = tuple(np.int0(np.array([float(uvb[0]),float(uvb[1])])))
            else:
                ygb = y
                uvg = xyToUV(self.Minv,xg,ygb)
                uvb = xyToUV(self.Minv,xb,ygb)
                centergreen = tuple(np.int0(np.array([float(uvg[0]),float(uvg[1])])))
                centerbrown = tuple(np.int0(np.array([float(uvb[0]),float(uvb[1])])))
                
            cv2.putText(self.rgb, str(checker_nums[0]),centergreen, font, fontScale=1,color=(0, 255, 0),thickness=2)
            cv2.putText(self.rgb, str(checker_nums[1]),centerbrown, font, fontScale=1,color=(0, 0, 255),thickness=2)
                
    def check_bucket(self, bucket):
        bucket_ind = np.where(self.board_buckets == bucket)[0][0]
        green_count = self.occupancy[bucket_ind][0]
        brown_count = self.occupancy[bucket_ind][1]
        return [green_count, brown_count]
                
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

    def publish_turn_signal(self, turn_signal):
        p = pxyz(turn_signal[0],turn_signal[1],0.0)
        R = Reye()
        msg = Pose_from_T(T_from_Rp(R,p))
        self.pub_turn_signal.publish(msg)

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
        DX = 1.1935 # horizontal center-center of table aruco markers
        DY = 0.579 # vertical center-center of table aruco markers
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
        DX = 1.1935 # horizontal center-center of table aruco markers
        DY = 0.579 # vertical center-center of table aruco markers
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

def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('det')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()