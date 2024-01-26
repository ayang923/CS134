#!/usr/bin/env python3
#
#   goals3.py
#
#   goals3 Node and Trajectory with touching point
#
import numpy as np
import rclpy

import cv2, cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import JointState, Image
from geometry_msgs.msg  import Point, Pose, Quaternion

from threedof.TrajectoryUtils import goto, goto5
from threedof.TransformHelpers import *

from threedof.KinematicChain import KinematicChain

from enum import Enum
from scipy.linalg import diagsvd


#
#   Definitions
#
RATE = 100.0            # Hertz

chainjointnames = ['base', 'shoulder', 'elbow']

class State(Enum):
    INIT = 1
    WAIT = 2
    ACTION = 3

#
#   Trajectory Node Class
#
class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.pub_strip = self.create_publisher(Image, '/goals4/binary',    3)

        self.pubpoints = self.create_publisher(Point, '/point', 3)
        self.rcvpoints = self.create_subscription(Point, '/point',
                                                  self.recvpoint, 3)
        
        self.pubposes = self.create_publisher(Pose, '/pose', 3)
        self.rcvposes = self.create_subscription(Pose, '/pose', self.recvposes, 3)
        
        self.checker_limits = np.array(([80, 120], [175, 255], [175, 255]))
        self.strip_limits = np.array(([80, 120], [110, 150], [175, 255]))
        
        self.bridge = cv_bridge.CvBridge()

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.actpos = self.position0
        self.get_logger().info("Initial positions: %r" % self.position0)
        
        self.trajectory = Trajectory(self, self.position0)
        self.jointnames = chainjointnames

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 100) # /joint_commands publisher

        self.rcvimages = self.create_subscription(Image, '/usb_cam/image_raw',
                                                  self.process, 1)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(JointState, '/joint_states',
                                               self.recvfbk, 100)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds

    def recvpoint(self, pointmsg):
        if not self.trajectory.state_handler.state_object.done:
            self.get_logger().info("in motion")
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
        self.trajectory.state_queue += [(State.ACTION, [point, 5]), (State.INIT, [])]
    
    def recvposes(self, posemsg):
        if not self.trajectory.state_handler.state_object.done:
            self.get_logger().info("in motion")
            return
        
        T = T_from_Pose(posemsg)

        p = p_from_T(T)
        R = R_from_T(T)

        pos_side_p =  p + (R@ex()*0.013)
        neg_side_p = p + (R@ex()*-0.013)


        self.trajectory.state_queue += [(State.ACTION, [pos_side_p, 5]), (State.ACTION, [pos_side_p + ez()*0.02, 1]), (State.ACTION, [neg_side_p, 1]), (State.INIT, [])]

        self.get_logger().info(f"{R @ np.array([1, 0, 0]).reshape(-1, 1)}")

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
        desired = self.trajectory.evaluate(self.actpos, self.t, 1 / RATE)
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
        tau_shoulder = -1.3 * np.sin(q[1])
        return tau_shoulder

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        (q, qdot) = self.update()
        tau_shoulder = self.gravitycomp(self.actpos)
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = self.jointnames
        self.cmdmsg.position     = q
        self.cmdmsg.velocity     = qdot
        self.cmdmsg.effort       = [0.0, tau_shoulder, 0.0]
        self.cmdpub.publish(self.cmdmsg)

    def process(self, msg):
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
        binary_strip = cv2.inRange(hsv, self.checker_limits[:, 0], self.checker_limits[:, 1])
        binary_checker = cv2.inRange(hsv, self.strip_limits[:,0], self.strip_limits[:,1])
        binary = cv2.bitwise_or(binary_checker, binary_strip)

        self.pub_strip.publish(self.bridge.cv2_to_imgmsg(binary))

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
            xy = self.pixelToWorld(frame, ur, vr, x0, y0)

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

                    self.pubposes.publish(pose_msg)

                
    
    # Pixel Conversion
    def pixelToWorld(self, image, u, v, x0, y0, annotateImage=True):
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

class InitState():
    def __init__(self, t, trajectory, initial = True):
        self.start = t
        self.trajectory = trajectory

        self.q0 = self.trajectory.q
        self.q1 = self.trajectory.q1
        if not initial:
            self.q1[2] = 0.0
        self.q2 = self.trajectory.q2

        self.done = np.linalg.norm(self.q2 - self.q0) < 0.1

    def evaluate(self, t, dt):
        t = t - self.start - dt
        if self.done:
            return self.trajectory.q, np.zeros((3, 1))
        elif (t < 3.0):
            return goto5(t, 3.0, self.q0, self.q1)
        elif (t < 6.0):
            return goto5(t-3, 3.0, self.q1, self.q2)
        else:
            self.done = True
            return self.trajectory.q, np.zeros((3, 1))

class ActionState():
    def __init__(self, t, trajectory, x_t, T):
        self.start = None
        self.trajectory = trajectory
        
        self.q0 = self.trajectory.q
        self.p0 = self.trajectory.x
        self.fakeq = self.q0
        self.fakex = self.p0

        self.T = T

        self.x_t = x_t
        self.done = False

    def evaluate(self, t, dt):
        if not self.start:
            self.start = t

        t = t - self.start

        if t < self.T:
            (pd, vd) = goto5(t, self.T, self.p0, self.x_t)

            (self.trajectory.x, _, Jv, _) = self.trajectory.chain.fkin(self.trajectory.q)

            e = ep(pd, self.trajectory.x)
            J = Jv
            xdotd = vd

            gamma = 0.1
            U, S, V = np.linalg.svd(Jv)

            msk = np.abs(S) >= gamma
            S_inv = np.zeros(len(S))
            S_inv[msk] = 1/S[msk]
            S_inv[~msk] = S[~msk]/gamma**2

            J_inv = V.T @ diagsvd(S_inv, *J.T.shape) @ U.T

            qdot = J_inv@(xdotd + self.trajectory.lam * e)

            q = self.trajectory.q + qdot*dt

        else:
            q, qdot = self.trajectory.q, np.zeros((3, 1))
            self.done = True

        return q, qdot

class StateHandler():
    def __init__(self, init_state: State, trajectory):
        self.trajectory = trajectory
        self.state = init_state
        self.state_object = InitState(0, trajectory)

    def set_state(self, state: State, t, *args):
        if self.state_object.done:
            self.state = state
            if self.state == State.INIT:
                self.state_object = InitState(t, self.trajectory, initial = False)
            elif self.state == State.ACTION:
                self.state_object = ActionState(t, self.trajectory, *args)
            return True
        else:
            return False
    
    def get_evaluator(self):
        return self.state_object.evaluate

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node, q0):
        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())
        
        # Define the starting sequence joint positions.
        self.q0 = np.array(q0).reshape(-1,1)
        self.q1 = np.array([q0[0], 0.0, q0[2]]).reshape(-1,1)
        self.q2 = np.array([0.0, 0.0, np.pi/2]).reshape(-1,1) # upright with elbow at 90, pointing fwd
        self.q = self.q0


        (self.p0, _, _, _) = self.chain.fkin(self.q2) # initial pos/Rot

        self.x = self.p0 # current state pos

        self.lam = 10

        self.state_handler = StateHandler(State.INIT, self)
        self.state_queue = []

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names
        return chainjointnames

    # Evaluate at the given time.
    def evaluate(self, actpos, t, dt):
        self.x, _, _, _ = self.chain.fkin(self.q)
        actx, _, _, _ = self.chain.fkin(actpos)
        self.q, qdot = self.state_handler.get_evaluator()(t, dt)


        if (self.state_handler.state == State.ACTION and
            np.linalg.norm(actx - self.x) > 0.0254):
                print("COLLISION DETECTED!\nActual: {}\nExpected: {}".
                      format(actx, self.x), flush=True)
                self.state_handler.state_object.done = True

        if self.state_queue:
            head_el = self.state_queue[0]
            if self.state_handler.set_state(head_el[0], t, *head_el[1]):
                self.state_queue.pop(0)
        
        # Return the position and velocity as flat python lists!
        return (self.q.flatten().tolist(), qdot.flatten().tolist())

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the Trajectory node.
    node = TrajectoryNode('Goals3', Trajectory)

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()