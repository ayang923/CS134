#!/usr/bin/env python3
#
#   goals3.py
#
#   goals3 Node and Trajectory with touching point
#
import numpy as np
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState

from basic134.TrajectoryUtils import goto, goto5
from basic134.TransformHelpers import *

from basic134.KinematicChain import KinematicChain

from enum import Enum


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

        self.fbksub = self.create_subscription(
            Point, '/point', self.recvpoint, 10)        

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.actpos = self.position0
        self.get_logger().info("Initial positions: %r" % self.position0)
        
        self.trajectory = Trajectory(self, self.position0)
        self.jointnames = chainjointnames

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 100) # /joint_commands publisher

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 100) # /joint_states subscriber (from Hebi!)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds

    def recvpoint(self, pointmsg):
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
        self.trajectory.state_queue += [(State.ACTION, point), (State.INIT, None)]

    # Called repeatedly by incoming messages - do nothing for now
    def recvfbk(self, fbkmsg):
        self.actpos = np.array(list(fbkmsg.position)).reshape(-1, 1)

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
        tau_shoulder = -1.6 * np.sin(q[1])
        tau_elbow = -0.1 * np.sin(q[1] + q[2])
        return [0.0, tau_shoulder[0], tau_elbow[0]]

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        (q, qdot) = self.update()
        effort = self.gravitycomp(self.actpos)
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = self.jointnames
        self.cmdmsg.position     = q
        self.cmdmsg.velocity     = qdot
        self.cmdmsg.effort       = effort
        self.cmdpub.publish(self.cmdmsg)

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
        t = t - self.start
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
    def __init__(self, t, trajectory, x_t):
        self.start = t
        self.trajectory = trajectory
        
        self.q0 = self.trajectory.q
        self.p0 = self.trajectory.x

        self.x_t = x_t
        self.done = False

    def evaluate(self, t, dt):
        t = t - self.start

        if t < 3.0:
            (pd, vd) = goto5(t, 3.0, self.p0, self.x_t)

            (self.trajectory.x, _, Jv, _) = self.trajectory.chain.fkin(self.trajectory.q)

            e = ep(pd, self.trajectory.x)
            J = Jv
            xdotd = vd

            J_Winv = np.transpose(J)@np.linalg.inv(J@np.transpose(J) + (0.3**2)*np.eye(3))

            qdot = J_Winv@(xdotd + e*self.trajectory.lam)

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

        self.table_pos = np.array([0.1, 0.4, 0.0]).reshape(-1, 1)

        self.state_handler = StateHandler(State.INIT, self)
        self.state_queue = [(State.ACTION, self.table_pos), (State.INIT, None)]

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names
        return chainjointnames

    # Evaluate at the given time.
    def evaluate(self, actpos, t, dt):
        # Compute the joint values.
        #if   self.state == State.INIT: return self.evaluate_init(t, dt)
        # self.q, qdot = self.state_trajectory.evaluate(t, dt, self.q)
        # elif (t < 12.0):
        #     if (t < 9.0):
        #         (pd, vd) = goto5(t-6, 3.0, self.p0, self.table_point) # task spline
        #     else:
        #         (pd, vd) = goto5(t-9, 3.0, self.table_point, self.p0) # task spline
            

        #     (self.x, _, Jv, _) = self.chain.fkin(self.q)

        #     e = ep(pd, self.x)
        #     J = Jv
        #     xdotd = vd

        #     qdot = np.linalg.inv(J)@(xdotd + e*self.lam)

        #     self.q = self.q + qdot*dt

        # else:
        #     qdot = np.zeros((3, 1))

        self.q, qdot = self.state_handler.get_evaluator()(t, dt)
        self.x, _, _, _ = self.chain.fkin(self.q)
        actx, _, _, _ = self.chain.fkin(actpos)

        if (self.state_handler.state == State.ACTION and
            np.linalg.norm(actx - self.x) > 0.03):
                print("COLLISION DETECTED!\nActual: {}\nExpected: {}".
                      format(actx, self.x), flush=True)
                self.state_handler.state_object.done = True

        if self.state_queue:
            head_el = self.state_queue[0]
            if self.state_handler.set_state(head_el[0], t, head_el[1]):
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