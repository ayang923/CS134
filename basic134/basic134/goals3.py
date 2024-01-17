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


#
#   Definitions
#
RATE = 100.0            # Hertz

jointnames = ['placement',
              'baseplate',
              'base',
              'lbracket',
              'shoulder',
              'upperarm',
              'elbow',
              'lowerarm']

#
#   Trajectory Node Class
#
class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)        

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.get_logger().info("Initial positions: %r" % self.position0)
        
        self.trajectory = Trajectory(self, self.position0)
        self.jointnames = self.trajectory.jointnames()

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds


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

    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # Just print the position (for now).
        # print(list(fbkmsg.position))
        pass
        
    def update(self):
        self.t = 1e-9 * self.get_clock().now().nanoseconds - self.start_time
         # Compute the desired joint positions and velocities for this time.
        desired = self.trajectory.evaluate(self.t, 1/RATE)
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

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        (q, qdot) = self.update()
        q = np.array([q[2], q[4], q[6]]).reshape(-1,1)
        qdot = np.array([qdot[2], qdot[4], qdot[6]]).reshape(-1,1)
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['one', 'two', 'three']
        self.cmdmsg.position     = q
        self.cmdmsg.velocity     = qdot
        self.cmdmsg.effort       = [0.0, 0.0, 0.0]
        self.cmdpub.publish(self.cmdmsg)

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


        (self.p0, self.R0, _, _) = self.chain.fkin(self.q2) # initial pos/Rot

        self.x = self.p0 # current state pos
        self.R = self.R0 # current state Rot

        self.table_point = np.array([-0.15, 0.55, 0.010]).reshape(-1,1) # hardcoded point to touch, 1cm off table

        self.lam = 20

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names
        return jointnames
        
    def longq(q):
        return np.array([0.0,0.0,q[0],0.0,q[1],0.0,q[2],0.0]).reshape(-1,1)

    def shortq(longq):
        return np.array([longq[2],longq[4],longq[6]]).reshape(-1,1)

    # Evaluate at the given time.
    def evaluate(self, t, dt):
        # Compute the joint values.
        if   (t < 3.0): (self.q, qdot) = goto(t, 3.0, self.q0, self.q1)
        elif (t < 4.5): (self.q, qdot) = goto(t, 4.5, self.q1, self.q2)
        else:
            (self.x, _, Jv, _) = self.chain.fkin(self.longq(self.q))
            print(self.x)
            qdot = np.zeros(3).reshape(-1,1)
        '''elif (t < 16.5):
            if (t < 10.5):
                (pd, vd) = goto(t, 6, self.p0, self.table_point)
            else:
                (pd, vd) = goto(t, 6, self.table_point, self.p0)
            

            (self.x, _, Jv, _) = self.chain.fkin(self.q)

            e = ep(pd, self.x)
            J = Jv
            xdotd = vd

            qdot = np.linalg.inv(J)@(xdotd + e*self.lam)

            self.q = self.q + qdot*dt

            print(self.q)
            print(qdot)'''

        print(self.q)
        q = self.longq(self.q)
        qdot = self.longq(qdot)

        # Return the position and velocity as flat python lists!
        return (q.flatten().tolist(), qdot.flatten().tolist())
        


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the Trajectory node.
    node = TrajectoryNode('Goals2', Trajectory)

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
