import numpy as np
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import PoseArray
from std_msgs.msg       import Bool

from sixdof.utils.TransformHelpers import *

from sixdof.states import Tasks, TaskHandler, JOINT_NAMES

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

        # Subscribers:
        # /joint_states from Hebi node
        self.fbksub = self.create_subscription(JointState, '/joint_states',
                                               self.recvfbk, 100)
        
        self.clear_sub = self.create_subscription(Bool, '/clear', self.rcvaction, 10)
        self.checker_move_sub = self.create_subscription(PoseArray, '/checker_move',
                                                       self.rcvaction, 10)
        self.dice_roll_sub = self.create_subscription(PoseArray, '/dice_roll',
                                                    self.rcvaction, 10)

        # creates task handler for robot
        self.task_handler = TaskHandler(self, np.array(self.position0).reshape(-1, 1))

        self.task_handler.add_state(Tasks.INIT)
        #self.test([0.3,0.5], [-0.3,0.5])

        self.waiting_for_move = False
        self.moveready_pub = self.create_publisher(Bool, '/move_ready', 1)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds

        # every 5s, check to see if queue is empty and we need a new move
        self.check_queue_timer = self.create_timer(5, self.check_queue)

    def test(self, source_pos, dest_pos):
        self.task_handler.move_checker(source_pos, dest_pos)
    
    # Called repeatedly by incoming messages - do nothing for now
    def recvfbk(self, fbkmsg):
        self.actpos = list(fbkmsg.position)

    def rcvaction(self, msg):
        if type(msg) is Bool:
            if msg.data == True:
                self.task_handler.clear()
        elif type(msg) is PoseArray:
            if len(msg.poses) == 2:
                action = []
                for pose in msg.poses:
                    p = p_from_T(T_from_Pose(pose))
                    action.append([p[0],p[1]])              
                self.task_handler.move_checker(action[0],action[1]) # (source, dest)
            else:
                pass
        self.waiting_for_move = False

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

    def check_queue(self):
        '''
        Check the spline queue. If empty, set ready for move to True, which will
        query the GameNode for a new message. Run this on a timer. 
        '''
        if self.task_handler.curr_task_object is not None:
            if self.task_handler.curr_task_object.done and len(self.task_handler.tasks) == 0 and not self.waiting_for_move:
                self.waiting_for_move = True
                msg = Bool()
                msg.data = self.waiting_for_move
                self.moveready_pub.publish(msg)
                

    # Receive new command update from trajectory - called repeatedly by incoming messages.
    def update(self):
        self.t = self.get_time()

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
        tau_elbow = -6 * np.sin(-q[1] + q[2])
        tau_shoulder = -tau_elbow + 9 * np.sin(-q[1])
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

def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = TrajectoryNode('traj')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()