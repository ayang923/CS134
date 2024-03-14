import numpy as np
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from geometry_msgs.msg  import PoseArray, Pose
from std_msgs.msg       import Bool, Float32MultiArray, UInt8MultiArray

from sixdof.utils.TransformHelpers import *

from sixdof.states import Tasks, TaskHandler, JOINT_NAMES

import copy

RATE = 100.0            # Hertz

nan = float("nan")

BEAROFF_BOUNDS = np.array([[1.105, 1.252], [0.138, 0.631]])

def reconstruct_gamestate_array(flattened_lst):
    return np.array(flattened_lst).reshape((26, 2)).tolist() if len(flattened_lst) == 52 else None

def reconstruct_checker_location_array(flattened_lst):
    game_state_list = flattened_lst[:52]
    checker_location_list = flattened_lst[52:]

    gamestate_array = reconstruct_gamestate_array(game_state_list)
    if len(checker_location_list) != 60:
        return None

    reconstructed_array = [[[], []] for _ in range(26)]
    curr_idx = 0
    for triangle_i, triangle in enumerate(gamestate_array):
        green_triangles = int(triangle[0])
        brown_triangles = int(triangle[1])

        for _ in range(green_triangles):
            reconstructed_array[triangle_i][0].append([checker_location_list[curr_idx], checker_location_list[curr_idx+1]])
            curr_idx += 2
        for _ in range(brown_triangles):
            reconstructed_array[triangle_i][1].append([checker_location_list[curr_idx], checker_location_list[curr_idx+1]])
            curr_idx += 2
    
    return reconstructed_array

def range_middle_out(start, stop, step=1):
    """
    Generates a list of integers similar to range() but starts from the middle and works outwards.
    
    Args:
        start (int): The starting value of the range.
        stop (int): The ending value of the range.
        step (int, optional): The step between each pair of consecutive values. Default is 1.
    
    Returns:
        list: A list of integers starting from the middle and working outwards.
    """
    middle = (start + stop) // 2  # Calculate the middle value
    
    # Generate the list from the middle value, working outwards
    result = []
    for i in range(0, middle - start + 1, step):
        result.append(middle + i)
        if middle - i != middle + i:
            result.append(middle - i)
    
    return result

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

        # creates task handler for robot
        self.task_handler = TaskHandler(self, np.array(self.position0).reshape(-1, 1))

        self.checker_locations = None

        # Subscribers:
        # /joint_states from Hebi node
        self.fbksub = self.create_subscription(JointState, '/joint_states',
                                               self.recvfbk, 100)
        
        self.clear_sub = self.create_subscription(Bool, '/clear', self.rcvaction, 10)
        self.checker_move_sub = self.create_subscription(UInt8MultiArray, '/checker_move',
                                                       self.rcvaction, 10)
        self.dice_roll_sub = self.create_subscription(PoseArray, '/dice_roll',
                                                    self.rcvaction, 10)

        self.sub_checker_locations = self.create_subscription(Float32MultiArray, '/checker_locations', self.recvcheckerlocations, 3)

        self.task_handler.add_state(Tasks.INIT)
        #self.test(np.array([0.3,0.6]).reshape(-1,1), np.array([1.0,0.6]).reshape(-1,1))

        self.sub_board = self.create_subscription(Pose, '/boardpose',
                                                                  self.recvboard, 3)

        self.waiting_for_move = False
        self.moveready_pub = self.create_publisher(Bool, '/move_ready', 1)

        self.sub_turn = self.create_subscription(Pose, '/turn', self.save_turn, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
        self.start_time = 1e-9 * self.get_clock().now().nanoseconds

        # every 5s, check to see if queue is empty and we need a new move
        self.check_queue_timer = self.create_timer(2, self.check_queue)

        # Physical Game Board Info
        self.board_x = None
        self.board_y = None
        self.board_theta = None # radians
        self.board_buckets = None # center locations [x,y]
        self.grid_centers = None # for placing

    def test(self, source_pos, dest_pos):
        self.task_handler.move_checker(source_pos, dest_pos)

    def save_turn(self, msg):
        # Save the detected turn signal
        data = p_from_T(T_from_Pose(msg))
        self.turn_signal_pos = [float(data[0]),float(data[1])]
        if self.turn_signal_pos[1] > 0.4:
            self.turn_signal = True
        else:
            self.turn_signal = False
    
    def recvcheckerlocations(self, msg):
        flattened_lst = list(msg.data)
        reconstructed_checker_array = reconstruct_checker_location_array(flattened_lst)
        if reconstructed_checker_array:
            self.checker_locations = reconstructed_checker_array

    def recvboard(self, msg):
        '''
        create / update belief of where the boards are based on most recent
        /boardpose msg

        in: /boardpose PoseArray
        updates self.boardpose
        '''
        self.save_board_dims(msg)
        self.update_centers()

    # Called repeatedly by incoming messages - do nothing for now
    def recvfbk(self, fbkmsg):
        self.actpos = list(fbkmsg.position)
        effort = list(fbkmsg.effort)
        #self.get_logger().info("gripper effort" + str(effort[5]))
        # self.get_logger().info("The angle of the gripper motor:" + str(self.actpos[5]))
        if effort[5] > -1.50 and self.task_handler.curr_task_type == Tasks.CHECK:
            self.task_handler.clear()


    def rcvaction(self, msg):
        if type(msg) is Bool:
            if msg.data == True:
                self.task_handler.clear()
        elif type(msg) is UInt8MultiArray:
            flattened_moves = list(msg.data)
            moves = list(np.array(flattened_moves).reshape((len(flattened_moves)//3, 3)))
            self.get_logger().info("moves: " + str(moves))
            checker_location_copy = copy.deepcopy(self.checker_locations)
            for source, dest, color in moves:
                if color == 10:
                    if source == 0:
                        self.task_handler.move_checker(self.turn_signal_pos, np.array([0.20,0.33]))
                    else:
                        self.task_handler.move_checker(self.turn_signal_pos, np.array([0.20,0.47]))
                    continue
                if not checker_location_copy[source][color]:
                    self.get_logger().info("checker locations " + str(checker_location_copy[source][color]))
                    continue
                source_pos = np.array(checker_location_copy[source][color][0])
                if dest <= 23:
                    if len(checker_location_copy[dest][0]) + len(checker_location_copy[dest][1]) == 0:
                        dest_pos = self.grid_centers[dest][0]
                    else:
                        if dest <= 11:
                            last_y = min(checker_location_copy[dest][0][0][1] if checker_location_copy[dest][0] else float('inf'), checker_location_copy[dest][1][0][1] if checker_location_copy[dest][1] else float('inf'))
                        else:
                            last_y = max(checker_location_copy[dest][0][0][1] if checker_location_copy[dest][0] else -float('inf'), checker_location_copy[dest][1][0][1] if checker_location_copy[dest][1] else -float('inf'))
                        dest_pos = np.array([self.grid_centers[dest][0][0], last_y + (-1 if dest <= 11 else 1)*0.045])
                elif dest == 24:
                    if len(checker_location_copy[dest][0]) + len(checker_location_copy[dest][1]) > 6:
                        continue

                    if color == 0:
                        if len(checker_location_copy[dest][0]) == 0:
                            dest_pos = self.grid_centers[dest][-2]
                            self.get_logger().info("grid centers " + str(self.grid_centers))
                            self.get_logger().info("grid center")
                        else:
                            last_y = checker_location_copy[dest][0][0][1]
                            dest_pos = np.array([self.grid_centers[dest][0][0], last_y + 0.045])

                    else:
                        if len(checker_location_copy[dest][1]) == 0:
                            dest_pos = self.grid_centers[dest][0]
                            
                        else:
                            last_y = checker_location_copy[dest][1][0][1]
                            dest_pos = np.array([self.grid_centers[dest][0][0], last_y - 0.045])
                else:
                    while True:
                        dest_pos = np.array([np.random.uniform(low=BEAROFF_BOUNDS[0, 0], high=BEAROFF_BOUNDS[0, 1]), np.random.uniform(low=BEAROFF_BOUNDS[1, 0], high=BEAROFF_BOUNDS[1, 1])])
                        for checker_location in checker_location_copy[dest][0]:
                            if np.linalg.norm(dest_pos - checker_location) >= 0.4:
                                continue
                        for checker_location in checker_location_copy[dest][1]:
                            if np.linalg.norm(dest_pos - checker_location) >= 0.4:
                                continue
                        
                        break
                        
                checker_location_copy[source][color].pop(0)
                checker_location_copy[dest][color].insert(0, dest_pos)
                self.task_handler.move_checker(source_pos, dest_pos)
        self.waiting_for_move = False

    def save_board_dims(self, msg:Pose):
        self.board_x = msg.position.x
        self.board_y = msg.position.y
        R = R_from_T(T_from_Pose(msg))
        t = np.arctan2(R[1,0],R[0,0]) # undo rotz
        self.board_theta = t
    
    def update_centers(self):
        centers = np.zeros((25,2))
        grid = np.zeros((25,6,2))
        
        # board dimensions (all in m):

        cx = self.board_x
        cy = self.board_y

        L = 1.061 # board length
        H = 0.536 # board width
        dL = 0.067 # triangle to triangle dist
        dH = 0.045 # checker to checker stack dist

        dL0 = 0.235 # gap from blue side to first triangle center
        dL1 = 0.117 - dL # gap between two sections of triangles (minus dL)

        for i in np.arange(6):
            x = cx + L/2 - dL0 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH - 0.01
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy + H/2 - dH/2 - j*dH - 0.01
                grid[i][j] = [x,y]

        for i in np.arange(6,12):
            x = cx + L/2 - dL0 - dL1 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH - 0.01
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy + H/2 - dH/2 - j*dH - 0.01
                grid[i][j] = [x,y]
            
        for i in np.arange(12,18):
            x = cx + L/2 - dL0 - dL1 - (23-i)*dL
            y = cy - H/2 + dH/2 + 2.5*dH
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy - H/2 + dH/2 + j*dH
                grid[i][j] = [x,y]                

    
        for i in np.arange(18,24):
            x = cx + L/2 - dL0 - (23-i)*dL
            y = cy - H/2 + dH/2 + 2.5*dH - 0.01
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy - H/2 + dH/2 + j*dH - 0.01
                grid[i][j] = [x,y]
                
        x = cx + L/2 - dL0 - 5*dL - (dL1+dL)/2
        y = cy
        centers[24] = [x,y] # bar
        for j in range_middle_out(0,5):
            y = cy + 2.5*dH - j*dH
            grid[24][j] = [x,y]
        
        rotated_centers = np.zeros((25,2))
        rotated_grid_centers = np.zeros((25,6,2))
        theta = self.board_theta
        for i in np.arange(25):
            x = centers[i][0]*np.cos(theta) - centers[i][1]*np.sin(theta)
            y = centers[i][0]*np.sin(theta) + centers[i][1]*np.cos(theta)
            rotated_centers[i] = [x,y]
            for j in np.arange(6):
                x = grid[i][j][0]*np.cos(theta) - grid[i][j][1]*np.sin(theta)
                y = grid[i][j][0]*np.sin(theta) + grid[i][j][1]*np.cos(theta)
                rotated_grid_centers[i][j] = [x,y]

        self.board_buckets = rotated_centers
        self.grid_centers = rotated_grid_centers

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
                if self.task_handler.curr_task_type is Tasks.INIT:
                    self.waiting_for_move = True
                    msg = Bool()
                    msg.data = self.waiting_for_move
                    self.moveready_pub.publish(msg)
                else:
                    self.task_handler.add_state(Tasks.INIT)
                

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
        tau_elbow = -6 * np.sin(-q[1] + q[2]) - 0.3
        tau_shoulder = -tau_elbow + 8.8 * np.sin(-q[1])
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