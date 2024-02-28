from geometry_msgs.msg  import Point, Pose, Quaternion, PoseArray
from std_msgs.msg import UInt8MultiArray

from enum import Enum

from rclpy.node         import Node

import numpy as np

from sixdof.TransformHelpers import *

import matplotlib.pyplot as plt

class Color(Enum):
    GREEN = 1
    BROWN = 2

LOCC = 0.3
LFREE = -0.05

# class passed into trajectory node to handle game logic
class GameDriver():
    def __init__(self, trajectory_node:Node, task_handler):
        self.trajectory_node = trajectory_node
        self.task_handler = task_handler
        
        # /boardpose from Detector
        self.sub_board = self.trajectory_node.create_subscription(Pose, '/boardpose',
                                                                  self.recvboard, 3)

        # Poses of all detected green checkers, /green from Detector
        self.sub_green = self.trajectory_node.create_subscription(PoseArray, '/green',
                                                                  self.recvgreen, 3)

        # Poses of all detected brown checkers, /brown from Detector
        self.sub_brown = self.trajectory_node.create_subscription(PoseArray, '/brown',
                                                                  self.recvbrown, 3)

        # /dice Unsigned int array with value of two detected dice from Detector
        self.sub_dice = self.trajectory_node.create_subscription(UInt8MultiArray, '/dice',
                                                                 self.recvdice, 3)
        
        # Representation of physical game board
        self.game_board = GameBoard()

        # log odds representation of occupancy by [green,brown]
        self.logoddsgrid = np.zeros((25,6,2))

        # Initial gamestate area assumes setup for beginning of game
        # each element indicates [num_green, num_brown]
        # beginning to end of array progresses ccw from robot upper right.
        # last item is middle bar
        self.gamestate = np.array([[2,0], [0,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0], [0,3], [0,0], [0,0], [0,0], [5,0],
                                   [0,5], [0,0], [0,0], [0,0], [3,0], [0,0],
                                   [5,0], [0,0], [0,0], [0,0], [0,0], [0,2],
                                   [0,0]])
        # self.recvgreen populates these arrays with detected green checker pos
        self.greenpos = None
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = None
        # self.recvdice populates this with detected [die1_int, die2_int]
        self.dice = np.array([])

    def recvboard(self, msg):
        '''
        create / update belief of where the boards are based on most recent
        /boardpose msg

        in: /boardpose PoseArray
        updates self.boardpose
        '''
        self.game_board.filtered_board_update(msg)
        self.pub_buckets()

    def recvgreen(self, msg):
        '''
        given most recent batch of green positions and board pose, update
        self.greenpos, self.logoddsgrid, and self.gamestate

        in: /green PoseArray

        updates:
        self.greenpos (extract xy positions only from most recent processed frame's
        PoseArray /green and refresh self.greenpos),
        self.logoddsgrid (log odds of grid spaces being occupied) using update_log_odds(),
        and self.gamestate (actionable game state representation)
        '''
        self.greenpos = []
        for pose in msg.poses:
            xy = [pose.position.x, pose.position.y]
            self.greenpos.append(xy)
        self.greenpos = np.array(self.greenpos)
        self.update_log_odds(Color.GREEN)
        self.update_gamestate()

    def recvbrown(self, msg):
        '''
        same as above, but for brown
        '''
        self.brownpos = []
        for pose in msg.poses:
            xy = [pose.position.x, pose.position.y]
            self.brownpos.append(xy)
        self.brownpos = np.array(self.brownpos)
        self.update_log_odds(Color.BROWN)
        self.update_gamestate()

    def recvdice(self, msg):
        '''
        process most recent /dice msg

        in: /dice UInt8MultiArray

        places values in self.dice
        TODO
        '''
        pass

    def update_log_odds(self, color:Color):   
        '''
        update log odds ratio for each of the grid spaces
        each space has p_green and p_brown (for being occupied by a green
        or brown checker respectively), but should implement some logic
        about columns only being able to contain one color? + complain if this
        is not the case?

        in: self (need self.greenpos and self.brownpos)
        
        updates self.logoddsgrid given the values in self.greenpos and self.brownpos
        from the most recent processed frame

        use self.game_board.get_grid_centers() to determine how to increment/decrement the log
        odds grid given the detected checker positions.
        FIXME
        '''
        if self.game_board.centers is None:
            return None

        if color == Color.GREEN:
            posarray = self.greenpos
        else:
            posarray = self.brownpos
        for i in np.arange(25):
            for j in np.arange(6):
                filled = False
                center = self.game_board.centers[i][j]
                for pos in posarray:
                    if np.sqrt((center[0]-pos[0])**2 + (center[1]-pos[1]**2)) < 0.02:
                        self.logoddsgrid[i][j][color.value - 1] += LOCC
                        filled = True
                        break
                if filled == False:
                    self.logoddsgrid[i][j][color.value - 1] += LFREE
                

    def update_gamestate(self):
        '''
        TODO
        '''
        prob = 1 - 1 / (1 + np.exp(self.logoddsgrid))
        for i in np.arange(25):
            greencount = 0
            browncount = 0
            for j in np.arange(6):
                if prob[i][j][0] > 0.9:
                    greencount += 1
                elif prob[i][j][1] > 0.9:
                    browncount += 1
            self.gamestate[i] = [greencount,browncount] # FIXME this is a dumb way to do it
        print(self.gamestate)

    def determine_action(self):
        '''
        given current knowledge of the gameboard + current state, figure out
        what to do next?
        this is where all of the actual "game logic" is held? basically a black
        box rn but this is probably the abstraction we eventually want?
        Proposed "flow":
        if it is my turn:
            if I am in the middle of an action:
                has anything happened that would make my intended action impossible?
                if yes: put piece back, evaluate board
                else: continue queued trajectories
            else (i have not acted yet):
                check previous dice roll and verify gamestate is a legal
                progression from the previous turn's gamestate
                "roll dice" (either abstracted rng or queue up trajectories)
                offload dice roll to decision engine which sends back intended move
                queue up trajectories necessary for these move(s)
        else:
            wait for my turn in the init position
        TODO
        '''
        pass

    def pub_buckets(self):
        bucket_poses = PoseArray()
        for i in np.arange(25):
            for j in np.arange(6):
                p = pxyz(self.game_board.centers[i][j][0], self.game_board.centers[i][j][1], 0)
                R = Reye()
                bucket_poses.poses.append(Pose_from_T(T_from_Rp(R,p)))
        self.trajectory_node.test_bucket_pub.publish(bucket_poses)

# physical representation of the two halves of the board
class GameBoard():
    def __init__(self):
        # FIXME with "expected" values for the boards
        self.board_x = 0.0
        self.board_y = 0.387
        self.board_t = np.pi/2

        self.tau = 3 # FIXME
        self.alpha = 1 - 1/self.tau

        self.L = 1.06 # board length
        self.H = 0.536 # board width
        self.dL = 0.067 # triangle to triangle dist
        self.dH = 0.040 # checker to checker stack dist

        self.dL0 = 0.2485 # gap from blue side to first triangle center
        self.dL1 = 0.117 - self.dL # gap between two sections of triangles (minus dL)

        self.centers = None

    def set_grid_centers(self):
        '''
        return 25 centers with 6 subdivisions each for bucketing given
        current belief of the board pose

        in: self (need board positional info)

        returns: center coordinates of 25*6 = 150 grid spaces that a checker
        could occupy.

        based only on geometry of board, basically need to measure this out
        FIXME Test this!!
        '''
        centers = np.zeros((25,6,2))
        for i in np.arange(6):
            for j in np.arange(6):
                x = self.board_x + self.L/2 - self.dL0 - i*self.dL
                y = self.board_y + self.H/2 - self.dH/2 - j*self.dH
                centers[i][j] = [x,y]

        for i in np.arange(6,12):
            for j in np.arange(6):
                x = self.board_x + self.L/2 - self.dL0 - self.dL1 - i*self.dL
                y = self.board_y + self.H/2 - self.dH/2 - j*self.dH
                centers[i][j] = [x,y]

        for i in np.arange(12,18):
            for j in np.arange(6):
                x = self.board_x + self.L/2 - self.dL0 - self.dL1 - (23-i)*self.dL
                y = self.board_y - self.H/2 + self.dH/2 + j*self.dH
                centers[i][j] = [x,y]

        for i in np.arange(18,24):
            for j in np.arange(6):
                x = self.board_x + self.L/2 - self.dL0 - (23-i)*self.dL
                y = self.board_y - self.H/2 + self.dH/2 + j*self.dH
                centers[i][j] = [x,y]

        for j in np.arange(6):
            x = self.board_x + self.L/2 - self.dL0 - 5*self.dL - (self.dL1+self.dL)/2
            y = self.board_y + 2.5*self.dH - j*self.dH
            centers[24][j] = [x,y]
        
        # planar rotate by theta - np.pi/2
        rotated_centers = np.zeros((25,6,2))
        theta = self.board_t - np.pi/2
        for i in np.arange(25):
            for j in np.arange(6):
                x = centers[i][j][0]*np.cos(theta) - centers[i][j][1]*np.sin(theta)
                y = centers[i][j][0]*np.sin(theta) + centers[i][j][1]*np.cos(theta)
                rotated_centers[i][j] = [x,y]
        '''        
        # uncomment for plotting centers
        
        flattened_centers = rotated_centers.reshape(-1, 2)

        # Extract x and y coordinates
        x_coords = flattened_centers[:, 0]
        y_coords = flattened_centers[:, 1]

        # Plot the points
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, color='blue')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Grid Centers')
        plt.grid(True)
        plt.show()'''
                                 
        self.centers = rotated_centers

    def filtered_board_update(self, measurement:Pose):
        self.board_x = self.filtered_update(self.board_x, measurement.position.x)
        self.board_y = self.filtered_update(self.board_y, measurement.position.y)
        R = R_from_T(T_from_Pose(measurement))
        t = np.arctan2(R[1,0],R[0,0]) # undo rotz
        self.board_t = self.filtered_update(self.board_t, t)
        self.set_grid_centers()

    def filtered_update(self, value, newvalue):
        value = (1 - self.alpha) * value + self.alpha * newvalue
        return value