from geometry_msgs.msg  import Point, Pose, Quaternion, PoseArray
from std_msgs.msg import UInt8MultiArray

from enum import Enum

from rclpy.node         import Node

import numpy as np
import random

from sixdof.TransformHelpers import *

import matplotlib.pyplot as plt

from sixdof.states import *

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
        self.gamestate = np.array([[5,0], [0,0], [0,0], [0,0], [0,3], [0,0],
                                   [0,5], [0,0], [0,0], [0,0], [0,0], [2,0],
                                   [0,2], [0,0], [0,0], [0,0], [0,0], [5,0],
                                   [0,0], [3,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0]])

        # self.recvgreen populates these arrays with detected green checker pos
        self.greenpos = None
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = None
        # self.recvdice populates this with detected [die1_int, die2_int]
        self.dice = np.array([])

        # Game engine
        self.game = Game(self.gamestate)

    def recvboard(self, msg):
        '''
        create / update belief of where the boards are based on most recent
        /boardpose msg

        in: /boardpose PoseArray
        updates self.boardpose
        '''
        #self.trajectory_node.get_logger().info("Board Received")
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
        #self.update_gamestate()

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
        #self.update_gamestate()

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

        use self.game_board.set_grid_centers() to determine how to increment/decrement the log
        odds grid given the detected checker positions.
        FIXME not working
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
                    if np.sqrt((center[0]-pos[0])**2 + (center[1]-pos[1])**2) < 0.02 and filled == False:
                        if self.logoddsgrid[i][j][color.value - 1] < 4: #probability of 0.98 limit
                            self.logoddsgrid[i][j][color.value - 1] += LOCC
                        filled = True
                if filled == False and (self.logoddsgrid[i][j][color.value - 1] > -2): # also limit on lower end for being free
                    self.logoddsgrid[i][j][color.value - 1] += LFREE
                

    def update_gamestate(self):
        '''
        FIXME not sure if this works
        '''
        prob = 1 - 1 / (1 + np.exp(self.logoddsgrid))
        for i in np.arange(25):
            greencount = 0
            browncount = 0
            for j in np.arange(6):
                if prob[i][j][0] > 0.8:
                    greencount += 1
                elif prob[i][j][1] > 0.8:
                    browncount += 1
            self.gamestate[i] = [greencount,browncount] # FIXME this is a dumb way to do it
        #self.trajectory_node.get_logger().info("Updated Game State:") #For debuggin purposes
        #self.trajectory_node.get_logger().info(str(self.gamestate))
        

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
        '''
        if len(self.task_handler.tasks) != 0:
            return

        self.game.set_state(self.gamestate)
        moves = self.handle_turn(self.game)

        print("Camera game state: {}".format(self.gamestate))
        print("Engine game state: {}".format(self.game.state))
        print("Dice roll: {}".format(self.game.dice))
        print("Number of moves: {}".format(len(moves)))
        print("Moves: {}".format(moves), flush = True)

        for (source, dest) in moves:
            print("Moving from {} to {}".format(source, dest))
            if source is None:
                self.execute_off_bar(dest)
            elif dest is None:
                self.execute_bear_off(source)
            elif np.sign(self.game.state[source]) != np.sign(self.game.state[dest]):
                self.execute_hit(source, dest)
            else:
                self.execute_normal(source, dest)
            self.game.move(source, dest)
        self.game.turn *= -1
    
    def execute_off_bar(self, dest):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        centers = self.game_board.centers
        dest_centers = centers[dest]
        bar_centers = centers[bar]

        if (self.turn == 1 and self.gamestate[dest][1] > 0 or
            self.turn == -1 and self.gamestate[dest][0] > 0):

            source_pos = dest_centers[0]
            dest_pos = bar_centers[num_bar if turn else 5 - num_bar]
            self.move_checker(source_pos, dest_pos)

        num_dest = self.gamestate[dest][turn]
        num_bar = self.gamestate[bar][turn]

        source_pos = bar_centers[6 - num_bar if turn else num_bar - 1]
        dest_pos = dest_centers[num_dest]

        self.move_checker(source_pos, dest_pos)

    def execute_bear_off(self, source):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        centers = self.game_board.centers
        source_centers = centers[source]
        bar_centers = centers[bar]

        num_source = self.gamestate[source][turn]

        source_pos = source_centers[num_source - 1]
        dest_pos = bar_centers[0]
        dest_pos[0] += 0.55

        self.move_checker(source_pos, dest_pos)
    
    def execute_hit(self, source, dest):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        if self.game_board.centers is None:
            return

        centers = self.game_board.centers
        source_centers = centers[source]
        dest_centers = centers[dest]
        bar_centers = centers[bar]

        num_source = self.gamestate[source][turn]
        num_bar = self.game.bar[0 if self.game.turn == 1 else 1]

        source_pos = dest_centers[0]
        dest_pos = bar_centers[5 - num_bar if turn else num_bar]

        self.move_checker(source_pos, dest_pos)

        source_pos = source_centers[num_source - 1]
        dest_pos = dest_centers[0]

        self.move_checker(source_pos, dest_pos)

    def execute_normal(self, source, dest):
        if self.game_board.centers is None:
            return
        
        turn = 0 if self.game.turn == 1 else 1

        centers = self.game_board.centers
        source_centers = centers[source]
        dest_centers = centers[dest]

        num_source = self.gamestate[source][turn]
        num_dest = self.gamestate[dest][turn]

        source_pos = source_centers[num_source - 1]
        dest_pos = dest_centers[num_dest]

        self.move_checker(source_pos, dest_pos)

    def move_checker(self, source_pos, dest_pos):
        source_pos = np.append(source_pos,[0.05, -np.pi / 2, np.pi / 2 - np.arctan2(source_pos[1], source_pos[0])])
        dest_pos = np.append(dest_pos, [0.05, -np.pi / 2, np.pi / 2 - np.arctan2(dest_pos[1], dest_pos[0])])

        self.trajectory_node.get_logger().info(f"source pos {source_pos}")
        self.trajectory_node.get_logger().info(f"dest pos {dest_pos}")
        
        self.task_handler.add_state(Tasks.INIT)
        self.task_handler.add_state(Tasks.TASK_SPLINE,
                                    x_final = np.array(source_pos), T = 5)
        self.task_handler.add_state(Tasks.GRIP)
        self.task_handler.add_state(Tasks.TASK_SPLINE,
                                    x_final = np.array(dest_pos), T = 5)
        self.task_handler.add_state(Tasks.GRIP, grip = False)
        self.task_handler.add_state(Tasks.INIT)

    def pub_buckets(self):
        bucket_poses = PoseArray()
        for i in np.arange(25):
            for j in np.arange(6):
                p = pxyz(self.game_board.centers[i][j][0], self.game_board.centers[i][j][1], 0)
                R = Reye()
                bucket_poses.poses.append(Pose_from_T(T_from_Rp(R,p)))
        # self.trajectory_node.get_logger().info("Sending bucket poses")
        self.trajectory_node.test_bucket_pub.publish(bucket_poses)
    
    def choose_move(self, game, moves):
        #print("Legal moves: {}".format(moves))
        if moves:
            move = random.choice(moves)
            return move
        if not game.num_checkers():
            game.done = True
        return None

    def handle_turn(self, game):
        moves = []
        game.roll()

        print("Dice: {}".format(game.dice))

        if game.dice[0] == game.dice[1]:
            for _ in range(4):
                moves = game.possible_moves(game.dice[0])
                move = self.choose_move(game, moves)
                if move is not None:
                    moves.append(move)
        else:
            larger = 1
            if game.dice[0] > game.dice[1]:
                larger = 0
            moves = game.possible_moves(larger)
            move = self.choose_move(game, moves)
            if move is not None:
                moves.append(move)
            moves = game.possible_moves(not larger)
            move = self.choose_move(game, moves)
            if move is not None:
                moves.append(move)

        return moves

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
        self.dL = 0.066 # triangle to triangle dist
        self.dH = 0.040 # checker to checker stack dist

        self.dL0 = 0.247 # gap from blue side to first triangle center
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
        theta = self.board_t
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

POINT_LIM = 6

class Game:
    def __init__(self, gamestate):
        self.set_state(gamestate)
        self.bar = [0, 0]
        self.dice = [0, 0]
        self.turn = 1
        self.done = False
        self.clicked = None

    # Converts state in GameDriver format to state in engine format.
    def set_state(self, gamestate):
        self.state = []
        self.bar = gamestate[24]
        for point in gamestate[:24]:
            if point[0]:
                self.state.append(point[0])
            else:
                self.state.append(-point[1])
        self.state = np.append(self.state[11::-1], self.state[:11:-1]).tolist()

    def roll(self):
        self.dice = np.random.randint(1, 7, size = 2).tolist()
    
    def move(self, point1, point2):
        if point1 is None:
            self.bar[0 if self.turn == 1 else 1] -= 1
        if self.turn == 1:
            if point2 is None:
                self.state[point1] -= 1
            elif self.state[point2] >= 0:
                if point1 is not None:
                    self.state[point1] -= 1
                self.state[point2] += 1
            elif self.state[point2] == -1:
                if point1 is not None:
                    self.state[point1] -= 1
                self.state[point2] = 1
                self.bar[1] += 1
        elif self.turn == -1:
            if point2 is None:
                self.state[point1] += 1
            elif self.state[point2] <= 0:
                if point1 is not None:
                    self.state[point1] += 1
                self.state[point2] -= 1
            elif self.state[point2] == 1:
                if point1 is not None:
                    self.state[point1] += 1
                self.state[point2] = -1
                self.bar[0] += 1

    def is_valid(self, point1, point2, die, tried = False):
        if point1 is None:
            if (self.turn == 1 and -1 <= self.state[point2] < POINT_LIM and point2 + 1 == die or
                self.turn == -1 and -POINT_LIM < self.state[point2] <= 1 and point2 == 24 - die):
                return True
            return False
        if point2 is None:
            if (self.turn == 1 and self.state[point1] > 0 and (point1 + die >= 24 if tried else point1 + die == 24) or
                self.turn == -1 and self.state[point1] < 0 and (point1 - die <= -1 if tried else point1 - die == -1)):
                return True
            return False
        if (point1 == point2 or
            np.sign(self.state[point1]) != self.turn or
            self.state[point1] > 0 and (point1 > point2 or self.turn == -1) or
            self.state[point1] < 0 and (point1 < point2 or self.turn == 1) or
            abs(point1 - point2) != die):
            return False
        if (self.state[point1] > 0 and -1 <= self.state[point2] < POINT_LIM or
            self.state[point1] < 0 and -POINT_LIM < self.state[point2] <= 1):
            return True
        return False

    def possible_moves(self, die):
        moves = []
        #print("Moves at the beginning of possible moves")
        # Move off bar
        if self.turn == 1 and self.bar[0]:
            #print("In move off bar turn 1")
            for point in range(6):
                if self.is_valid(None, point, die):
                    #print("Inside valid")
                    moves.append((None, point))
            return moves
        elif self.turn == -1 and self.bar[1]:
            #print("In move off bar turn -1 ")
            for point in range(18, 24):
                if self.is_valid(None, point, die):
                    #print("Inside valid")
                    moves.append((None, point))
            return moves
        
        # Move off board
        if self.all_checkers_in_end():
            #print("Inside move off board")
            if self.turn == 1:
                for point in range(18, 24):
                    if self.is_valid(point, None, die):
                        #print("Inside valid")
                        moves.append((point, None))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, None, die):
                        #print("Inside valid")
                        moves.append((point, None))

        # Normal moves
        if not moves:
            #print("Inside normal moves")
            for point1 in range(24):
                for point2 in range(24):
                    if self.is_valid(point1, point2, die):
                        #print("Inside valid")
                        #print("Source point: ",point1)
                        #print("Destination point: ", point2)
                        moves.append((point1, point2))

        # Move off board (again)
        if not moves and self.all_checkers_in_end():
            #print("Inside move off board again")
            if self.turn == 1:
                for point in range(18, 24):
                    if self.is_valid(point, None, die, True):
                        #print("Inside valid")
                        moves.append((point, None))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, None, die, True):
                        #print("Inside valid")
                        moves.append((point, None))
    
        return moves
    
    def all_checkers_in_end(self):
        if self.turn == 1:
            for i in range(18):
                if self.state[i] > 0:
                    return False
            return True
        elif self.turn == -1:
            for i in range(6, 24):
                if self.state[i] < 0:
                    return False
            return True
        
    def num_checkers(self):
        if self.turn == 1:
            sum = self.bar[0]
            for point in self.state:
                if point > 0:
                    sum += point
            return sum
        if self.turn == -1:
            sum = self.bar[1]
            for point in self.state:
                if point < 0:
                    sum -= point
            return sum