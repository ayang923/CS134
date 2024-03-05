from geometry_msgs.msg  import Pose, PoseArray
from std_msgs.msg import UInt8MultiArray, Bool

from enum import Enum

from rclpy.node         import Node

import numpy as np
import random

from sixdof.utils.TransformHelpers import *

from sixdof.states import *

class Color(Enum):
    GREEN = 1
    BROWN = 2

# class passed into trajectory node to handle game logic
class GameNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # /boardpose from Detector
        self.sub_board = self.create_subscription(Pose, '/boardpose',
                                                                  self.recvboard, 3)

        # Poses of all detected green checkers, /green from Detector
        self.sub_green = self.create_subscription(PoseArray, '/green',
                                                                  self.recvgreen, 3)

        # Poses of all detected brown checkers, /brown from Detector
        self.sub_brown = self.create_subscription(PoseArray, '/brown',
                                                                  self.recvbrown, 3)

        # /dice Unsigned int array with value of two detected dice from Detector
        self.sub_dice = self.create_subscription(UInt8MultiArray, '/dice',
                                                                 self.recvdice, 3)
        
        self.sub_moveready = self.create_subscription(Bool, '/move_ready',
                                                      self.determine_action, 1)
        
        self.determining = False
        
        self.pub_clear = self.create_publisher(Bool, '/clear', 10)
        self.pub_checker_move = self.create_publisher(PoseArray, '/checker_move', 10)
        self.pub_dice_roll = self.create_publisher(PoseArray, '/dice_roll', 10)

        # Physical Game Board Info
        self.board_x = None
        self.board_y = None
        self.board_theta = None # radians
        self.board_buckets = None # center locations [x,y]
        self.grid_centers = None # for placing

        # Initial gamestate area assumes setup for beginning of game
        # each element indicates [num_green, num_brown]
        # beginning to end of array progresses ccw from robot upper right.
        # last item is middle bar
        self.gamestate = np.array([[5,0], [0,0], [0,0], [0,0], [0,3], [0,0],
                                   [0,5], [0,0], [0,0], [0,0], [0,0], [2,0],
                                   [0,2], [0,0], [0,0], [0,0], [0,0], [5,0],
                                   [0,0], [3,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0]])

        # each entry has a list of [x,y] positions of the checkers in that bar
        # right now basically redundant with gamestate
        self.checker_locations = None
        
        self.scored = 0

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
        self.save_board_dims(msg)
        self.update_centers()

    def recvgreen(self, msg:PoseArray):
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
        self.sort_checkers()

    def recvbrown(self, msg:PoseArray):
        '''
        same as above, but for brown
        '''
        self.brownpos = []
        for pose in msg.poses:
            xy = [pose.position.x, pose.position.y]
            self.brownpos.append(xy)
        self.brownpos = np.array(self.brownpos)
        self.sort_checkers()

    def recvdice(self, msg):
        '''
        process most recent /dice msg

        in: /dice UInt8MultiArray

        places values in self.dice
        TODO
        '''
        pass

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
        dH = 0.040 # checker to checker stack dist

        dL0 = 0.247 # gap from blue side to first triangle center
        dL1 = 0.117 - dL # gap between two sections of triangles (minus dL)

        for i in np.arange(6):
            x = cx + L/2 - dL0 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy + H/2 - dH/2 - j*dH
                grid[i][j] = [x,y]

        for i in np.arange(6,12):
            x = cx + L/2 - dL0 - dL1 - i*dL
            y = cy + H/2 - dH/2 - 2.5*dH
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy + H/2 - dH/2 - j*dH
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
            y = cy - H/2 + dH/2 + 2.5*dH
            centers[i] = [x,y]
            for j in np.arange(6):
                y = cy - H/2 + dH/2 + j*dH
                grid[i][j] = [x,y]
                
        x = cx + L/2 - dL0 - 5*dL - (dL1+dL)/2
        y = cy
        centers[24] = [x,y] # bar
        for j in np.arange(6):
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

    def sort_checkers(self):
        checker_locations = [[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]]]
        
        if self.greenpos is None or self.brownpos is None or self.board_buckets is None:
            return

        for green in self.greenpos:
            for bucket in self.board_buckets:
                xg = green[0]
                yg = green[1]
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                if ((xg >= xmin and xg <= xmax) and (yg >= ymin and yg <= ymax)):
                    bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                    checker_locations[bucket_ind][0].append(green)
        for brown in self.brownpos:
            for bucket in self.board_buckets:
                xb = brown[0]
                yb = brown[1]
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                if ((xb >= xmin and xb <= xmax) and (yb >= ymin and yb <= ymax)):
                    bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                    checker_locations[bucket_ind][1].append(brown)

        # Check that we have the right amount of checkers!

        total = 0
        for i in np.arange(25):
            greencount = len(checker_locations[i][0])
            browncount = len(checker_locations[i][1])
            if (greencount != 0 and browncount !=0) and i != 24:
                return None # don't update, something is changing or bad data
            total += greencount + browncount
        total += self.scored
        if total != 30:
            return None # don't update, something is changing
        else:
            for i in np.arange(25):
                greencount = len(checker_locations[i][0])
                browncount = len(checker_locations[i][1])            
                self.gamestate[i] = [greencount,browncount]
            self.checker_locations = checker_locations

    def determine_action(self, msg):
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
        if self.board_buckets is None or self.checker_locations is None:
            self.get_logger().info('no data')
            return

        self.get_logger().info('determine action running')
        
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
            elif (np.sign(self.game.state[source]) != np.sign(self.game.state[dest]) and 
                  self.game.state[dest] != 0):
                self.execute_hit(source, dest)
            else:
                self.execute_normal(source, dest)
            self.game.move(source, dest)
            self.get_logger().info("exectued the move")
        self.game.turn *= -1
    
    def execute_off_bar(self, dest):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        dest_centers = self.grid_centers[dest] # list of coordinates for dest row

        num_dest = self.gamestate[dest][turn] # basically next empty for the dest row

        source_pos = self.last_checker(bar,turn)
        dest_pos = dest_centers[num_dest]

        self.publish_checker_move(source_pos, dest_pos)

    def execute_bear_off(self, source):
        turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source,turn)
        dest_pos = [0.5,0.3]

        self.publish_checker_move(source_pos, dest_pos)
    
    def execute_hit(self, source, dest):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        dest_centers = self.grid_centers[dest]
        bar_centers = self.grid_centers[bar]

        num_bar = self.game.bar[0 if self.game.turn == 1 else 1]

        source_pos = self.last_checker(dest,turn) # grab solo checker
        dest_pos = bar_centers[5 - num_bar if turn else num_bar] # and move to bar

        self.publish_checker_move(source_pos, dest_pos) # publish move

        source_pos = self.last_checker(source,turn) # grab my checker
        dest_pos = dest_centers[0] # and move where I just removed

        self.publish_checker_move(source_pos, dest_pos)

    def execute_normal(self, source, dest):        
        turn = 0 if self.game.turn == 1 else 1

        dest_centers = self.grid_centers[dest]
        
        num_dest = self.gamestate[dest][turn]

        source_pos = self.last_checker(source,turn)
        dest_pos = dest_centers[num_dest]

        self.publish_checker_move(source_pos, dest_pos)
    
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
        final_moves = []

        print("Dice: {}".format(game.dice))

        if game.dice[0] == game.dice[1]:
            for _ in range(4):
                moves = game.possible_moves(game.dice[0])
                move = self.choose_move(game, moves)
                if move is not None:
                    final_moves.append(move)
        else:
            # larger = 1
            # if game.dice[0] > game.dice[1]:
            #     larger = 0
            moves = game.possible_moves(game.dice[0])
            move = self.choose_move(game, moves)
            if move is not None:
                final_moves.append(move)
            moves = game.possible_moves(game.dice[1])
            move = self.choose_move(game, moves)
            if move is not None:
                final_moves.append(move)
            print("Moves in handle turn:", final_moves)

        return final_moves

    def last_checker(self,row,color):
        '''
        Get the [x,y] of the last checker in the row (closest to middle)
        '''
        positions = self.checker_locations[row][color]
        mindist = np.inf
        min_index = None
        i = 0
        for position in positions:
            dist = np.sqrt((position[0]-self.board_x)**2 + (position[1]-self.board_y)**2)
            if dist < mindist:
                mindist = dist
                min_index = i
            i += 1
        return self.checker_locations[row][color][min_index]
        
    def publish_checker_move(self, source, dest):
        msg = PoseArray()
        for xy in [source, dest]:
            p = pxyz(xy[0],xy[1],0.01)
            R = Reye()
            msg.poses.append(Pose_from_T(T_from_Rp(R,p)))
        self.pub_checker_move.publish(msg)

    def publish_dice_roll(self,source,dest):
        pass

    def publish_clear(self):
        msg = Bool()
        msg.data = True
        self.pub_clear.publish(msg)

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
            # If the column has a green
            if point[0]:
                self.state.append(point[0])
            # If the columan has a brown
            else:
                self.state.append(-point[1])
        #self.state = np.append(self.state[11::], self.state[:11:]).tolist()

    def roll(self):
        self.dice = np.random.randint(1, 7, size = 2).tolist()
    
    def move(self, point1, point2):
        if point1 is None:
            self.bar[0 if self.turn == 1 else 1] -= 1
        # The move turn of player One
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
        # The move turn of player Two
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
            print("Inside normal moves")
            for point1 in range(24):
                for point2 in range(24):
                    if self.is_valid(point1, point2, die):
                        print("Inside valid")
                        print("Source point: ",point1)
                        print("Destination point: ", point2)
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
        
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = GameNode('traj')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()