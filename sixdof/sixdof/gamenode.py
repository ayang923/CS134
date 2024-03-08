# TODO: Finish fixing checker selection, ie making sure the robot does not try
# to pick / place from the same row twice during a move.
    # 3/8 Morning Update: seems to be mostly resolved
# TODO: fix hardcoded placing positions (as good as we can)
    # Still tbd
# TODO: handle_turn() returning illegal sets of moves
    # improvements over night into 3/8 but still not perfect, biggest issue is
    # repeated moves off of the bar
# TODO: (lower priority!) verify game.legal_states functionality
    # gunter said for now assume the opponent makes legal moves. Recognizing our own
    # gripping mistake / using tip camera to adjust our pointing is a more interesting challenge!
# TODO: incorporate ipad/device which reads dice topic to display roll
# TODO: (lower!! priority) write and test GameNode.fix_board()
    # For now, fine to just have it print / log an error

from geometry_msgs.msg  import Pose, PoseArray
from std_msgs.msg import UInt8MultiArray, Bool

from enum import Enum

from rclpy.node         import Node

import numpy as np
import random

from sixdof.utils.TransformHelpers import *

from sixdof.states import *

import copy

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
                                                      self.raise_determine_action_flag, 1)
        
        self.sub_turn = self.create_subscription(Pose, '/turn', self.save_turn, 10)

        self.determine_move_timer = self.create_timer(3, self.determine_action)
        
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
        self.gamestate = np.array([[2,0], [0,0], [0,0], [0,0], [0,0], [0,5],
                                   [0,0], [0,3], [0,0], [0,0], [0,0], [5,0],
                                   [0,5], [0,0], [0,0], [0,0], [3,0], [0,0],
                                   [5,0], [0,0], [0,0], [0,0], [0,0], [0,2],
                                   [0,0]])

        # each entry has a list of [x,y] positions of the checkers in that bar
        self.checker_locations = None
        
        self.scored = 0

        # self.recvgreen populates these arrays with detected green checker pos
        self.greenpos = None
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = None
        # self.recvdice populates this with detected [die1_int, die2_int]
        #self.dice = []
        # self.save turn populates this with turn indicator position
        self.turn_signal_pos = None
        self.turn_signal_dest = [0.20,0.33]

        # Flags
        self.two_colors_row = False # raised if any row other than bar contains more than 0 of both colors
        self.out_of_place = False # raised if any checkers are outside of the row buckets
        self.turn_signal = True # constantly updated indication of who's move. True: robot, False:human
        self.firstmove = True # one-time make sure we move first, false otherwise
        self.determine_action_flag = True # True if the robot is ready for new action, false after actions sent. /move_ready Bool sets to True again

        # Source and destination info
        self.repeat_source = 0
        self.repeat_dest = 0

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
        '''
        dice = []
        for die in msg.data:
            dice.append(die)
        self.dice = dice
        self.game.dice = dice

    def save_turn(self, msg):
        # Save the detected turn signal
        data = p_from_T(T_from_Pose(msg))
        self.turn_signal_pos = [float(data[0]),float(data[1])]
        if self.turn_signal_pos[1] > 0.4:
            self.turn_signal = True
        else:
            self.turn_signal = False

    def save_board_dims(self, msg:Pose):
        self.board_x = msg.position.x
        self.board_y = msg.position.y
        R = R_from_T(T_from_Pose(msg))
        t = np.arctan2(R[1,0],R[0,0]) # undo rotz
        self.board_theta = t

    def raise_determine_action_flag(self, msg:Bool):
        self.determine_action_flag = msg.data

    def update_centers(self):
        centers = np.zeros((25,2))
        grid = np.zeros((25,6,2))
        
        # board dimensions (all in m):

        cx = self.board_x
        cy = self.board_y

        L = 1.061 # board length
        H = 0.536 # board width
        dL = 0.067 # triangle to triangle dist
        dH = 0.050 # checker to checker stack dist

        dL0 = 0.235 # gap from blue side to first triangle center
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

    def sort_checkers(self):
        checker_locations = [[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                             [[],[]], [[],[]]] # last two are bar and unsorted
        
        if self.greenpos is None or self.brownpos is None or self.board_buckets is None:
            # self.get_logger().info("green pos " + str(self.greenpos))
            # self.get_logger().info("brown pos " + str(self.brownpos))
            # self.get_logger().info("board buckets " + str(self.board_buckets))
            return

        for green in self.greenpos:
            sorted = False
            for bucket in self.board_buckets:
                xg = green[0]
                yg = green[1]
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                if ((xg >= xmin and xg <= xmax) and (yg >= ymin and yg <= ymax)):
                    checker_locations[bucket_ind][0].append(green)
                    sorted = True
            if sorted == False:
                checker_locations[25][0].append(green)

        for brown in self.brownpos:
            sorted = False
            for bucket in self.board_buckets:
                xb = brown[0]
                yb = brown[1]
                xmin = bucket[0] - 0.03
                xmax = bucket[0] + 0.03
                ymin = bucket[1] - 0.13
                ymax = bucket[1] + 0.13
                bucket_ind = np.where(self.board_buckets == bucket)[0][0]
                if ((xb >= xmin and xb <= xmax) and (yb >= ymin and yb <= ymax)):
                    checker_locations[bucket_ind][1].append(brown)
                    sorted = True
            if sorted == False:
                checker_locations[25][1].append(brown)

        # Check that we have the right amount of checkers!

        total = 0
        for i in np.arange(25):
            greencount = len(checker_locations[i][0])
            browncount = len(checker_locations[i][1])
            if (greencount != 0 and browncount !=0) and i != 24:
                self.two_colors_row = True # raise flag that a row has more than one color (not bar)
            total += greencount + browncount
        total += self.scored
        if total != 30:
            if total + len(checker_locations[25][0]) + len(checker_locations[25][1]) == 30:
                self.out_of_place = True # raise flag that a checker is mispositioned wrt the triangles
            else:
                return None # don't update, something is changing/blocked        
        for i in np.arange(25):
            greencount = len(checker_locations[i][0])
            browncount = len(checker_locations[i][1])            
            self.gamestate[i] = [greencount,browncount] # this will only be updated if none of the above flags were raised.
            # if a flag was raised, we must have the last gamestate stored.
        self.checker_locations = checker_locations

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
        if self.board_buckets is None or self.checker_locations is None:
            self.get_logger().info("board buckets "+str(self.board_buckets))
            self.get_logger().info("checker location "+str(self.checker_locations))
            self.get_logger().info('no data')
            self.pub_checker_move.publish(PoseArray())
            return

        #self.get_logger().info('determine action running')
        
        if self.turn_signal: # checks whether it is our turn
            self.get_logger().info("Robot (Green) Turn!")
            #Stops sending action commands with determine action after received an action
            if not self.determine_action_flag:
                self.get_logger().info('Already sent an action!')
                return None
        else:
            self.get_logger().info("Human Turn!")
            return None # comment out if having robot play against itself

        
        checkgame = Game(self.gamestate) # save the detected gamestate as a game object for comparison against the old gamestate
        #if ((checkgame.state in self.game.legal_next_states(self.game.dice)) or self.firstmove): # Check that self.gamestate is a legal progression from last state given roll
        if True:
            # FIXME: remove "or True" if we can get legal states working
            if self.firstmove:
                self.firstmove = False
            
            self.game.set_state(self.gamestate) # update game.state
            self.get_logger().info("Gamestate is:" + str(self.gamestate))
            self.get_logger().info("Game state is" + str(self.game.state))
            moves = self.handle_turn(self.game) # roll dice and decide moves (with new gamestate)
            self.get_logger().info("After Handle Turn Game state is" + str(self.game.state))

            # Debugs
            self.get_logger().info("Robot turn dice roll: {}".format(self.game.dice))
            # TODO: publish dice roll display to ipad through topic?
            self.get_logger().info("Robot chosen Moves: {}".format(moves))

            sources = []
            dests = []
            for (source,dest) in moves:
                # if source in dests: go to hardcoded center number (repeat - 1) of row source
                self.repeat_source = sources.count(source) + sources.count(dests)
                self.repeat_dest = dests.count(dest) + dests.count(source)
                sources.append(source)
                dests.append(dest)
                self.get_logger().info("Moving from {} to {}".format(source, dest))
                self.get_logger().info("Source number"+str((self.game.state[source])))
                self.get_logger().info("Dest number" + str(self.game.state[dest]))
                if source == 24:
                    if (np.sign(self.game.state[source][0]) != np.sign(self.game.state[dest]) and 
                    self.game.state[dest] != 0):
                        self.get_logger().info("Hit off bar!!")
                        self.execute_hit(source, dest)
                    else:
                        self.execute_normal(source, dest)
                    # self.execute_off_bar(dest)
                elif dest == 24:
                    self.execute_bear_off(source)
                elif (np.sign(self.game.state[source]) != np.sign(self.game.state[dest]) and 
                    self.game.state[dest] != 0):
                    self.get_logger().info("Hit!!")
                    self.execute_hit(source, dest)
                else:
                    self.execute_normal(source, dest)
            self.publish_checker_move(self.turn_signal_pos,self.turn_signal_dest) # move the turn signal to indicate human turn
            self.determine_action_flag = False
            self.game.roll() # next roll (for human)
            self.get_logger().info('Human roll is: ' + str(self.game.dice))
            # TODO: publish this roll to a topic so it can be displayed on an ipad?

                #self.game.move(source, dest) # comment out to rely only on detectors for gamestate
                #self.get_logger().info("Gamestate after move:"+str(self.game.state))
                #self.get_logger().info("exectued the move")
            #self.game.turn *= -1 # comment out if robot only plays as one color
        else:
            self.get_logger().info('Fixing board!!')
            self.fix_board() # TODO!!

        #self.get_logger().info("Gamestate:"+str(self.game.state))
    
    def fix_board(self):
        # TODO!!!
        pass
    
    def execute_off_bar(self, dest):
        turn = 0 if self.game.turn == 1 else 1
        bar = 24

        source_pos = self.last_checker(bar,turn, self.repeat_source)
        dest_pos = self.next_free_place(dest, turn, self.repeat_dest)
        
        self.game.move(24,dest)

        self.publish_checker_move(source_pos, dest_pos)

    def execute_bear_off(self, source):
        turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source,turn, self.repeat_source)
        dest_pos = [0.5,0.3]
        

        self.publish_checker_move(source_pos, dest_pos)
    
    def execute_hit(self, source, dest):
        # self.game.turn == 1 during robot (green) turn
        turn = self.game.turn
        bar = 24
        
        dest_centers = self.grid_centers[dest]

        # there is a problem here with how the new last checkers work 
        source_pos_1 = self.last_checker(dest, turn, repeat=0) # grab solo checker
        dest_pos = self.next_free_place(bar, turn, repeat=0)
        self.game.move(dest,bar)

        self.publish_checker_move(source_pos_1, dest_pos) # publish move
        turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source, turn, self.repeat_source) # grab my checker
        dest_pos = source_pos_1 # and move where I just removed
        self.game.move(source,dest)

        self.publish_checker_move(source_pos, dest_pos)

    def execute_normal(self, source, dest):        
        turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source,turn,self.repeat_source)
        dest_pos = self.next_free_place(dest,turn,self.repeat_dest)
        self.game.move(source,dest)

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
        bar = 24
        gamecopy = Game(self.gamestate)
        moves = []
        game.roll()
        final_moves = []
        #print(self.game.state)
        if game.dice[0] == game.dice[1]:
            for _ in range(4):
                moves = gamecopy.possible_moves(game.dice[0])
                move = self.choose_move(gamecopy, moves)
                if move is not None:
                    final_moves.append(move)
                    if move[0] == 24: # off bar
                        if np.sign(gamecopy.state[move[0]][0]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                            # hit off bar
                            gamecopy.move(move[1],bar)
                            gamecopy.move(move[0],move[1])
                        else:
                            gamecopy.move(move[0],move[1])
                    elif np.sign(gamecopy.state[move[0]]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                        # normal hit
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    else:
                        gamecopy.move(move[0],move[1])
        else:
            # larger = 1
            # if game.dice[0] > game.dice[1]:
            #     larger = 0
            moves = gamecopy.possible_moves(game.dice[0])
            move = self.choose_move(gamecopy, moves)
            if move is not None:
                final_moves.append(move)
                if move[0] == 24: # off bar
                    if np.sign(gamecopy.state[move[0]][0]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                        # hit off bar
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    else:
                        gamecopy.move(move[0],move[1]) # bar to chosen (not a hit)
                elif np.sign(gamecopy.state[move[0]]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                    # normal hit
                    gamecopy.move(move[1],bar)
                    gamecopy.move(move[0],move[1])
                else:
                    gamecopy.move(move[0],move[1])
                
            moves = gamecopy.possible_moves(game.dice[1])
            move = self.choose_move(gamecopy, moves)
            if move is not None:
                final_moves.append(move)
                if move[0] == 24: # off bar
                    if np.sign(gamecopy.state[move[0]][0]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                        # hit off bar
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    else:
                        gamecopy.move(move[0],move[1])
                elif np.sign(gamecopy.state[move[0]]) != np.sign(gamecopy.state[move[1]]) and gamecopy.state[move[1]] != 0:
                    # normal hit
                    gamecopy.move(move[1],bar)
                    gamecopy.move(move[0],move[1])
                else:
                    gamecopy.move(move[0],move[1])

            #print("Moves in handle turn:", final_moves)

        return final_moves

    def next_free_place(self,row,color,repeat):
        # color is 0 or 1 for the turn, ie green 0 brown 1
        positions = self.grid_centers[row]
        num_dest = self.gamestate[row][color] + repeat
        if row == 24:
            num_dest += self.gamestate[row][0 if color == 1 else 1]        
        return positions[num_dest]


    def last_checker(self,row,color,repeat):
        '''
        Get the [x,y] of the last checker in the row (closest to middle)
        Repeat shoudl be set to zero if the row is not accessed more then
        once in one players turn
        '''
        #self.get_logger().info('checker locations' + str(self.checker_locations))
        if row <= 11:
            sorted_positions = sorted(self.checker_locations[row][color], key=lambda x: x[1])
        elif row <=24:
            sorted_positions = sorted(self.checker_locations[row][color], key=lambda x: x[1], reverse=True)
        #self.get_logger().info("Repeat:"+str(repeat))
        #self.get_logger().info("sorted_pos"+str(sorted_positions))
        #self.get_logger().info("Choosen position:" + str(sorted_positions[repeat]))
        # When the position is the bar
            
        if not sorted_positions: # if an empty row because we moved a checker into a place where there was previosly 0
            # catching two different situations:
            # 1. in a hit
            # 2. consecutive moves: 
            return self.grid_centers[row][5 - repeat] # be aware this may not work in both situations?
        
        return sorted_positions[repeat]
    
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
        #INITIALIZATION OF BAR
        self.bar = [0, 0]
        self.dice = [0, 0]
        self.turn = 1
        self.done = False
        self.clicked = None

    # Converts state in GameDriver format to state in engine format.
    def set_state(self, gamestate):
        self.state = []
        # BAR USED AGAIN AS GAME STATE 24
        self.bar = gamestate[24]
        for point in gamestate[:24]:
            # If the column has a green
            if point[0]:
                self.state.append(point[0])
            # If the columan has a brown
            else:
                self.state.append(-point[1])
        self.state.append(self.bar)
        #self.state = np.append(self.state[11::], self.state[:11:]).tolist()

    def roll(self):
        self.dice = np.random.randint(1, 7, size = 2).tolist()
    
    def move(self, point1, point2):
        if point1 == 24:
            self.bar[0 if self.turn == 1 else 1] -= 1
        # The move turn of player One
        if self.turn == 1:
            if point2 == 24:
                self.state[point1] -= 1
            elif self.state[point2] >= 0:
                if point1 != 24:
                    self.state[point1] -= 1
                self.state[point2] += 1
            elif self.state[point2] == -1:
                if point1 != 24:
                    self.state[point1] -= 1
                else:
                    self.state[24][0] -= 1
                self.state[point2] = 1
                self.state[24][1] += 1
        # The move turn of player Two
        elif self.turn == -1:
            if point2 == 24:
                self.state[point1] += 1
            elif self.state[point2] <= 0:
                if point1 != 24:
                    self.state[point1] += 1
                self.state[point2] -= 1
            elif self.state[point2] == 1:
                if point1 != 24:
                    self.state[point1] += 1
                else:
                    self.state[24][1] += 1
                self.state[point2] = -1
                self.state[24][0] += 1

    def is_valid(self, point1, point2, die, tried = False):
        if point1 == 24:
            if (self.turn == 1 and -1 <= self.state[point2] < POINT_LIM and point2 + 1 == die or
                self.turn == -1 and -POINT_LIM < self.state[point2] <= 1 and point2 == 24 - die):
                return True
            return False
        if point2 == 24:
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
        if self.turn == 1 and self.state[24][0] > 0:
            print('move off bar in possible moves')
            print('self.state[24]',self.state[24])
            #print("In move off bar turn 1")
            for point in range(6):
                if self.is_valid(24, point, die):
                    #print("Inside valid")
                    moves.append((24, point))
            return moves
        elif self.turn == -1 and self.state[24][1] < 0:
            #print("In move off bar turn -1 ")
            for point in range(18, 24):
                if self.is_valid(24, point, die):
                    #print("Inside valid")
                    moves.append((24, point))
            return moves
        
        # Move off board
        if self.all_checkers_in_end():
            #print("Inside move off board")
            if self.turn == 1:
                for point in range(18, 24):
                    if self.is_valid(point, 24, die):
                        #print("Inside valid")
                        moves.append((point, 24))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, 24, die):
                        #print("Inside valid")
                        moves.append((point, 24))

        # Normal moves
        if not moves:
            # print("Inside normal moves")
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
                    if self.is_valid(point, 24, die, True):
                        #print("Inside valid")
                        moves.append((point, 24))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, 24, die, True):
                        #print("Inside valid")
                        moves.append((point, 24))
    
        return moves

    # def is_legal_transition(self,newgame,dice):
    #     [die1, die2] = dice
    #     changed_rows = []
    #     i = 0
    #     for old_row_count, new_row_count in zip(self.state,newgame.state):
    #         if old_row_count != new_row_count:
    #             changed_rows.append([i,new_row_count-old_row_count])
    #         i += 1
    #     # now we have a list full of [changed_row_number,delta of checker numbers]
    #     # this is now a list of integer rows which have more than they did before
    #     for row in changed_rows:
    #         # check if either of the dice could get it there
    #         # if a die can get you from this row to 
    #         for die in [die1,die2]:
    #             for lowrow in lower_rows:
    #                 if lowrow + die == highrow
                
    #         # check if both can get it there
            
    #         # also handle when we have doubles??
    #         if die1 == die2:
    #             pass
    #     pass

    def legal_next_states(self, dice):
        # TODO: make sure this actually works!!        
        [die1, die2] = dice
        states = []
        if die1 == die2: # four moves, check all possible resulting states
            moves1 = self.possible_moves(die1)
            print('possiblemoves1', moves1)
            for move1 in moves1:
                copy1 = copy.copy(self)
                copy1.move(move1)
                moves2 = copy1.possible_moves(die1)
                for move2 in moves2:
                    copy2 = copy.copy(copy1)
                    copy2.move(move2)
                    moves3 = copy2.possible_moves(die1)
                    for move3 in moves3:
                        copy3 = copy.copy(copy2)
                        copy3.move(move3)
                        moves4 = copy3.possible_moves(die1)
                        for move4 in moves4:
                            copy4 = copy.copy(copy3)
                            copy4.move(move4)
                            if copy4.state not in states:
                                states.append(copy4.state)
        else:
            # find all states that could result from using die 1 then die 2
            moves = self.possible_moves(die1)
            for move1 in moves1:
                copy1 = copy.copy(self)
                copy1.move(move1)
                moves2 = copy1.possible_moves(die2)
                for move2 in moves2:
                    copy2 = copy.copy(copy1)
                    copy2.move(move2)
                    if copy2.state not in states: # only add if not already in states
                        states.append(copy2.state)
            # find all states that could result from using die 2 then die 1
            moves = self.possible_moves(die2)
            for move1 in moves1:
                copy1 = copy.copy(self)
                copy1.move(move1)
                moves2 = copy1.possible_moves(die1)
                for move2 in moves2:
                    copy2 = copy.copy(copy1)
                    copy2.move(move2)
                    if copy2.state not in states: # only add if not already in states
                        states.append(copy2.state)
        return states        
        

    
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
            sum = self.state[24][0]
            for point in self.state[:24]:
                if point > 0:
                    sum += point
            return sum
        if self.turn == -1:
            sum = self.state[24][1]
            for point in self.state[:24]:
                if point < 0:
                    sum -= point
            return sum
        
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