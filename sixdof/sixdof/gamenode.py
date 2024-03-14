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

import rclpy
import numpy as np
from sensor_msgs.msg    import Image
from cv_bridge          import CvBridge
import cv2
import random

from sixdof.utils.TransformHelpers import *

from sixdof.states import *
from sixdof.backgammon import Render

import copy

class Color(Enum):
    GREEN = 1
    BROWN = 2

# Initial Board States
STANDARD = [[2,0], [0,0], [0,0], [0,0], [0,0], [0,5],
                     [0,0], [0,3], [0,0], [0,0], [0,0], [5,0],
                     [0,5], [0,0], [0,0], [0,0], [3,0], [0,0],
                     [5,0], [0,0], [0,0], [0,0], [0,0], [0,2],
                     [0,0], [0,0]]

BEAR_OFF = [[0,3], [0,3], [0,3], [0,3], [0,3], [0,0],
                     [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                     [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
                     [0,0], [3,0], [3,0], [3,0], [3,0], [3,0],
                     [0,0], [0,0]]
                     

def reconstruct_gamestate_array(flattened_lst):
    return np.array(flattened_lst).reshape((26, 2)).tolist() if len(flattened_lst) == 52 else None

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

        self.robot_move_sub = self.create_subscription(UInt8MultiArray, '/hardcode_robot_move', self.hardcode_move, 3)

        self.create_subscription(UInt8MultiArray, '/board_state', self.recvstate, 3)
        
        self.sub_turn = self.create_subscription(Pose, '/turn', self.save_turn, 10)

        self.determine_move_timer = self.create_timer(2, self.determine_action)
        
        self.determining = False
        
        self.pub_clear = self.create_publisher(Bool, '/clear', 10)
        self.pub_checker_move = self.create_publisher(PoseArray, '/checker_move', 10)
        self.pub_dice_roll = self.create_publisher(PoseArray, '/dice_roll', 10)
        self.dice_publisher = self.create_publisher(Image, '/dice_roll_image', 10)
        self.dice_image_timer = self.create_timer(1, self.publish_dice_roll)
        self.bridge = CvBridge()

        # Physical Game Board Info
        self.board_x = None
        self.board_y = None
        self.board_theta = None # radians
        self.board_buckets = None # center locations [x,y]
        self.grid_centers = None # for placing

        # Choose initial state
        # Initial gamestate area assumes setup for beginning of game
        # each element indicates [num_green, num_brown]
        # beginning to end of array progresses ccw from robot upper right.
        # last item is middle bar
        self.gamestate = STANDARD

        # each entry has a list of [x,y] positions of the checkers in that bar
        self.checker_locations = None
        
        self.scored = 0

        # self.recvgreen populates these arrays with detected green checker pos
        self.greenpos = None
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = None
        # self.recvdice populates this with detected [die1_int, die2_int]
        self.dice = []
        # self.save turn populates this with turn indicator position
        self.turn_signal_pos = None
        self.turn_signal_dest = [0.20,0.33]

        # Flags
        self.two_colors_row = False # raised if any row other than bar contains more than 0 of both colors
        self.out_of_place = False # raised if any checkers are outside of the row buckets
        self.turn_signal = True # constantly updated indication of who's move. True: robot, False:human
        self.firstmove = True # one-time make sure we move first, false otherwise
        self.determine_action_flag = False # True if the robot is ready for new action, false after actions sent. /move_ready Bool sets to True again
        self.first_human_turn_report = False # logging flag

        # Source and destination info
        self.repeat_source = 0
        self.repeat_dest = 0

        self.curr_move_src_counter = self.clear_move_counter()
        self.curr_move_dest_counter = self.clear_move_counter()

        self.curr_move_checker_pushes = self.empty_checker_count()
        self.curr_move_tops = self.empty_checker_count()

        # Game engine
        self.game = Game(self.gamestate)
        # self.render = Render(self.game)
        # self.render.draw()
        # self.render.update()
    
    def recvstate(self, msg: UInt8MultiArray):
        flattened_lst = list(msg.data)
        reconstructed_lst = reconstruct_gamestate_array(flattened_lst)
        
        #self.get_logger().info("board state " + str(len(flattened_lst)))
        if reconstructed_lst:
            self.gamestate = reconstructed_lst
    
    def hardcode_move(self, msg):
        self.determine_action_flag = False
        source, dest = msg.data[0], msg.data[1]

        self.execute_normal(source, dest)

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
        # self.get_logger().info("self.gamestate in det_action cbk" + str(self.gamestate))
        # checkgame = Game(self.gamestate)
        
        # green_fixes, brown_fixes = self.fix_board(checkgame.state, self.game.state)
        # self.get_logger().info('fixes in raise_det_action_flag callback' + str(green_fixes) + str(brown_fixes))

        # if self.firstmove and not self.turn_signal:
        #     self.publish_checker_move(self.turn_signal_pos, [0.20, 0.47])
        #     self.determine_action = False
        #     return None

        # if self.turn_signal and (brown_fixes or green_fixes):
        #     self.execute_fixes(green_fixes, brown_fixes, checkgame)
        #     self.determine_action_flag = False

        #     if (True if self.game.turn==1 else False) != self.turn_signal:
        #         self.publish_checker_move(self.turn_signal_pos, self.turn_signal_dest)
        #         self.determine_action_flag = False
            
        # else:
        #     self.determine_action_flag = msg.data
        self.determine_action_flag = False # comment out for not debugs

    def fix_board(self, actual_state, expected_state):
        actual_state_copy = copy.deepcopy(actual_state)
        expected_state_copy = copy.deepcopy(expected_state)
        green_incorrect = []
        brown_incorrect = []
        green_moves = []
        brown_moves = []

        for i, (triangle_actual, triangle_expected) in enumerate(zip(actual_state_copy[:24], expected_state_copy[:24])):
            if np.sign(triangle_actual) == np.sign(triangle_expected)*-1:
                if np.sign(triangle_actual) > 0:
                    green_moves.append([i, 24] * abs(triangle_actual))
                    actual_state_copy[i] = 0
                    actual_state_copy[24][0] += abs(triangle_actual)
                elif np.sign(triangle_actual) < 0:
                    brown_moves.append([i, 24] * abs(triangle_actual))
                    actual_state_copy[i] = 0
                    actual_state_copy[24][1] += abs(triangle_actual)

        for i, (triangle_actual, triangle_expected) in enumerate(zip(actual_state_copy[:25], expected_state_copy[:25])):
            # print(triangle_actual)
            # print(triangle_expected)
            # print(i)
            if triangle_actual != triangle_expected:
                if i < 24:
                    brown_actual = -triangle_actual if triangle_actual < 0 else 0
                    brown_expected = -triangle_expected if triangle_expected < 0 else 0
                    green_actual = triangle_actual if triangle_actual > 0 else 0
                    green_expected = triangle_expected if triangle_expected > 0 else 0
                else:
                    green_expected = triangle_expected[0]
                    green_actual = triangle_actual[0]
                    brown_expected = triangle_expected[1]
                    brown_actual = triangle_actual[1]

                if green_actual != green_expected:
                    delta_green = green_expected - green_actual
                    if len(green_incorrect) != 0 and np.sign(delta_green) != np.sign(green_incorrect[-1]):
                        for _ in range(abs(delta_green)):
                            if len(green_incorrect) != 0:
                                green_moves.append([i, abs(green_incorrect.pop(-1))-1] if np.sign(delta_green) == -1 else [abs(green_incorrect.pop(-1))-1, i])
                    else:
                        green_incorrect = [np.sign(delta_green)*(i+1)] * abs(delta_green) + green_incorrect

                elif brown_actual != brown_expected:
                    delta_brown = brown_expected - brown_actual
                    if len(brown_incorrect) != 0 and np.sign(delta_brown) != np.sign(brown_incorrect[-1]):
                        for _ in range(abs(delta_brown)):
                            if len(brown_incorrect) != 0:
                                brown_moves.append([i, abs(brown_incorrect.pop(-1))-1] if np.sign(delta_brown) == -1 else [abs(brown_incorrect.pop(-1))-1,i])
                    else:
                        brown_incorrect = [np.sign(delta_brown)*(i+1)] * abs(delta_brown) + brown_incorrect
        return green_moves, brown_moves
    
    def publish_dice_roll(self):
        width, height = 640, 480
        dice_img = np.zeros((height, width, 3), dtype=np.uint8)
        font_scale = 10.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.game.dice is not None:
            dice_roll = str(self.game.dice[0]) + ',' + str(self.game.dice[1])
            cv2.putText(dice_img, dice_roll, (50, 300), font, font_scale, (255, 255, 255), 6, cv2.LINE_AA)

        dice_roll_image = self.bridge.cv2_to_imgmsg(dice_img, encoding='bgr8')
        self.dice_publisher.publish(dice_roll_image)

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
        checker_locations = self.empty_checker_count() # last two are bar and unsorted
        
        if self.greenpos is None or self.brownpos is None or self.board_buckets is None:
            # self.get_logger().info("green pos " + str(self.greenpos))
            # self.get_logger().info("brown pos " + str(self.brownpos))
            # self.get_logger().info("board buckets " + str(self.board_buckets))
            return

        for green in self.greenpos:
            in_triangle = False
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
                    in_triangle = True
            if not in_triangle:
                checker_locations[25][0].append(green)

        for brown in self.brownpos:
            in_triangle = False
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
                    in_triangle= True
            if not in_triangle:
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

        # TODO Need to implement the 25th row
        for triangle in range(25):
            if triangle <= 11:
                checker_locations[triangle][0] = sorted(checker_locations[triangle][0], key=lambda x: x[1])
                checker_locations[triangle][1] = sorted(checker_locations[triangle][1], key=lambda x: x[1])

            elif triangle <=24:
                checker_locations[triangle][0] = sorted(checker_locations[triangle][0], key=lambda x: x[1], reverse=True)
                checker_locations[triangle][1] = sorted(checker_locations[triangle][1], key=lambda x: x[1], reverse=True)
        self.checker_locations = checker_locations

    def determine_action(self):
        '''
        given current knowledge of the gameboard + current state, figure out
        what to do next?
        this is where all of the actual "game logic" is held? basically a black
        box rn but this is probably the abstraction we eventually want
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
        
        # self publish robot moves for debugging

        if self.turn_signal: # checks whether it is our turn
            if not self.first_human_turn_report:
                self.get_logger().info("Robot (Green) Turn!")
                #Stops sending action commands with determine action after received an action
                self.first_human_turn_report = True
            if not self.determine_action_flag: # only valid in non testing mode
                # uncomment if running normal, comment if testing
                # self.get_logger().info('Already sent an action!')
                return None
        else:
            if self.first_human_turn_report:
                self.get_logger().info("Human Turn!")
                self.first_human_turn_report = False
            return None # comment out if having robot play against itself

        
        checkgame = Game(self.gamestate) # save the detected gamestate as a game object for comparison against the old gamestate
        self.get_logger().info('checkgame.state' + str(checkgame.state))
        self.get_logger().info('GameNode.game.state' + str(self.game.state))
        [low_row, high_row] = self.get_changed_bounds(checkgame) # as far as I can tell seems functional

        self.get_logger().info('game turn' + str(self.game.turn))
        self.get_logger().info('self.game.dice: ' +  str(self.game.dice))
        self.get_logger().info('legal next states' + str(self.game.legal_next_states(self.game.dice)))
        if ((checkgame.state in self.game.legal_next_states(self.game.dice)) or self.firstmove or not self.changed): # Check that self.gamestate is a legal progression from last state given roll
            # FIXME: remove "or True" if we can get legal states working
            if self.firstmove:
                self.firstmove = False
            else:
                self.game.turn *= -1 # from human turn to robot turn
            
            self.game.set_state(self.gamestate) # update game.state
            moves = self.handle_turn(self.game) # roll dice and decide moves (with new gamestate)
            # self.render.draw()
            # self.render.update()
            self.get_logger().info("After Handle Turn Game state is" + str(self.game.state))

            # Debugs
            self.get_logger().info("Robot turn dice roll: {}".format(self.game.dice))
            # TODO: publish dice roll display to ipad through topic?
            self.get_logger().info("Robot chosen Moves: {}".format(moves))

            sources = []
            dests = []
            for move in moves:
                source, dest = move[0], move[1]
                # if source in dests: go to hardcoded center number (repeat - 1) of row source
                self.repeat_source = sources.count(source) 
                self.repeat_dest = dests.count(dest) 
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
                elif dest == 25: #moving off game with bear off
                    self.execute_bear_off(source)
                elif (np.sign(self.game.state[source]) != np.sign(self.game.state[dest]) and 
                    self.game.state[dest] != 0):
                    self.get_logger().info("Hit!!")
                    self.execute_hit(source, dest)
                else:
                    self.execute_normal(source, dest)
            self.get_logger().info('game.state after sending moves: ' + str(self.game.state))
            self.publish_checker_move(self.turn_signal_pos,self.turn_signal_dest) # move the turn signal to indicate human turn
            self.game.turn *= -1 # if we get here, we moved the turn signal! From robot turn to human turn TODO maybe need to handle missing the turn signal differently?
            self.determine_action_flag = False
            self.game.roll() # next roll (for human)
            self.get_logger().info('Human roll is: ' + str(self.game.dice))
            # TODO: publish this roll to a topic so it can be displayed on an ipad?

                #self.game.move(source, dest) # comment out to rely only on detectors for gamestate
                #self.get_logger().info("Gamestate after move:"+str(self.game.state))
                #self.get_logger().info("exectued the move")
            # # comment out if robot only plays as one color
        else:
            self.get_logger().info('Fixing board!!')
            checkgame = Game(self.gamestate)
            green_fixes, brown_fixes = self.fix_board(checkgame.state, self.game.state) # here, we use the old state as the "expected" and just return to that old state
            self.get_logger().info('fixes in determine action' + str(brown_fixes + green_fixes))
            self.execute_fixes(green_fixes, brown_fixes, checkgame)
            self.publish_checker_move(self.turn_signal_pos,self.turn_signal_dest) # move the turn signal to indicate human has to re-play
            self.get_logger().info("Please Play Correctly!!")
            self.get_logger().info("Redo Move with Dice: " + str(self.game.dice))
            self.get_logger().info("self.game turn check" + str(self.game.turn))
            self.determine_action_flag = False

        #self.get_logger().info("Gamestate:"+str(self.game.state))
    def execute_fixes(self, green_moves, brown_moves, checkgame):
        sources = []
        dests = []

        self.repeat_source = 0
        self.repeat_dest = 0
        for (source,dest) in green_moves:
            # if source in dests: go to hardcoded center number (repeat - 1) of row source
            self.repeat_source = sources.count(source) + sources.count(dest)
            self.repeat_dest = dests.count(dest) + dests.count(source)
            sources.append(source)
            dests.append(dest)
            self.get_logger().info("Moving from {} to {}".format(source, dest))
            self.get_logger().info("Source number"+str((checkgame.state[source])))
            self.get_logger().info("Dest number" + str(checkgame.state[dest]))
            self.execute_normal(source, dest, change_game=False, simulate_change_game=checkgame, turn=0)

        self.repeat_source = 0
        self.repeat_dest = 0
        sources = []
        dest = []
        for (source,dest) in brown_moves:
            reversed_state = [-triangle if i <= 23 else triangle for i, triangle in enumerate(checkgame.state)]
            checkgame.state = reversed_state
            # if source in dests: go to hardcoded center number (repeat - 1) of row source
            self.repeat_source = sources.count(source) + sources.count(dest)
            self.repeat_dest = dests.count(dest) + dests.count(source)
            sources.append(source)
            dests.append(dest)
            self.get_logger().info("Moving from {} to {}".format(source, dest))
            self.get_logger().info("Source number"+str((checkgame.state[source])))
            self.get_logger().info("Dest number" + str(checkgame.state[dest]))
            self.execute_normal(source, dest, change_game=False, simulate_change_game=checkgame, turn=1)
                

    def empty_checker_count(self):
        return [[[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                [[],[]], [[],[]], [[],[]], [[],[]], [[],[]], [[],[]],
                [[],[]], [[],[]]]

    def get_changed_bounds(self, checkgame):
        low_row = 0
        self.changed = True
        while checkgame.state[low_row] == self.game.state[low_row]:
            low_row += 1
            if low_row == 24:
                if (checkgame.state[low_row][0] != self.game.state[low_row][0] or
                   checkgame.state[low_row][1] != self.game.state[low_row][1]):
                    break
                else:
                    self.changed = False
                    return [0,0]
                    break
        high_row = 24
        if low_row != 24 and self.changed:
            if not (checkgame.state[high_row][0] != self.game.state[high_row][0] or
                   checkgame.state[high_row][1] != self.game.state[high_row][1]):
                high_row -= 1
                while (checkgame.state[high_row] == self.game.state[high_row]):
                    if high_row == 0:
                        self.changed = False
                        break
                    high_row -= 1
        self.get_logger().info('lowest changed row' + str(low_row))        
        self.get_logger().info('highest changed row' + str(high_row))
        self.get_logger().info('Changed?' + str(self.changed))
        return [low_row, high_row]
    
    # def execute_off_bar(self, dest):
    #     turn = 0 if self.game.turn == 1 else 1
    #     bar = 24

    #     source_pos = self.last_checker(bar,turn, self.repeat_source)
    #     dest_pos = self.next_free_place(dest, turn, self.repeat_dest)
        
    #     self.game.move(24,dest)

    #     self.publish_checker_move(source_pos, dest_pos)

    def execute_bear_off(self, source, change_game=True, simulate_change_game=None, turn=None):
        if turn is None:
            turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source, turn, self.repeat_source)
        dest_pos = [0.5, 0.3]

        if change_game:
            self.game.move(source, 25)
        
        if simulate_change_game is not None:
            simulate_change_game.move(source, 25)

        self.publish_checker_move(source_pos, dest_pos)
    
    def execute_hit(self, source, dest, change_game=True, simulate_change_game=None, turn=None):
        # self.game.turn == 1 during robot (green) turn
        if turn is None:
            turn = 1 if self.game.turn == 1 else 0
        bar = 24
        
        dest_centers = self.grid_centers[dest]

        # there is a problem here with how the new last checkers work 
        source_pos_1 = self.last_checker(dest, turn, repeat=0) # grab solo checker
        dest_pos = self.next_free_place(bar, turn, repeat=0)

        if change_game:
            self.game.move(dest,bar) # move oppo checker to bar in game.state
        if simulate_change_game is not None:
            simulate_change_game.move(dest, bar)

        self.publish_checker_move(source_pos_1, dest_pos) # publish move
        turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source, turn, self.repeat_source) # grab my checker
        dest_pos = source_pos_1 # and move where I just removed

        if change_game:
            self.game.move(source,dest) # move my checker to destination in game.state
        if simulate_change_game is not None:
            simulate_change_game.move(source, dest)

        self.publish_checker_move(source_pos, dest_pos)

    def execute_normal(self, source, dest, change_game=True, simulate_change_game=None, turn=None):        
        if turn is None:
            turn = 0 if self.game.turn == 1 else 1

        source_pos = self.last_checker(source,turn,self.repeat_source)
        dest_pos = self.next_free_place(dest,turn,self.repeat_dest)
        self.get_logger().info("source_pos: " + str(source_pos))

        if change_game:
            self.game.move(source,dest)
        if simulate_change_game is not None:
            simulate_change_game.move(source, dest)

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
        game.roll() # roll dice, sets game.dice with new dice
        final_moves = []
        #print(self.game.state)
        #HANDLING DOUBLE ROLLS
        if game.dice[0] == game.dice[1]:
            for _ in range(4):
                moves = gamecopy.possible_moves(game.dice[0])
                move = self.choose_move(gamecopy, moves)
                if move is not None:
                    self.get_logger().info("Move[1] is:" + str(move[1]))
                if move is not None:
                    final_moves.append(move)
                    # Moving off the bar
                    if move[0] == 24: # off bar
                        if np.sign(gamecopy.state[move[1]]) == -1: # check for hit off bar
                            # hiting while moving off bar
                            gamecopy.move(move[1],bar)
                            gamecopy.move(move[0],move[1])
                        else:
                            # just a normal moving of the bar
                            gamecopy.move(move[0],move[1])
                    # Executing a hit 
                    elif np.sign(gamecopy.state[move[1]]) == -1:
                        # normal hit
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    # Executing a normal move
                    else:
                        gamecopy.move(move[0],move[1])
            self.get_logger().info("first chosen move: " + str(move))
            self.get_logger().info("gamecopy state after single chosen move: " + str(gamecopy.state))
        #HANDLING SINGLE ROLLS
        else:
            # larger = 1
            # if game.dice[0] > game.dice[1]:
            #     larger = 0
            moves = gamecopy.possible_moves(game.dice[0])
            move1 = self.choose_move(gamecopy, moves)
            if move1 is not None:
                move = move1
                final_moves.append(move)
                self.get_logger().info('move' + str(move))
                if move[0] == 24: # off bar
                    if np.sign(gamecopy.state[move[1]]) == -1: # check for hit off bar
                        # hiting while moving off bar
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    else:
                        # just a normal moving of the bar
                        gamecopy.move(move[0],move[1])
                # Executing bear off
                elif move[1] == 25:
                    gamecopy.move(move[0],move[1])
                # Executing hit
                elif np.sign(gamecopy.state[move[1]]) == -1:
                    # normal hit
                    gamecopy.move(move[1],bar)
                    gamecopy.move(move[0],move[1])
                # Executing a normal move
                else:
                    gamecopy.move(move[0],move[1])
            
                self.get_logger().info("first chosen move: " + str(move[1]))
            self.get_logger().info("gamecopy state after single chosen move: " + str(gamecopy.state))
                
            moves = gamecopy.possible_moves(game.dice[1])
            move = self.choose_move(gamecopy, moves)
            if move is not None:
                final_moves.append(move)
                if move[0] == 24: # off bar
                    if np.sign(gamecopy.state[move[1]]) == -1: # check for hit off bar
                        # hiting while moving off bar
                        gamecopy.move(move[1],bar)
                        gamecopy.move(move[0],move[1])
                    else:
                        # just a normal moving of the bar
                        gamecopy.move(move[0],move[1])
                # Executing a hit 
                elif np.sign(gamecopy.state[move[1]]) == -1:
                    # normal hit
                    gamecopy.move(move[1],bar)
                    gamecopy.move(move[0],move[1])
                # Executing a normal move
                else:
                    gamecopy.move(move[0],move[1])
                if move1 is None: # if we couldn't find a move with die0 the first time, try again now that we made a move with die1
                    moves = gamecopy.possible_moves(game.dice[0])
                    move1 = self.choose_move(gamecopy, moves)
                    # Trying to find if there is another move possible
                    if move1 is not None:
                        move = move1
                        final_moves.append(move)
                        if move[0] == 24: # off bar
                            if np.sign(gamecopy.state[move[1]]) == -1: # check for hit off bar
                                # hiting while moving off bar
                                gamecopy.move(move[1],bar)
                                gamecopy.move(move[0],move[1])
                            else:
                                # just a normal moving of the bar
                                gamecopy.move(move[0],move[1])
                        # Executing a hit 
                        elif np.sign(gamecopy.state[move[1]]) == -1:
                            # normal hit
                            gamecopy.move(move[1],bar)
                            gamecopy.move(move[0],move[1])
                        # Executing a normal move
                        else:
                            gamecopy.move(move[0],move[1])


            #print("Moves in handle turn:", final_moves)

        return final_moves

    def next_free_place(self,row,color,repeat):
        # color is 0 or 1 for the turn, ie green 0 brown 1
        positions = self.grid_centers[row]
        occupied_num = self.gamestate[row][color] # debug: tthe gamestate is not updated to move this while fixing the board    
        self.get_logger().info("Occupied number is:" + str(occupied_num))
        self.get_logger().info("Repeat in nfp:" + str(repeat))
        self.get_logger().info("Checker locatiins:" + str(self.checker_locations[row][color]))
        if row <= 11:
            sign = -1
        elif row <= 23:
            sign = 1
        elif row == 24: # the bar and the bear off
            occupied_num += self.gamestate[row][0 if color == 1 else 1] # consider number of spots occupied by both green and purple
            
        if occupied_num == 0:
            pos =  positions[repeat]
        else:
            last = self.checker_locations[row][color][0]
            pos =  [last[0],last[1] + (1+ repeat) * sign * 0.050]
        self.get_logger().info("Next free place position:" + str(pos))
        return pos 


    def last_checker(self,row,color,repeat):
        '''
        Get the [x,y] of the last checker in the row (closest to middle)
        Repeat should be set to zero if the row is not accessed more then
        once in one players turn
        '''
        if self.checker_locations is None:
            return
        self.get_logger().info('checker locations' + str(self.checker_locations[row][color]))
        if row <= 11:
            sorted_positions = sorted(self.checker_locations[row][color], key=lambda x: x[1])
        elif row <=24:
            sorted_positions = sorted(self.checker_locations[row][color], key=lambda x: x[1], reverse=True)
        #self.get_logger().info("Repeat:"+str(repeat))
        #self.get_logger().info("sorted_pos"+str(sorted_positions))
        # When the position is the bar
            
        if not sorted_positions: # if an empty row because we moved a checker into a place where there was previosly 0
            # catching two different situations:
            # 1. in a hit
            # 2. consecutive moves: 
            return self.grid_centers[row][repeat] # be aware this may not work in both situations?
        self.get_logger().info("Repeat in last checkers:" + str(repeat))
        self.get_logger().info("Choosen position:" + str(sorted_positions[repeat]))
        return sorted_positions[repeat]
    
    def publish_checker_move(self, source, dest):
        msg = PoseArray()
        for xy in [source, dest]:
            p = pxyz(xy[0],xy[1],0.01)
            R = Reye()
            msg.poses.append(Pose_from_T(T_from_Rp(R,p)))
        self.pub_checker_move.publish(msg)

    def publish_clear(self):
        msg = Bool()
        msg.data = True
        self.pub_clear.publish(msg)

    def clear_move_counter(self):
        clear_move_counter = [0 for _ in range(25)]
        clear_move_counter[-1] = [0, 0]

POINT_LIM = 6

class Game:
    def __init__(self, gamestate):
        self.set_state(gamestate)
        #INITIALIZATION OF BAR
        self.bar = [0, 0]
        self.off = [0, 0]
        self.dice = [0, 0]
        self.turn = 1
        self.done = False

    # Converts state in GameDriver format to state in engine format.
    def set_state(self, gamestate):
        self.state = []
        # BAR USED AGAIN AS GAME STATE 24
        self.bar = gamestate[24]
        self.off = gamestate[25]
        for point in gamestate[:24]:
            # If the column has a green
            if point[0]:
                self.state.append(point[0])
            # If the columan has a brown
            else:
                self.state.append(-point[1])
        self.state.append(self.bar)
        self.state.append(self.off)
        #self.state = np.append(self.state[11::], self.state[:11:]).tolist()

    def roll(self):
        self.dice = np.random.randint(1, 7, size = 2).tolist()
    
    def move(self, point1, point2):
        # The move turn of player One
        if self.turn == 1:
            if point2 == 25: # Bear off
                self.state[point1] -= 1
                self.state[point2][0] += 1 # Incremement number of green checkers that are off
            elif point2 == 24: # hitting the opponent checker to the bar  
                self.state[point1] += 1 # opponent is negative, so we increase 1 to 0
                self.state[point2][1] += 1 # add that brown checker to the bar
            elif self.state[point2] >= 0: # not first part of hit, if destination has nothing or some number of greens 
                if point1 != 24: # not coming off bar
                    self.state[point1] -= 1 # take one away from where we take from(green)
                else: # coming off the bar
                    self.state[point1][0] -= 1 # take the one that are taking away
                self.state[point2] += 1 # add one to our destination
            elif self.state[point2] == -1: # to moving to an purple occupied destination(hit)
                if point1 != 24: # not coming off the bar
                    self.state[point1] -= 1 # remove from the place we are taking
                else: # coming off bar
                    self.state[24][0] -= 1 # remove from the bar counter for green checker
                self.state[point2] += 1 # add green to our destination
        # The move turn of player Two
        elif self.turn == -1:
            if point2 == 25: # Bear off
                self.state[point1] += 1
                self.state[point2][1] += 1 #Increment number of purple checkers that are off            
            elif point2 == 24: # hitting opponent checker to bar
                self.state[point1] -= 1 # opponent is positive, so we decrease 1 to 0
                self.state[point2][0] += 1 # add that green checker to the bar
            elif self.state[point2] <= 0: # not first part of hit, if destination has nothing or some number of greens 
                if point1 != 24: #  not coming off bar
                    self.state[point1] += 1 # take one away from where we take from(purple)
                else: # coming off bar
                    self.state[point1][1] -= 1 # take the one we are taking off bar
                self.state[point2] -= 1 # add one to our destination
            elif self.state[point2] == 1: # moving to green occupied single  (hit)
                if point1 != 24: # if not coming off bar
                    self.state[point1] += 1 # remove from place we are taking
                else: # coming off bar
                    self.state[24][1] -= 1 # remove from bar counter for green checker
                self.state[point2] = -1 # increment purple at destination

    def is_valid(self, point1, point2, die, tried = False):
        if point2 < 0 or point2 > 25:
            return False
        if point1 == 24: # coming off bar
            if point2 == 24 or point2 == 25: # souurce can't be bar or bear off
                return False
            if (self.turn == 1 and -1 <= self.state[point2] < POINT_LIM and point2 + 1 == die or
                self.turn == -1 and -POINT_LIM < self.state[point2] <= 1 and point2 == 24 - die):
                return True
            return False
        if point2 == 25: # bearing off 
            if point1 == 24 or point1 == 25: # source can't be bar or bear off
                return False
            if (self.turn == 1 and self.state[point1] > 0 and (point1 + die >= 24 if tried else point1 + die == 24) or
                self.turn == -1 and self.state[point1] < 0 and (point1 - die <= -1 if tried else point1 - die == -1)):
                return True
            return False
        
        if (point1 == point2 or
            np.sign(self.state[point1]) != self.turn or # sign of game.state row for point1 not the same as which turn it is
            (self.state[point1] > 0 and (point1 > point2 or self.turn == -1)) or # point1 currently occupied by green and moving the "wrong way" (or the other turn)
            (self.state[point1] < 0 and (point1 < point2 or self.turn == 1)) or # same but for brown
            abs(point1 - point2) != die): # difference between points not equal to the roll
            #print('first normal conditional failing')
            return False
        if ((self.state[point1] > 0 and -1 <= self.state[point2] < POINT_LIM) or # point1 occupied by green and point2 is between occupancy -1 and 6
            (self.state[point1] < 0 and -POINT_LIM < self.state[point2] <= 1)): # point1 occupied by brown and point2 between occupancy -6 and 1
            return True
        #print('second normal conditional failing')
        return False

    def possible_moves_in_range(self, die, low_row, high_row):
        moves = []
        if high_row == 24: # no
            if self.turn == 1 and self.state[24][0] > 0: # green turn, checker in bar
                #print('move off bar in possible moves')
                #print('self.state[24]',self.state[24])
                #print("In move off bar turn 1")
                for point in range(low_row, 6):
                    if self.is_valid(24, point, die): # if our die can take the checker to the point
                        #print("Inside valid")
                        moves.append((24, point)) # add as a possible move
                return moves # only returns the one off-bar move legal for this die
            elif self.turn == -1 and self.state[24][1] > 0: # same for brown
                #print("In move off bar turn -1 ")
                for point in range(18, high_row):
                    if self.is_valid(24, point, die):
                        #print("Inside valid")
                        moves.append((24, point))
                return moves
        elif self.all_checkers_in_end(): # all checkers in home board (true if true for either side)
            #print("Inside move off board")
            if self.turn == 1:
                for point in range(low_row, high_row+1):
                    if self.is_valid(point, 24, die):
                        #print("Inside valid")
                        moves.append((point, 25))
            elif self.turn == -1:
                for point in range(low_row, high_row+1):
                    if self.is_valid(point, 24, die):
                        #print("Inside valid")
                        moves.append((point, 25))
        # Normal moves, no moves off the board found
        if not moves:
            #print("Inside normal moves")
            for point1 in range(low_row, high_row+1):
                #print('point1: ',point1)
                for point2 in range(low_row, high_row+1):
                    #print('point2: ',point2)
                    if self.is_valid(point1, point2, die):
                        #print("Inside valid")
                        #print("Source point: ",point1)
                        #print("Destination point: ", point2)
                        moves.append((point1, point2))

        # Move off board (again)
        if not moves and self.all_checkers_in_end():
            #print("Inside move off board again")
            if self.turn == 1:
                for point in range(low_row, high_row+1):
                    if self.is_valid(point, 24, die, tried=True):
                        #print("Inside valid")
                        moves.append((point, 24))
            elif self.turn == -1:
                for point in range(low_row,high_row+1):
                    if self.is_valid(point, 24, die, tried=True):
                        #print("Inside valid")
                        moves.append((point, 24))
        #print(moves)
        return moves
    
    def possible_moves(self, die):
        moves = []
        #print("Moves at the beginning of possible moves")
        # Move off bar
        if self.turn == 1 and self.state[24][0] > 0:
            #print('move off bar in possible moves')
            #print('self.state[24]',self.state[24])
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
                    if self.is_valid(point, 25, die):
                        #print("Inside valid")
                        moves.append((point, 25))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, 25, die):
                        #print("Inside valid")
                        moves.append((point, 25))

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

    def legal_next_states(self, prev_states, cur_states, dice):
        pass
        dices = []
        pur_inc = []
        pur_dec = []
        gre_inc = []
        gre_dec = []
        row = 0
        done = False
        if dice[0] == dice[1]:
            for i in range(4):
                dices.append(dice[0])
        else:
            dices.append(dice[0])
            dices.append(dice[1])
        
        for row1, row2 in zip(prev_states, cur_states): #taking each row from both previos and current states
            
            if row1[0] != row2[0]: # comparing green checkers in each row
                if row1[0] > row2[0]:
                    for j in range(abs(row1[0]-row2[0])):                
                        gre_dec.append[row]
                else:
                    for j in range(abs(row2[0]-row1[0])):
                        gre_inc.append[row]
                
            if row1[1] != row2[1]: # comparing purple checkers
                if row1[1] > row2[1]:
                    for j in range(abs(row1[1]-row2[1])):
                        pur_dec.append[row]
                else:
                    for j in range(abs(row2[1]-row1[1])):
                        pur_inc.append[row]
            row += 1
            
        if (not pur_inc or not pur_dec) and (gre_dec or gre_inc):
            return False
        
        while pur_dec:
            if not dices:
                return False
            pur_cur = pur_dec.pop(0)
            for inc in pur_inc:
                for roll in dices:
                    if self.is_valid(pur_cur, inc, roll):
                        dices.remove(roll)
                        pur_inc.remove(inc)
                        done = True
                        break
                else:
                    continue
                break
            len_dices = len(dices)
            if not done and len_dices > 1:
                if dice[0] != dice[1]:
                    copy1 = copy.deepcopy(self)
                    if copy1.is_valid(pur_cur, pur_cur - dices[0], dices[0]):
                        copy1.move(pur_cur, pur_cur - dices[0])
                        dices.remove(dices[0])
                        if copy1.is_valid(pur)
                    copy2 = copy.deepcopy(self)
                    
            elif not done and len_dices == 1:
                return False
            
                
                
             
            
            
        return True
        # combine two approaches: 
        # first, take the "new" state and figure out which rows have changed from
        # the "old" state
        # create a more limited set of rows only within the range of [lowest_changed_row, highest_changed_row]
        # within this range, 
        # TODO: make sure this actually works!!        
        [die1, die2] = dice
        states = []
        if die1 == die2: # four moves, check all possible resulting states
            moves1 = self.possible_moves(die1)
            #print('possiblemoves1', moves1)
            for (point1,point2) in moves1:
                copy1 = copy.deepcopy(self)
                if point1 == 24: # off bar
                    if np.sign(copy1.state[point2]) == 1:
                        copy1.move(point2, point1)
                        copy1.move(point1,point2)
                    else:
                        copy1.move(point1,point2)
                elif np.sign(copy1.state[point2]) == 1:
                    copy1.move(point2, 24)
                    copy1.move(point1, point2)
                else:
                    copy1.move(point1,point2)
                moves2 = copy1.possible_moves(die1)
                for (point1,point2) in moves2:
                    copy2 = copy.deepcopy(copy1)
                    if point1 == 24: # off bar
                        if np.sign(copy2.state[point2]) == 1:
                            copy2.move(point2, point1)
                            copy2.move(point1,point2)
                        else:
                            copy2.move(point1,point2)
                    elif np.sign(copy2.state[point2]) == 1:
                        copy2.move(point2, 24)
                        copy2.move(point1, point2)
                    else:
                        copy2.move(point1,point2)
                    moves3 = copy2.possible_moves(die1)
                    for (point1,point2) in moves3:
                        copy3 = copy.deepcopy(copy2)
                        if point1 == 24: # off bar
                            if np.sign(copy3.state[point2]) == 1:
                                copy3.move(point2, point1)
                                copy3.move(point1,point2)
                            else:
                                copy3.move(point1,point2)
                        elif np.sign(copy3.state[point2]) == 1:
                            copy3.move(point2, 24)
                            copy3.move(point1, point2)
                        else:
                            copy3.move(point1,point2)
                        moves4 = copy3.possible_moves(die1)
                        for (point1,point2) in moves4:
                            copy4 = copy.deepcopy(copy3)
                            if point1 == 24: # off bar
                                if np.sign(copy4.state[point2]) == 1:
                                    copy4.move(point2, point1)
                                    copy4.move(point1,point2)
                                else:
                                    copy4.move(point1,point2)
                            elif np.sign(copy4.state[point2]) == 1:
                                copy4.move(point2, 24)
                                copy4.move(point1, point2)
                            else:
                                copy4.move(point1,point2)
                            if copy4.state not in states:
                                states.append(copy4.state)
        else:
            # find all states that could result from using die 1 then die 2
            moves1 = self.possible_moves(die1)
            for (point1, point2) in moves1:
                copy1 = copy.deepcopy(self)
                if point1 == 24: # off bar
                    if np.sign(copy1.state[point2]) == 1:
                        copy1.move(point2, point1)
                        copy1.move(point1,point2)
                    else:
                        copy1.move(point1,point2)
                elif np.sign(copy1.state[point2]) == 1:
                    copy1.move(point2, 24)
                    copy1.move(point1, point2)
                else:
                    copy1.move(point1,point2)
                moves2 = copy1.possible_moves(die2)
                for (point1,point2) in moves2:
                    copy2 = copy.deepcopy(copy1)
                    if point1 == 24: # off bar
                        if np.sign(copy2.state[point2]) == 1:
                            copy2.move(point2, point1)
                            copy2.move(point1,point2)
                        else:
                            copy2.move(point1,point2)
                    elif np.sign(copy2.state[point2]) == 1:
                        copy2.move(point2, 24)
                        copy2.move(point1, point2)
                    else:
                        copy2.move(point1,point2)
                    if copy2.state not in states: # only add if not already in states
                        states.append(copy2.state)
            # find all states that could result from using die 2 then die 1
            moves1 = self.possible_moves(die2)
            for (point1,point2) in moves1:
                copy1 = copy.deepcopy(self)
                if point1 == 24: # off bar
                    if np.sign(copy1.state[point2]) == 1:
                        copy1.move(point2, point1)
                        copy1.move(point1,point2)
                    else:
                        copy1.move(point1,point2)
                elif np.sign(copy1.state[point2]) == 1:
                    copy1.move(point2, 24)
                    copy1.move(point1, point2)
                else:
                    copy1.move(point1,point2)
                moves2 = copy1.possible_moves(die1)
                for (point1,point2) in moves2:
                    copy2 = copy.deepcopy(copy1)
                    if point1 == 24: # off bar
                        if np.sign(copy2.state[point2]) == 1:
                            copy2.move(point2, point1)
                            copy2.move(point1,point2)
                        else:
                            copy2.move(point1,point2)
                    elif np.sign(copy2.state[point2]) == 1:
                        copy2.move(point2, 24)
                        copy2.move(point1, point2)
                    else:
                        copy2.move(point1,point2)
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