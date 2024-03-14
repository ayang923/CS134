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
from collections.abc import Iterable

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

def flatten_list(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_list(x)
        else:
            yield x
                     

def reconstruct_gamestate_array(flattened_lst):
    return np.array(flattened_lst).reshape((26, 2)).tolist() if len(flattened_lst) == 52 else None

# class passed into trajectory node to handle game logic
class GameNode(Node):
    def __init__(self, name):
        super().__init__(name)

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
        self.pub_checker_move = self.create_publisher(UInt8MultiArray, '/checker_move', 10)
        self.pub_dice_roll = self.create_publisher(PoseArray, '/dice_roll', 10)
        self.dice_publisher = self.create_publisher(Image, '/dice_roll_image', 10)
        self.dice_image_timer = self.create_timer(1, self.publish_dice_roll)
        self.bridge = CvBridge()

        # Choose initial state
        # Initial gamestate area assumes setup for beginning of game
        # each element indicates [num_green, num_brown]
        # beginning to end of array progresses ccw from robot upper right.
        # last item is middle bar
        self.gamestate = None
        
        self.scored = 0

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

        # Game engine
        self.game = Game(STANDARD)
        self.display_on = True
        if self.display_on:
            self.render = Render(self.game)
            self.render.draw()
            self.render.update()
    
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

    def raise_determine_action_flag(self, msg:Bool):
        self.get_logger().info("self.gamestate in det_action cbk" + str(self.gamestate))
        checkgame = Game(self.gamestate)
        
        fixes = self.fix_board(checkgame.state, self.game.state)
        self.get_logger().info('fixes in raise_det_action_flag callback' + str(fixes))

        if self.firstmove and not self.turn_signal:
            self.publish_checker_move([1, 0, 10])
            self.determine_action = False
            return None

        if self.turn_signal and not fixes and (True if self.game.turn==1 else False) != self.turn_signal:
            if (True if self.game.turn==1 else False) != self.turn_signal:
                self.publish_checker_move([0, 1, 10])
                self.determine_action_flag = False
        elif self.turn_signal and fixes:
            self.execute_fixes(fixes)
            self.determine_action_flag = False

            if (True if self.game.turn==1 else False) != self.turn_signal:
                self.publish_checker_move([0, 1, 10])
                self.determine_action_flag = False
            
        else:
            self.determine_action_flag = msg.data

    def fix_board(self, actual_state, expected_state):
        actual_state_copy = copy.deepcopy(actual_state)
        expected_state_copy = copy.deepcopy(expected_state)
        green_incorrect = []
        brown_incorrect = []
        moves = []

        for i, (triangle_actual, triangle_expected) in enumerate(zip(actual_state_copy[:24], expected_state_copy[:24])):
            if np.sign(triangle_actual) == np.sign(triangle_expected)*-1:
                if np.sign(triangle_actual) > 0:
                    moves += ([[i, 25, 0]]* abs(triangle_actual))
                    actual_state_copy[i] = 0
                    actual_state_copy[25][0] += abs(triangle_actual)
                elif np.sign(triangle_actual) < 0:
                    moves += ([[i, 25, 1]] * abs(triangle_actual))
                    actual_state_copy[i] = 0
                    actual_state_copy[25][1] += abs(triangle_actual)

        for i, (triangle_actual, triangle_expected) in enumerate(zip(actual_state_copy, expected_state_copy)):
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
                        for green_count in range(abs(delta_green)):
                            if len(green_incorrect) != 0:
                                moves += ([[i, abs(green_incorrect.pop(-1))-1, 0]] if np.sign(delta_green) == -1 else [[abs(green_incorrect.pop(-1))-1, i, 0]])
                            else:
                                green_incorrect = [np.sign(delta_green)*(i+1)] * abs(delta_green-green_count) + green_incorrect 
                                break 
                    else:
                        green_incorrect = [np.sign(delta_green)*(i+1)] * abs(delta_green) + green_incorrect

                elif brown_actual != brown_expected:
                    delta_brown = brown_expected - brown_actual
                    if len(brown_incorrect) != 0 and np.sign(delta_brown) != np.sign(brown_incorrect[-1]):
                        for brown_count in range(abs(delta_brown)):
                            if len(brown_incorrect) != 0:
                                moves += ([[i, abs(brown_incorrect.pop(-1))-1, 1]] if np.sign(delta_brown) == -1 else [[abs(brown_incorrect.pop(-1))-1,i, 1]])
                            else:
                                brown_incorrect = [np.sign(delta_brown)*(i+1)] * abs(delta_brown-brown_count) + brown_incorrect
                                break
                    else:
                        brown_incorrect = [np.sign(delta_brown)*(i+1)] * abs(delta_brown) + brown_incorrect
        return moves
    
    def publish_dice_roll(self):
        width, height = 640, 480
        dice_img = np.zeros((height, width, 3), dtype=np.uint8)
        font_scale = 10.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.game.dice is not None:
            dice_roll = str(self.game.dice[0]) + ',' + str(self.game.dice[1])
            cv2.putText(dice_img, dice_roll, (50, 300), font, font_scale, (255, 255, 255), 6, cv2.LINE_AA)
        if self.turn_signal:
            cv2.putText(dice_img, "Robot Turn", (25, 300), font, font_scale, (0, 255, 0), 5, cv2.LINE_AA)
        else:
            cv2.putText(dice_img, "Human Turn", (25, 300), font, font_scale, (128, 0, 128), 5, cv2.LINE_AA)
        #if self.correction_notif:
        #    cv2.putText(dice_img, "Redo move, play correctly!", (100, 300), font, font_scale, (255, 255, 255), 5, cv2.LINE_AA)

        dice_roll_image = self.bridge.cv2_to_imgmsg(dice_img, encoding='bgr8')
        self.dice_publisher.publish(dice_roll_image)

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
        if self.gamestate is None:
            self.get_logger().info("board buckets "+str(self.board_buckets))
            self.get_logger().info("checker location "+str(self.checker_locations))
            self.get_logger().info('no data')
            self.pub_checker_move.publish([])
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
        self.get_logger().info("previous self.gamestate from get gamestate: " + str(self.game.get_gamestate()))
        self.get_logger().info("Legal next states:" + str(self.game.legal_next_states(self.game.get_gamestate(), self.gamestate,self.game.dice)))
        if ((self.game.legal_next_states(self.game.get_gamestate(), self.gamestate,self.game.dice)) or self.firstmove or not self.changed): # Check that self.gamestate is a legal progression from last state given roll
            # FIXME: remove "or True" if we can get legal states working
            if self.firstmove:
                self.firstmove = False
            else:
                self.game.turn *= -1 # from human turn to robot turn
            
            self.game.set_state(self.gamestate) # update game.state
            moves = self.handle_turn(self.game) # roll dice and decide moves (with new gamestate)
            self.get_logger().info("After Handle Turn Game state is" + str(self.game.state))

            # Debugs
            self.get_logger().info("Robot turn dice roll: {}".format(self.game.dice))
            # TODO: publish dice roll display to ipad through topic?
            self.get_logger().info("Robot chosen Moves: {}".format(moves))
            
            move_list = []
            for move in moves:
                source, dest = move[0], move[1]
                if source == 24:
                    if (np.sign(self.game.state[dest]) == -1):
                        self.get_logger().info("Hit off bar!!")
                        move_list += self.execute_hit(source, dest)
                    else:
                        move_list += self.execute_normal(source, dest, 0)
                elif dest == 25: #moving off game with bear off
                    move_list += self.execute_bear_off(source)
                elif (np.sign(self.game.state[source]) != np.sign(self.game.state[dest]) and 
                    self.game.state[dest] != 0):
                    self.get_logger().info("Hit!!")
                    move_list += self.execute_hit(source, dest)
                else:
                    move_list += self.execute_normal(source, dest, 0)
            self.get_logger().info('game.state after sending moves: ' + str(self.game.state))
            self.publish_checker_move(move_list)
            self.publish_checker_move([0, 1, 10]) # move the turn signal to indicate human turn
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
            fixes = self.fix_board(checkgame.state, self.game.state) # here, we use the old state as the "expected" and just return to that old state
            self.get_logger().info('fixes in determine action' + str(fixes))
            self.execute_fixes(fixes)
            self.publish_checker_move([0, 1, 10]) # move the turn signal to indicate human has to re-play
            self.dice
            self.get_logger().info("Please Play Correctly!!")
            self.get_logger().info("Redo Move with Dice: " + str(self.game.dice))
            self.get_logger().info("self.game turn check" + str(self.game.turn))
            self.determine_action_flag = False

        #self.get_logger().info("Gamestate:"+str(self.game.state))
    def execute_fixes(self, moves):
        move_list = []
        for (source, dest, color) in moves:
            # if source in dests: go to hardcoded center number (repeat - 1) of row source
            move_list += self.execute_normal(source, dest, color, change_game=False)
        
        self.publish_checker_move(move_list)

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

    def execute_bear_off(self, source, change_game=True):
        color = 0 if self.game.turn == 1 else 0
        if change_game:
            self.game.move(source, 25)
            if self.display_on:
                self.render.draw()
                self.render.update()

        return [source, 25, color]
    
    def execute_hit(self, source, dest, change_game=True):
        color = 0 if self.game.turn == 1 else 0

        if change_game:
            self.game.move(dest, 24) # move oppo checker to bar in game.state
            self.game.move(source,dest) # move my checker to destination in game.state
            if self.display_on:
                self.render.draw()
                self.render.update()            

        return [[dest, 24, 0 if color == 1 else 1], [source, dest, color]]

    def execute_normal(self, source, dest, color, change_game=True):        
        if change_game:
            self.game.move(source, dest)
            if self.display_on:
                self.render.draw()
                self.render.update()

        return [[source, dest ,color]]
    
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
    
    def publish_checker_move(self, moves):
        msg = UInt8MultiArray(data=list(flatten_list(moves)))
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
        self.bar = copy.deepcopy(gamestate[24])
        self.off = copy.deepcopy(gamestate[25])
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
    
    def get_gamestate(self):
        gamestate = []
        for row in self.state[:24]:
            if row <= -1:
                gamestate.append([0,abs(row)])
            else:
                gamestate.append([row,0])
        gamestate.append(self.state[24])
        gamestate.append(self.state[25])
        return gamestate

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
        print("Legal Next States started")
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
                        gre_dec.append(row)
                else:
                    for j in range(abs(row2[0]-row1[0])):
                        gre_inc.append(row)
                
            if row1[1] != row2[1]: # comparing purple checkers
                if row1[1] > row2[1]:
                    for j in range(abs(row1[1]-row2[1])):
                        pur_dec.append(row)
                else:
                    for j in range(abs(row2[1]-row1[1])):
                        pur_inc.append(row)
            row += 1
            
        if (not pur_inc or not pur_dec) and (gre_dec or gre_inc):
            print("error with no purple move only green")
            return False
        
        copy1 = copy.deepcopy(self)
        print("Increased purple rows:" + str(pur_inc))
        print("Decreased purple rows:" + str(pur_dec))
        print("Increased green rows:" + str(gre_inc))
        print("Decreased green rows:" + str(gre_dec))
        print("Dices is:" + str(dices))
        
        while pur_dec:
            
            if not dices:
                print("dices are empty")
                return False
            len_dices = len(dices)
            pur_cur = pur_dec.pop(0)
            print("purple decreased is:" + str(pur_dec) )
            for inc in pur_inc:
                for roll in dices:
                    if self.is_valid(pur_cur, inc, roll):
                        dices.remove(roll)
                        print(str(roll) + " was removed from dices")
                        pur_inc.remove(inc)
                        if inc < 24: # checks if the destination is not bar or bear off
                            if copy1.state[inc] == 1: # if destination is a hit
                                copy1.move(inc, 24) # move the green to the bar
                                if inc in gre_dec and 24 in gre_inc:
                                    gre_dec.remove(inc)
                                    gre_inc.remove(24)
                                else:
                                    print("for single moves, there was no green moved with hit")
                                    return False
                        copy1.move(pur_cur, inc)
                        done = True
                        print("Executed a move")
                        break
                else:
                    continue
                break
            if not done and len_dices > 1:
                if dice[0] != dice[1]: # If the dice is a normal roll
                    if copy1.is_valid(pur_cur, pur_cur - dices[0], dices[0]):
                        cur_pos = pur_cur - dices[0]
                        copy2 = copy.deepcopy(copy1)
                        if cur_pos < 24:
                            if copy2.state[cur_pos] == 1:
                                copy2.move(cur_pos, 24)
                        copy2.move(pur_cur, cur_pos)
                        nex_pos = cur_pos - dices[1]
                        if copy2.is_valid(cur_pos, nex_pos, dices[1]) and ((nex_pos) in pur_inc):
                            dices.remove(dices[1])
                            dices.remove(dices[0])
                            print("dices is :" + str(dices))
                            pur_inc.remove(nex_pos)
                            if cur_pos < 24:
                                if copy1.state[cur_pos] == 1:
                                    copy1.move(cur_pos, 24)
                                    if cur_pos in gre_dec and 24 in gre_inc:
                                        gre_dec.remove(cur_pos)
                                        gre_inc.remove(24)
                                    else:
                                        print("In single dice roll there was no green moved with hit")
                                        return False
                            copy1.move(pur_cur, cur_pos)
                            if (cur_pos - dice[0])  < 24:
                                if copy1.state[cur_pos - dice[0]] == 1:
                                    copy1.move(cur_pos - dice[0], 24)
                                    if (cur_pos - dice[0]) in gre_dec and 24 in gre_inc:
                                        gre_dec.remove(cur_pos - dice[0])
                                        gre_inc.remove(24)
                                    else:
                                        print("In single dice roll there was no green moved with hit")
                                        return False
                            copy1.move(cur_pos, cur_pos - dice[0])
                            done = True
                            print("Done after updating: " + str(done))
                            
                    if copy1.is_valid(pur_cur, pur_cur - dices[1], dices[1]) and not done:
                        cur_pos = pur_cur - dices[1]
                        copy2 = copy.deepcopy(copy1)
                        if cur_pos < 24:
                            if copy2.state[cur_pos] == 1:
                                copy2.move(cur_pos, 24)
                        copy2.move(pur_cur, cur_pos)
                        if copy2.is_valid(cur_pos, cur_pos - dices[0], dices[0]) and ((cur_pos - dices[0]) in pur_inc): # Change is valid 
                            dices.remove(dices[1])
                            dices.remove(dices[0])
                            pur_inc.remove(cur_pos - dices[0])
                            if cur_pos < 24:
                                if copy1.state[cur_pos] == 1:
                                    copy1.move(cur_pos, 24)
                                    if cur_pos in gre_dec and 24 in gre_inc:
                                        gre_dec.remove(cur_pos)
                                        gre_inc.remove(24)
                                    else:
                                        print("In single dice role part 2 there was no green moved with hit")
                                        return False
                            copy1.move(pur_cur, cur_pos)
                            if (cur_pos - dice[0]) < 24:
                                if copy1.state[cur_pos - dice[0]] == 1:
                                    copy1.move(cur_pos - dice[0], 24)
                                    if (cur_pos - dice[0]) in gre_dec and 24 in gre_inc:
                                        gre_dec.remove(cur_pos - dice[0])
                                        gre_inc.remove(24)
                                    else:
                                        print("In single dice role part 2 there was no green moved with hit")
                                        return False
                            copy1.move(cur_pos, cur_pos - dice[0])
                            done = True
                            
                else: # If the dice is a double roll
                    if copy1.is_valid(pur_cur, pur_cur - dices[0], dices[0]):
                        pos1 = pur_cur - dices[0]
                        copy2 = copy.deepcopy(copy1)
                        if pos1 < 24:
                                if copy2.state[pos1] == 1:
                                    copy2.move(pos1, 24)
                        copy2.move(pur_cur, pos1)
                        pos2 =  pos1 - dices[0]
                        if copy2.is_valid(pos1, pos2, dices[0]):
                            if pos2 < 24:
                                if copy2.state[pos2] == 1:
                                    copy2.move(pos2, 24)
                            copy2.move(pos1, pos2)
                            pos3 = pos2 - dices[0]
                            if pos2 in pur_inc: # Registering a valid change in 2 of double rolls
                                dices.remove(dices[0])
                                dices.remove(dices[0])
                                pur_inc.remove(pos2)
                                if pos1 < 24:
                                    if copy1.state[pos1] == 1:
                                        copy1.move(pos1, 24)
                                        if pos1 in gre_dec and 24 in gre_inc:
                                            gre_dec.remove(pos1)
                                            gre_inc.remove(24)
                                        else:
                                            print("In double dice move 2 repeats there was no green moved with hit")
                                            return False
                                copy1.move(pur_cur, pos1)
                                if pos2 < 24:
                                    if copy1.state[pos2] == 1:
                                        copy1.move(pos2, 24)
                                        if pos2 in gre_dec and 24 in gre_inc:
                                            gre_dec.remove(pos2)
                                            gre_inc.remove(24)
                                        else:
                                            print("In double dice move 2 repeats there was no green moved with hit")
                                            return False
                                copy1.move(pos1, pos2)
                                done = True
                                
                            elif copy2.is_valid(pos2, pos3, dices[0]) and len_dices >2:
                                if pos3 < 24:
                                    if copy2.state[pos3] == 1:
                                        copy2.move(pos3, 24)
                                copy2.move(pos2, pos3)
                                pos4 = pos3 - dices[0]
                                if pos3 in pur_inc: # Registering a valid change in 3 of double rolls
                                    dices.remove(dices[0]) # Remove dice 1
                                    dices.remove(dices[0]) # Remove dice 2
                                    dices.remove(dices[0]) # Remove dice 3
                                    pur_inc.remove(pos3)
                                    if pos1 < 24:
                                        if copy1.state[pos1] == 1:
                                            copy1.move(pos1, 24)
                                            if pos1 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos1)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 3 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pur_cur, pos1)
                                    if pos2 < 24:
                                        if copy1.state[pos2] == 1:
                                            copy1.move(pos2, 24)
                                            if pos2 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos2)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 3 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pos1, pos2)
                                    if pos3 < 24:
                                        if copy1.state[pos3] == 1:
                                            copy1.move(pos3, 24)
                                            if pos3 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos3)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 3 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pos2, pos3)
                                    done = True
                                    
                                elif copy2.isValid(pos3, pos4, dices[0]) and len_dices == 4 and (pos4 in pur_inc): # Registering a change in 4 of double rolls
                                    dices.remove(dices[0]) # Remove dice 1
                                    dices.remove(dices[0]) # Remove dice 2
                                    dices.remove(dices[0]) # Remove dice 3
                                    dices.remove(dices[0]) # Remove dice 4
                                    pur_inc.remove(pos4)
                                    if pos1 < 24:
                                        if copy1.state[pos1] == 1:
                                            copy1.move(pos1, 24)
                                            if cur_pos in pos1 and 24 in gre_inc:
                                                gre_dec.remove(pos1)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 4 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pur_cur, pos1)
                                    if pos2 < 24:
                                        if copy1.state[pos2] == 1:
                                            copy1.move(pos2, 24)
                                            if pos2 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos2)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 4 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pos1, pos2)
                                    if pos3 < 24:
                                        if copy1.state[pos3] == 1:
                                            copy1.move(pos3, 24)
                                            if pos3 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos3)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 4 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pos2, pos3)
                                    if pos4 < 24:
                                        if copy1.state[pos4] == 1:
                                            copy1.move(pos4, 24)
                                            if pos4 in gre_dec and 24 in gre_inc:
                                                gre_dec.remove(pos4)
                                                gre_inc.remove(24)
                                            else:
                                                print("In double dice move 4 repeats there was no green moved with hit")
                                                return False
                                    copy1.move(pos3, pos4)
                                    done = True    
                                    
            elif not done and len_dices == 1: # If the action from decreased purple row can not be done and the dice roll has a lenght of 1
                print("error with move not getting executed and having a dice roll")
                return False
            done = False
        if pur_inc or gre_dec or gre_inc:
            print("moved green lists are not cleared")
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