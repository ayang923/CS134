from geometry_msgs.msg  import Point, Pose, Quaternion, PoseArray
from std_msgs.msg import UInt8MultiArray

import numpy as np

# class passed into trajectory node to handle game logic
class GameDriver():
    def __init__(self, trajectory_node, task_handler):
        self.trajectory_node = trajectory_node
        self.task_handler = task_handler
        
        # /boardpose from Detector
        self.sub_board = self.trajectory_node.create_subscription(PoseArray, '/boardpose',
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
        
        # Stored gamestate info
        self.board1pose = 0 # TODO
        self.board2pose = 0 # TODO
        self.logoddsgrid = 0 # TODO log odds representation of occupancy by green/brown

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
        self.greenpos = np.array([[]])
        # self.recvbrown populates these arrays with detected brown checker pos
        self.brownpos = np.array([[]])
        # self.recvdice populates this with detected [die1_int, die2_int]
        self.dice = np.array([])

    def recvboard(self, msg):
        '''
        create / update belief of where the boards are based on most recent
        /boardpose msg

        in: /boardpose PoseArray
        updates self.board1pose and self.board2pose, no return
        TODO
        '''
        pass

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
        TODO
        '''
        pass

    def recvbrown(self, msg):
        '''
        same as above, but for brown
        TODO
        '''
        pass

    def recvdice(self, msg):
        '''
        process most recent /dice msg

        in: /dice UInt8MultiArray

        places values in self.dice
        TODO
        '''
        pass 

    def grid_from_board(self):
        '''
        return 25 regions with 6 subdivisions each for bucketing given
        current belief of the board pose

        in: self (need self.boardpose1 and self.boardpose2)

        returns: center coordinates of 25*6 = 150 grid spaces that a checker
        could occupy.

        based only on geometry of board, basically need to measure this out
        TODO
        '''
        pass

    def update_log_odds(self):   
        '''
        update log odds ratio for each of the grid spaces
        each space has p_green and p_brown (for being occupied by a green
        or brown checker respectively), but should implement some logic
        about columns only being able to contain one color? + complain if this
        is not the case?

        in: self (need self.greenpos and self.brownpos)
        
        updates self.logoddsgrid given the values in self.greenpos and self.brownpos
        from the most recent processed frame

        use grid_from_board() to determine how to increment/decrement the log
        odds grid given the detected checker positions.
        TODO
        '''
        pass

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
        pass