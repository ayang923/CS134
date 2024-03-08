# TODO: make sure the wrist roll angle is always correct when going to pick/place
# a checker and gives us minimal chance of missing.
# TODO: (higher! priority) add a missed grab "detector" which re-tries grabbing the checker if it
# missed the first time and the gripper closed all the way (based on joint angle)
# TODO: (lower priority) finish fixing JointSplineTask N-R iteration
    # Gunter recommended adding something to the algorithm which does not allow
    # the solver to get outside our *known* bounds for the joint angles within the
    # workspace

from sixdof.utils.TrajectoryUtils import *
from sixdof.utils.KinematicChain import *
from sixdof.utils.TransformHelpers import *

from enum import Enum
import numpy as np
from scipy.linalg import diagsvd

JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wristpitch', 'wristroll', 'grip']

J_EULER = np.array([[0, 1, -1, 1, 0],[0, 0, 0, 0, 1]]).reshape(2,5) # xdot4 = qdot4 / x4 = q4

RIGHT_SIDE_JOINT_ANGLES = np.array([])
LEFT_SIDE_JOINT_ANGLES = np.array([])

GRIP_OPEN = -0.2
GRIP_DIE = -0.6
GRIP_CHECKER = -0.6
GRIP_CUP = -0.45

def alpha(q): # letting alpha = 0 be "level with table"
    return q[1] - q[2] + q[3] + np.pi

def beta(q): # letting beta = 0 be "in line with wrist bracket"
    return q[4]

class Tasks(Enum):
    INIT = 1
    JOINT_SPLINE = 2
    TASK_SPLINE = 3
    GRIP = 4

class GamePiece(Enum):
    CHECKER = 1
    DIE = 2
    CUP = 3

class TaskObject():
    def __init__(self, start_time, task_manager):
        # shared data stored in task_manager
        self.task_manager = task_manager

        # initial data
        self.start_time = start_time
        self.done = False
        self.q0 = self.task_manager.q
        self.p0 = self.task_manager.p
        self.R0 = self.task_manager.R

    def evaluate(self, t, dt):
        raise NotImplementedError
    

class InitTask(TaskObject):
    def __init__(self, start_time, task_manager):
        super().__init__(start_time, task_manager)

        self.SHOULDER_UP = np.array([self.q0[0, 0], 0.75, self.q0[2, 0], self.q0[3,0], self.q0[4,0], self.q0[5,0]]).reshape(-1,1)
        self.ELBOW_UP = np.array([0.0, 0.75, np.pi/2, -3*np.pi/4, 0.0, self.q0[5,0]]).reshape(-1,1)


        # check what needs to be done
        self.in_shoulder_up = np.linalg.norm(self.SHOULDER_UP - self.q0) < 0.1
        self.in_elbow_up = np.linalg.norm(self.ELBOW_UP - self.q0) < 0.1

        self.done = self.in_elbow_up

    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if self.done:
            return self.task_manager.q, np.zeros((6, 1))
        elif (self.in_shoulder_up):
            if (t < 5.0):
                return goto5(t, 5.0, self.SHOULDER_UP, self.ELBOW_UP)
            else:
                self.done = True
                return self.task_manager.q, np.zeros((6, 1))
        else:
            if (t < 5.0):
                return goto5(t, 5.0, self.q0, self.SHOULDER_UP)
            elif (t < 10.0):
                return goto5(t-5, 5.0, self.SHOULDER_UP, self.ELBOW_UP)
            else:
                self.done = True
                return self.task_manager.q, np.zeros((6, 1))

class TaskSplineTask(TaskObject):
    def __init__(self, start_time, task_manager, x_final=np.zeros((5, 1)), T=3.0, lam=20):
        super().__init__(start_time, task_manager)

        self.q = self.q0[:5] # ignore gripper when computing taskspline
        self.p0 = self.p0[:5] # ignore gripper when computing taskspline
        self.pd = self.p0 

        # Pick the convergence bandwidth.
        self.lam = lam

        self.x_final = np.array(x_final).reshape(5,1)
        self.T = T
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            (pd, vd) = goto5(t, self.T, self.p0, self.x_final)

            qlast = self.q
            pdlast = self.pd

            (p, _, Jv, _) = self.task_manager.chain.fkin(qlast)
            e = ep(pdlast, np.vstack((p,alpha(qlast),beta(qlast))))
            J = np.vstack((Jv, J_EULER))            

            gamma = 0.1
            U, S, V = np.linalg.svd(J)

            msk = np.abs(S) >= gamma
            S_inv = np.zeros(len(S))
            S_inv[msk] = 1/S[msk]
            S_inv[~msk] = S[~msk]/gamma**2

            J_inv = V.T @ diagsvd(S_inv, *J.T.shape) @ U.T

            qdot = J_inv@(vd + self.lam * e)

            q = qlast + qdot*dt

            self.q = np.array(q)
            self.pd = pd

        else:
            q, qdot = np.array(self.q), np.zeros((5, 1))
            self.done = True

        return np.vstack((q,self.q0[5])), np.vstack((qdot,np.zeros((1,1))))
    
class JointSplineTask(TaskObject):
    def __init__(self, start_time, task_manager, x_final, T=3.0):
        super().__init__(start_time, task_manager)
        self.T = T

        # Newton Raphson iteration to determine joint states
        qf = np.array([0, -0.2, 1.88, -2.25, 0]).reshape(-1, 1) # initial guess basically right in the middle of the workspace
        while True:
            (p, _, Jv, _) = self.task_manager.chain.fkin(qf)
            e = ep(x_final, np.vstack((p, alpha(qf),beta(qf))))
            if np.linalg.norm(e) <= 1e-12:
                break
            J = np.vstack((Jv, J_EULER))
            qf = qf + np.linalg.solve(J, e)

        self.qdest = qf
        self.task_manager.node.get_logger().info('previous (current) joint angles' + str(self.q0[:5]))
        self.task_manager.node.get_logger().info('computed destination joint angles: ' + str(qf))
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            q, qdot = goto5(t, self.T, self.q0[:5], self.qdest)
            return np.vstack((q,self.q0[5])), np.vstack((qdot,np.zeros((1,1))))
        else:
            self.done = True
            return self.task_manager.q, np.zeros((6, 1))
    
class GripperTask(TaskObject):
    def __init__(self, start_time, task_manager, piece:GamePiece=GamePiece.CHECKER, grip=True):
        # GamePiece enum object as above
        # grip=True means grip, grip=False means release
        super().__init__(start_time, task_manager)

        self.T = 0.75

        self.qgrip0 = self.q0[5]
        self.qgrip = self.qgrip0

        self.piece = piece
        self.grip = grip

    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            if not self.grip:
                (self.qgrip, qdotgrip) = goto(t, self.T, self.qgrip0, GRIP_OPEN)
            elif self.piece == GamePiece.CHECKER:
                (self.qgrip, qdotgrip) = goto(t, self.T, self.qgrip0, GRIP_CHECKER)
            elif self.piece == GamePiece.DIE:
                (self.qgrip, qdotgrip) = goto(t, self.T, self.qgrip0, GRIP_DIE)
            elif self.piece == GamePiece.CUP:
                (self.qgrip, qdotgrip) = goto(t, self.T, self.qgrip0, GRIP_CUP)
        else:
            self.qgrip, qdotgrip = self.qgrip, np.zeros(1)
            self.done = True
        
        return np.vstack((self.q0[:5],self.qgrip)), np.vstack((np.zeros((5,1)),qdotgrip))

class TaskHandler():
    def __init__(self, node, q0):
        self.node = node

        self.tasks = []

        self.curr_task_type = None
        self.curr_task_object = None

        self.q = q0

        self.chain = KinematicChain(node, 'world', 'tip', JOINT_NAMES[:5])
        self.p, self.R, _, _ = self.chain.fkin(self.q[:5])
        self.p = np.vstack((self.p, alpha(self.q), beta(self.q), self.q[5]))

    def add_state(self, task_type, **kwargs):
        self.tasks.append((task_type, kwargs))
    
    def evaluate_task(self, t, dt):
        if self.curr_task_object is None and len(self.tasks) == 0:
            return(self.q.flatten().tolist(), np.zeros((6, 1)).flatten().tolist())
        elif (self.curr_task_object is None or self.curr_task_object.done) and len(self.tasks) != 0:
            new_task_type, new_task_data = self.tasks.pop(0)
            self.set_state(new_task_type, t, **new_task_data)
        #elif (self.curr_task_type is not Tasks.INIT and len(self.tasks) == 0 and self.curr_task_object.done):
        #    self.add_state(Tasks.INIT)
        
        # updates q and p
        self.q, qdot = self.curr_task_object.evaluate(t, dt)
        self.p, _, _, _ = self.chain.fkin(self.q[:5])
        self.p = np.vstack((self.p, alpha(self.q), beta(self.q), self.q[5]))

        return (self.q.flatten().tolist(), qdot.flatten().tolist())


    def set_state(self, task_type, t, **kwargs):
        self.curr_task_type = task_type
        if task_type == Tasks.INIT:
            self.curr_task_object = InitTask(t, self)
        elif task_type == Tasks.JOINT_SPLINE:
            self.curr_task_object = JointSplineTask(t, self, **kwargs)
        elif task_type == Tasks.TASK_SPLINE:
            self.curr_task_object = TaskSplineTask(t, self, **kwargs)
        elif task_type == Tasks.GRIP:
            self.curr_task_object = GripperTask(t, self, **kwargs)
    
    def get_evaluator(self):
        return self.state_object.evaluate
    
    # Macro Add Behavior Functions

    def clear(self):
        self.tasks = [InitTask]

    def move_checker(self, source_pos, dest_pos):
        '''
        source_pos and dest_pos np.array [x, y]
        '''
        #anglestring = 'angle' + str((np.pi/2 - np.arctan2(source_pos[1], source_pos[0])) % np.pi/2)
        #self.node.get_logger().info(anglestring)
        robotx = 0.7745
        roboty = 0.0394
        
        # (p, _, Jv, _) = self.task_manager.chain.fkin(qlast)
        # e = ep(pdlast, np.vstack((p,alpha(qlast),beta(qlast))))
        # J = np.vstack((Jv, J_EULER)) 
        # FIXME Known Issue: ensuring the wrist is always parallel to
        # the long axis of the table for picking/placing is not working!
        # The last item appended to source pos and dest pos below
        source_pos_xyz = np.vstack((source_pos + np.array([0.01, 0]).reshape(-1, 1), np.array([[0.00005]]).reshape(-1,1)))
        source_pos_angles = np.array([-np.pi / 2, 0.2+float(np.arctan2(-(source_pos[0]-robotx), source_pos[1]-roboty))]).reshape(-1, 1)
        source_pos = np.vstack((source_pos_xyz, source_pos_angles))
        above_source_pos = source_pos+np.array([0, 0, 0.1, 0, 0]).reshape(-1, 1)

        dest_pos_xyz = np.vstack((dest_pos, np.array([[0.05]]).reshape(-1,1)))
        dest_pos_angles = np.array([-np.pi / 2, 0.2+float(np.arctan2(-(dest_pos[0]-robotx), dest_pos[1]-roboty))]).reshape(-1, 1)
        dest_pos = np.vstack((dest_pos_xyz, dest_pos_angles))
        
        #self.node.get_logger().info("given source: " + str(source_pos))
        #self.node.get_logger().info("given dest: " + str(dest_pos))

        # Queue Trajectories
        # Joint spline to 10cm above pick checker
        #self.add_state(Tasks.JOINT_SPLINE, x_final=above_source_pos, T=5)
        self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, T=5)
        # Task spline to pick checker
        self.add_state(Tasks.TASK_SPLINE, x_final=source_pos, T=2)
        # Grip checker
        self.add_state(Tasks.GRIP, grip=True)
        # Task spline to pull up from checker
        self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, T=2)
        # uncomment for debug
        #self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, T=20)
        # Joint spline to destination
        #self.add_state(Tasks.JOINT_SPLINE, x_final=dest_pos, T=5)
        self.add_state(Tasks.TASK_SPLINE, x_final=dest_pos, T=5)
        # uncomment for debug
        #self.add_state(Tasks.TASK_SPLINE, x_final=dest_pos, T=20)
        # Release checker
        self.add_state(Tasks.GRIP, grip=False)
        # Back to wait position
        #self.add_state(Tasks.INIT)


        #self.node.get_logger().info(f"source pos {source_pos}")
        #self.node.get_logger().info(f"dest pos {dest_pos}")

        # if source_pos[0] - robotx < -0.1:
        #     source_left = True
        # else:
        #     source_left = False
        # if dest_pos[0] - robotx < -0.1:
        #     dest_left = True
        # else:
        #     dest_left = False   

        # if source_left:
        #     self.add_state(Tasks.JOINT_SPLINE, side=0, T = 5)
        # else:
        #     self.add_state(Tasks.JOINT_SPLINE, side=1, T = 5)
        # self.add_state(Tasks.TASK_SPLINE,x_final = np.array(source_pos), T = 5)
        # self.add_state(Tasks.GRIP)
        # if source_left:
        #     self.add_state(Tasks.JOINT_SPLINE, side=0, T = 5)
        # else:
        #     self.add_state(Tasks.JOINT_SPLINE, side=1, T = 5)
        # if dest_left and not source_left:
        #     self.add_state(Tasks.JOINT_SPLINE, side=0, T = 5)
        # elif not dest_left and source_left:
        #     self.add_state(Tasks.JOINT_SPLINE, side=1, T = 5)
        # # self.add_state(Tasks.TASK_SPLINE, x_final = source_pos_star, T = 5)
        # # self.add_state(Tasks.TASK_SPLINE, x_final = source_pos, T = 5)
        # # self.add_state(Tasks.GRIP, grip = True)
        #self.add_state(Tasks.TASK_SPLINE, x_final = dest_pos, T = 5)
        # self.add_state(Tasks.GRIP, grip = False)  
        # self.add_state(Tasks.INIT)

    def pick_and_drop(self, pos):
        p1 = np.array([pos[0], pos[1], pos[2] + 0.05, -np.pi / 2, 0]).reshape(-1, 1)
        p2 = np.array([pos[0], pos[1], pos[2] + 0.005, -np.pi / 2, 0]).reshape(-1, 1)
        self.add_state(Tasks.INIT)
        self.add_state(Tasks.TASK_SPLINE, x_final = p1, T = 4)
        self.add_state(Tasks.TASK_SPLINE, x_final = p2, T = 4)
        self.add_state(Tasks.GRIP)
        self.add_state(Tasks.INIT)
        self.add_state(Tasks.GRIP, grip=False)