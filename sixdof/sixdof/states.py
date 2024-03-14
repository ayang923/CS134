from sixdof.utils.TrajectoryUtils import *
from sixdof.utils.KinematicChain import *
from sixdof.utils.TransformHelpers import *

from enum import Enum
import numpy as np
from scipy.linalg import diagsvd

import copy

JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wristpitch', 'wristroll', 'grip']

BOARD_HEIGHT = 0.005
GRIPPER_OFFSET = 0.13

J_EULER = np.array([[0, 1, -1, 1, 0],[0, 0, 0, 0, 1]]).reshape(2,5) # xdot4 = qdot4 / x4 = q4

RIGHT_SIDE_JOINT_ANGLES = np.array([])
LEFT_SIDE_JOINT_ANGLES = np.array([])

GRIP_OPEN = -0.37
GRIP_DIE = -0.6
GRIP_CHECKER = -0.58
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
    WAIT = 5
    WIGGLE = 6

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
        self.qdot0 = self.task_manager.qdot
        self.p0 = self.task_manager.p
        self.v0 = self.task_manager.v
        self.R0 = self.task_manager.R

    def evaluate(self, t, dt):
        raise NotImplementedError
    

class InitTask(TaskObject):
    def __init__(self, start_time, task_manager):
        super().__init__(start_time, task_manager)

        self.SHOULDER_UP = np.array([self.q0[0, 0], 0.75, self.q0[2, 0], self.q0[3,0], self.q0[4,0], self.q0[5,0]]).reshape(-1,1)
        self.ELBOW_UP = np.array([0.0, 0.75, np.pi/2, -3*np.pi/4, 0.0, GRIP_OPEN]).reshape(-1,1)


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
            elif (t < 8.0):
                return goto5(t-5, 3.0, self.SHOULDER_UP, self.ELBOW_UP)
            else:
                self.done = True
                return self.task_manager.q, np.zeros((6, 1))

class TaskSplineTask(TaskObject):
    def __init__(self, start_time, task_manager, x_final=np.zeros((5, 1)), dir=0, T=3.0, lam=20):
        super().__init__(start_time, task_manager)

        self.q = self.q0[:5] # ignore gripper when computing taskspline
        self.qdot = self.qdot0[:5]
        self.p0 = self.p0[:5] # ignore gripper when computing taskspline
        self.pd = self.p0

        # Pick the convergence bandwidth.
        self.lam = lam

        self.x_final = np.array(x_final).reshape(5,1)
        self.T = T
        self.v0 = np.vstack((self.v0, np.array([0.0, 0.0]).reshape(-1,1)))
        self.vf = np.array([0.0, 0.0, dir*0.05, 0.0, 0.0]).reshape(-1,1)
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            (pd, vd) = spline5(t, self.T, self.p0, self.x_final, self.v0, self.vf, np.zeros((5,1)).reshape(-1,1), np.zeros((5,1)).reshape(-1,1))
            #(pd, vd) = goto5(t, self.T, self.p0, self.x_final)

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

            q = qlast + self.qdot*dt

            self.q = np.array(q)
            self.qdot = np.array(qdot)
            self.pd = pd
            self.v = Jv @ self.qdot
            self.Jv = Jv

        else:
            q, qdot = np.array(self.q), np.array(self.qdot)
            self.done = True

        return np.vstack((q,self.q0[5])), np.vstack((qdot,np.zeros((1,1))))

class WiggleTask(TaskObject):
    def __init__(self, start_time, task_manager, T=3.0, lam=40):
        super().__init__(start_time, task_manager)

        self.q = self.q0[:5] # ignore gripper when computing taskspline
        self.p0 = self.p0[:5] # ignore gripper when computing taskspline
        self.pd = self.p0 

        # Pick the convergence bandwidth.
        self.lam = lam

        self.x_L = self.p0 + np.array([-0.008, 0, -0.005, 0, 0]).reshape(-1,1)
        self.x_Lh = self.p0 + np.array([-0.008, 0, 0.002, 0, 0]).reshape(-1,1)
        self.x_R = self.p0 + np.array([0.008, 0, -0.005, 0, 0]).reshape(-1,1)
        self.x_Rh = self.p0 + np.array([0.008, 0, 0.002, 0, 0]).reshape(-1,1)
        self.x_F = self.p0 + np.array([0, 0, 0.0, 0, 0]).reshape(-1,1)

        self.T = T
        
        self.in_end = False
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T/6:
            (pd, vd) = goto5(t, self.T/6, self.p0, self.x_L)
        elif t < 2*self.T/6:
            (pd, vd) = goto5(t - self.T/6, self.T/6, self.x_L, self.x_Lh)
        elif t < 4*self.T/6:
            (pd, vd) = goto5(t - self.T/3, self.T/3, self.x_Lh, self.x_R)
        elif t < 5*self.T/6:
            (pd, vd) = goto5(t - 4*self.T/6, self.T/6, self.x_R, self.x_Rh)
        elif t < self.T:
            (pd, vd) = goto5(t - 5*self.T/6, self.T/6, self.x_Rh, self.x_F)
        else:
            self.in_end = True
        
        if not self.in_end:
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

class WaitTask(TaskObject):
    def __init__(self, start_time, task_manager, T=1.0):
        super().__init__(start_time, task_manager)
        self.T = T
        self.q = self.q0
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            self.q = self.q + dt * self.qdot0
            self.qdot = self.qdot0
            return self.q, self.qdot0
        else:
            self.done = True
            self.q = self.q + dt * self.qdot0
            self.qdot = self.qdot0
            return self.q, self.qdot0

class WaitTask(TaskObject):
    def __init__(self, start_time, task_manager, T=0.5, lam=20):
        super().__init__(start_time, task_manager)      
        self.q = self.q0[:5] # ignore gripper when computing taskspline
        self.qdot = self.qdot0[:5]
        self.p0 = self.p0[:5]
        self.pd = self.p0
        # Pick the convergence bandwidth.
        self.lam = lam
        self.T = T
        self.v0 = np.vstack((self.v0, np.array([0.0, 0.0]).reshape(-1,1)))
        self.vf = self.v0
        self.x_final = self.p0 * self.v0 * self.T
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            (pd, vd) = spline5(t, self.T, self.p0, self.x_final, self.v0, self.vf, np.zeros((5,1)).reshape(-1,1), np.zeros((5,1)).reshape(-1,1))
            #(pd, vd) = goto5(t, self.T, self.p0, self.x_final)

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

            q = qlast + self.qdot*dt

            self.q = np.array(q)
            self.qdot = np.array(qdot)
            self.pd = pd
            self.v = Jv @ self.qdot

        else:
            q, qdot = np.array(self.q), np.array(self.qdot)
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
        #self.task_manager.node.get_logger().info('previous (current) joint angles' + str(self.q0[:5]))
        #self.task_manager.node.get_logger().info('computed destination joint angles: ' + str(qf))
    
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
        self.qdot = np.zeros((6,1)).reshape(-1,1)

        self.chain = KinematicChain(node, 'world', 'tip', JOINT_NAMES[:5])
        self.p, self.R, self.Jv, _ = self.chain.fkin(self.q[:5])
        self.p = np.vstack((self.p, alpha(self.q), beta(self.q), self.q[5]))
        self.v = np.zeros((3,1)).reshape(-1,1)

        self.clearing = False

    def add_state(self, task_type, **kwargs):
        self.tasks.append((task_type, kwargs))
    
    def evaluate_task(self, t, dt):
        if self.curr_task_object is None and len(self.tasks) == 0:
            return(self.q.flatten().tolist(), np.zeros((6, 1)).flatten().tolist())
        elif (self.curr_task_object is None or self.curr_task_object.done) and len(self.tasks) != 0:
            new_task_type, new_task_data = self.tasks.pop(0)
            self.set_state(new_task_type, t, **new_task_data)
        elif self.curr_task_object.done and len(self.tasks) == 0:
            self.clearing = False
        #elif (self.curr_task_type is not Tasks.INIT and len(self.tasks) == 0 and self.curr_task_object.done):
        #    self.add_state(Tasks.INIT)
        
        # updates q and p
        self.q, self.qdot = self.curr_task_object.evaluate(t, dt)
        self.p, _, self.Jv, _ = self.chain.fkin(self.q[:5])
        self.p = np.vstack((self.p, alpha(self.q), beta(self.q), self.q[5]))
        self.v = self.Jv @ self.qdot[:5]

        return (self.q.flatten().tolist(), self.qdot.flatten().tolist())


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
        elif task_type == Tasks.WAIT:
            self.curr_task_object = WaitTask(t, self, **kwargs)
        elif task_type == Tasks.WIGGLE:
            self.curr_task_object = WiggleTask(t, self, **kwargs)
    
    def get_evaluator(self):
        return self.state_object.evaluate
    
    # Macro Add Behavior Functions

    def clear(self):
        if not self.clearing:
            self.tasks = []
            self.add_state(Tasks.INIT)
            self.add_state(Tasks.GRIP, grip=False)
            self.curr_task_object.done = True

            self.clearing = True

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
        source_pos_xyz = np.vstack((source_pos + np.array([0.01, 0.01]).reshape(-1, 1), np.array([GRIPPER_OFFSET + BOARD_HEIGHT]).reshape(-1,1)))
        source_pos_angles = np.array([-np.pi / 2, 0.1+float(np.arctan2(-(source_pos[0]-robotx), source_pos[1]-roboty))]).reshape(-1, 1)
        source_pos = np.vstack((source_pos_xyz, source_pos_angles))
        above_source_pos = source_pos+np.array([0, 0, 0.075, 0, 0]).reshape(-1, 1)
        if source_pos_xyz[1] < 0.4: # bottom part of board
            y_offset_dir = 1
        else:
            y_offset_dir = -1

        dest_pos_xyz = np.vstack((dest_pos, np.array([GRIPPER_OFFSET + 0.035 + BOARD_HEIGHT]).reshape(-1,1)))
        dest_pos_angles = np.array([-np.pi / 2, 0.1+float(np.arctan2(-(dest_pos[0]-robotx), dest_pos[1]-roboty))]).reshape(-1, 1)
        dest_pos = np.vstack((dest_pos_xyz, dest_pos_angles))
        
        #self.node.get_logger().info("given source: " + str(source_pos))
        #self.node.get_logger().info("given dest: " + str(dest_pos))

        # Queue Trajectories
        # Joint spline to 10cm above pick checker
        #self.add_state(Tasks.JOINT_SPLINE, x_final=above_source_pos, T=5)
        self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, dir=-1, T=4) # FIXME this is the buggy one
        #self.add_state(Tasks.WAIT, T=20)
        # uncomment for debug
        #self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, T=20)
        # Task spline to pick checker
        self.add_state(Tasks.TASK_SPLINE, x_final=source_pos, T=1.5)
        # wiggle
        #self.add_state(Tasks.WAIT, T=15)
        #self.add_state(Tasks.WIGGLE, T=2.0)
        self.add_state(Tasks.WIGGLE, T=1.0)
        # Grip checker
        self.add_state(Tasks.GRIP, grip=True)
        # Task spline to pull up from checker
        self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos+np.array([0, y_offset_dir*0.05, 0.0, 0, 0]).reshape(-1, 1), dir=1, T=2)
        self.add_state(Tasks.WAIT, T=0.5)
        # uncomment for debug
        #self.add_state(Tasks.TASK_SPLINE, x_final=above_source_pos, T=20)
        # Joint spline to destination
        #self.add_state(Tasks.JOINT_SPLINE, x_final=dest_pos, T=5)
        self.add_state(Tasks.TASK_SPLINE, x_final=dest_pos, T=4)
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
