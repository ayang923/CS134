from sixdof.TrajectoryUtils import *
from sixdof.KinematicChain import *
from sixdof.TransformHelpers import *

from enum import Enum
import numpy as np
from scipy.linalg import diagsvd

JOINT_NAMES = ['base', 'shoulder', 'elbow']

class Tasks(Enum):
    INIT = 1
    JOINT_SPLINE = 2
    TASK_SPLINE = 3

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

        self.SHOULDER_UP = np.array([self.q0[0, 0], 0.0, self.q0[2, 0]]).reshape(-1,1)
        self.ELBOW_UP = np.array([0.0, 0.0, np.pi/2]).reshape(-1,1)

        # check what needs to be done
        self.in_shoulder_up = np.linalg.norm(self.SHOULDER_UP - self.q0) < 0.1
        self.in_elbow_up = np.linalg.norm(self.ELBOW_UP- self.q0) < 0.1

        self.done = self.in_elbow_up

    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if self.done:
            return self.task_manager.q, np.zeros((3, 1))
        elif (self.in_shoulder_up):
            if (t < 3.0):
                return goto5(t, 3.0, self.SHOULDER_UP, self.ELBOW_UP)
            else:
                self.done = True
                return self.task_manager.q, np.zeros((3, 1))
        else:
            if (t < 3.0):
                return goto5(t, 3.0, self.q0, self.SHOULDER_UP)
            elif (t < 6.0):
                return goto5(t-3, 3.0, self.SHOULDER_UP, self.ELBOW_UP)
            else:
                self.done = True
                return self.task_manager.q, np.zeros((3, 1))

class TaskSplineTask(TaskObject):
    def __init__(self, start_time, task_manager, x_final=np.zeros((3, 1)), T=3.0, lam=20):
        super().__init__(start_time, task_manager)

        self.q = self.q0
        self.pd = self.p0

        # Pick the convergence bandwidth.
        self.lam = lam

        self.x_final = x_final
        self.T = 3.0
    
    def evaluate(self, t, dt):
        t = t - self.start_time - dt
        if t < self.T:
            (pd, vd) = goto5(t, self.T, self.p0, self.x_final)

            qlast = self.q
            pdlast = self.pd

            (p, _, Jv, _) = self.task_manager.chain.fkin(qlast)
            e = ep(pdlast, p)
            J = Jv

            gamma = 0.1
            U, S, V = np.linalg.svd(Jv)

            msk = np.abs(S) >= gamma
            S_inv = np.zeros(len(S))
            S_inv[msk] = 1/S[msk]
            S_inv[~msk] = S[~msk]/gamma**2

            J_inv = V.T @ diagsvd(S_inv, *J.T.shape) @ U.T

            qdot = J_inv@(vd + self.lam * e)
            q = qlast + qdot*dt

            self.q = q
            self.pd = pd

        else:
            q, qdot = self.q, np.zeros((3, 1))
            self.done = True

        return q, qdot


class TaskHandler():
    def __init__(self, node, q0):
        self.node = node

        self.tasks = []

        self.curr_task_type = None
        self.curr_task_object = None

        self.q = q0

        self.chain = KinematicChain(node, 'world', 'tip', JOINT_NAMES)
        self.p, self.R, _, _ = self.chain.fkin(self.q)

    def add_state(self, task_type, **kwargs):
        self.tasks.append((task_type, kwargs))
    
    def evaluate_task(self, t, dt):
        if self.curr_task_object is None and len(self.tasks) == 0:
            return(self.q.flatten().tolist(), np.zeros(3, 1).flatten().tolist())
        elif (self.curr_task_object is None or self.curr_task_object.done) and len(self.tasks) != 0:
            new_task_type, new_task_data = self.tasks.pop(0)
            self.set_state(new_task_type, t, **new_task_data)
            self.node.get_logger().info(str(self.curr_task_object))
        
        # updates q and p
        self.q, qdot = self.curr_task_object.evaluate(t, dt)
        self.p, _, _, _ = self.chain.fkin(self.q)

        return (self.q.flatten().tolist(), qdot.flatten().tolist())


    def set_state(self, task_type, t, **kwargs):
        self.curr_task_type = task_type
        if task_type == Tasks.INIT:
            self.curr_task_object = InitTask(t, self)
        elif task_type == Tasks.TASK_SPLINE:
            self.curr_task_object = TaskSplineTask(t, self, **kwargs)
    
    def get_evaluator(self):
        return self.state_object.evaluate