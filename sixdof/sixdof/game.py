from geometry_msgs.msg  import Point, Pose, Quaternion

# class passed into trajectory node to handle game logic
class GameDriver():
    def __init__(self, trajectory_node, task_handler):
        self.trajectory_node = trajectory_node
        self.task_handler = task_handler

        self.rcvpoints = self.trajectory_node.create_subscription(Point, '/point',
                                                  self.recvpoint, 3)
        
    def recvpoint(self, pointmsg):
        if self.task_handler.tasks:
            return

        # Extract the data.
        pos = [pointmsg.x, pointmsg.y, pointmsg.z]

        print("FOUND CHECKER AT ({:.2f}, {:.2f}, {:.2f})".format(pos[0], pos[1], pos[2]),
              flush=True)
        
        self.task_handler.pick_and_drop(pos)