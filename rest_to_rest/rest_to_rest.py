import os
import math
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from px4_autonomous_interfaces.action import ExecuteTrajectory
from .qlearning import QLearning
from .util import *
from .environment import GridEnvironment
from .agent import DynamicalSystem
from transforms3d.euler import euler2quat

class RestToRest(Node):
    def __init__(self):
        super().__init__('rest_to_rest')

        map = np.ones((10, 15))
        map[7, 0] = 0
        map[3, 2] = 0
        map[9, 4] = 0
        map[4, 6] = 0
        map[8, 8] = 0
        map[0:3, 4:9] = 0
        map[7:10, 12:15] = 0
        map[2:4, 12] = 0
        map[2, 9:12] = 0

        initial_state = (0, 0, 0, 0)
        final_state = (11, 0, 0, 0)
        self.altitude = 5.0

        vstates = generate_states_vector(map, [-2, -1, 0, 1, 2])
        vactions = generate_actions_vector([-1, 0, 1])

        env = GridEnvironment(map)
        agent = DynamicalSystem(vstates, vactions, -2, 2)
        
        self.qlearning = QLearning((0, 0, 0, 0), (11, 0, 0, 0), agent, env, 1, 0.95, False)
        self.qlearning.load(os.path.join(get_package_share_directory('rest_to_rest'), 'models/Q.npy'))

        self.execute_path_action = self.declare_parameter('execute_trajectory_action', 'execute_trajectory')
        self.execute_path_action_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

    def get_path(self):
        policy = self.qlearning.get_policy()

        quaternion = euler2quat(0, 0, math.pi / 2)
        orientation = Quaternion(w=quaternion[0], x=quaternion[1], y=quaternion[2], z=quaternion[3])
        
        poses = []
        for s in policy[0]:
            pose = PoseStamped()
            pose.pose.position.x = float(s[0])
            pose.pose.position.y = float(s[1])
            pose.pose.position.z = self.altitude
            pose.pose.orientation = orientation
            poses.append(pose)
        
        return Path(poses=poses)

    def send_goal(self, path):
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.path = path

        self.execute_path_action_client.wait_for_server()

        return self.execute_path_action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)

    trajectory_generator = RestToRest()
    trajectory = trajectory_generator.get_path()
    future = trajectory_generator.send_goal(trajectory)

    """
    pose1 = PoseStamped()
    pose1.pose.position.x = 1.0
    pose1.pose.position.y = 1.0
    pose1.pose.position.z = 1.0
    pose2 = PoseStamped()
    pose2.pose.position.x = 2.0
    pose2.pose.position.y = 2.0
    pose2.pose.position.z = 2.0
    pose3 = PoseStamped()
    pose3.pose.position.x = 3.0
    pose3.pose.position.y = 3.0
    pose3.pose.position.z = 3.0
    poses = [pose1, pose2, pose3]
    path = Path(poses=poses)
    """
    
    rclpy.spin_until_future_complete(trajectory_generator, future)


if __name__ == '__main__':
    main()