import os
import math
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from px4_autonomous_interfaces.msg import Trajectory
from px4_autonomous_interfaces.action import ExecuteTrajectory
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from .qlearning import QLearning
from .util import *
from .environment import GridEnvironment
from .agent import DynamicalSystem
from transforms3d.euler import euler2quat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class RestToRest(Node):
    def __init__(self):
        super().__init__('rest_to_rest')

        self.grid = np.ones((10, 10))
        self.grid[0:8, 1] = 0
        self.grid[7:, 4] = 0
        self.grid[0:5, 4] = 0
        self.grid[2:, 7] = 0

        env = GridEnvironment(self.grid)
        size = np.shape(self.grid)
        vstates = generate_states_vector(size[1], size[0], range(-2, 2 + 1))
        vactions = generate_actions_vector(range(-1, 1 + 1))
        agent = DynamicalSystem(vstates, vactions, -2, 2)
        self.qlearning = QLearning((0, 0, 0, 0), (9, 9, 0, 0), agent, env, 50000, 0.9, 0.9, 0.1, False)
        self.qlearning.load(os.path.join(get_package_share_directory('rest_to_rest'), 'rest_to_rest/models/Q.npy'))

        self.execute_path_action = self.declare_parameter('execute_trajectory_action', 'execute_trajectory')
        self.execute_path_action_client = ActionClient(self, ExecuteTrajectory, '/uav_1/execute_trajectory')

    def get_trajectory(self):
        policy = self.qlearning.get_policy()       

        setpoints = []
        print(policy[0])
        for i in range(policy[0].shape[0]):
            s = policy[0][i]
            setpoint = TrajectorySetpoint()
            setpoint.position = [float(s[0]), float(s[1]), 1.5]
            setpoint.yaw = math.pi / 2
            setpoints.append(setpoint)
        
        return Trajectory(setpoints=setpoints)

    def send_goal(self, trajectory):
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.execute_path_action_client.wait_for_server()

        self.send_goal_future = self.execute_path_action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Trajectory execution rejected')
            return

        self.get_logger().info('Trajectory execution accepted')

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.log = False
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.final_reached))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        self.log = True
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.progress))

def main(args=None):
    rclpy.init(args=args)

    trajectory_generator = RestToRest()
    trajectory = trajectory_generator.get_trajectory()
    trajectory_generator.send_goal(trajectory)
    rclpy.spin(trajectory_generator)


if __name__ == '__main__':
    main()