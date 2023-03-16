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

def polyx_fourth(t):
    return 0.0013 * t**4 - 0.0629 * t**3 + 0.8751 * t**2 - 2.8376 * t + 2.2531

def polyy_fourth(t):
    return 0.0024 * t**4 - 0.0641 * t**3 + 0.3837 * t**2 + 0.7598 * t - 1.2001

def polyx_third(t):
    return -0.0211 * t**3 + 0.4353 * t**2 - 1.1507 * t + 0.5172

def polyy_third(t):
    return 0.0116 * t**3 - 0.4133 * t**2 + 3.8168 * t - 4.3458

class RestToRest(Node):
    def __init__(self):
        super().__init__('rest_to_rest')

        self.log = False
        self.pos = []
        self.vel = []
        self.acc = []

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
        self.altitude = 2.5

        vstates = generate_states_vector(map, [-2, -1, 0, 1, 2])
        vactions = generate_actions_vector([-1, 0, 1])

        env = GridEnvironment(map)
        agent = DynamicalSystem(vstates, vactions, -2, 2)
        
        self.qlearning = QLearning(initial_state, final_state, agent, env, 0.9, 0.9, 0.1,  False)
        self.qlearning.load(os.path.join(get_package_share_directory('rest_to_rest'), 'rest_to_rest/models/Q.npy'))

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.state_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.subscribe_state,
            qos_profile)

        self.execute_path_action = self.declare_parameter('execute_trajectory_action', 'execute_trajectory')
        self.execute_path_action_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

    def subscribe_state(self, state):
        if self.log:
            self.pos.append((state.y, state.x))
            self.vel.append((state.vy, state.vx))
            self.acc.append((state.ay, state.ax))

    def get_path(self):
        policy = self.qlearning.get_policy()       

        setpoints = []
        print(policy[0])
        for i in range(policy[0].shape[0]):
            s = policy[0][i]
            a = policy[1][i]
            #x = policy[0][i]
            #y = policy[1][i]
            setpoint = TrajectorySetpoint()
            setpoint.position = [float(s[0]), float(s[1]), self.altitude]
            #setpoint.velocity = [float(s[2]), float(s[3]), 0.0]
            #setpoint.acceleration = [float(a[0]), float(a[1]), 0.0]
            setpoint.yaw = math.pi / 2
            setpoints.append(setpoint)
        
        return Trajectory(setpoints=setpoints)

    def get_polynomial_path(self):
        policy = np.array((polyx_fourth(np.linspace(0, 14, 150)), polyy_fourth(np.linspace(0, 14, 150))))

        setpoints = []
        print(policy[0])
        for i in range(policy[0].shape[0]):
            x = policy[0][i]
            y = policy[1][i]
            setpoint = TrajectorySetpoint()
            setpoint.position = [float(x), float(y), self.altitude]
            #setpoint.velocity = [float(s[2]), float(s[3]), 0.0]
            #setpoint.acceleration = [float(a[0]), float(a[1]), 0.0]
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
        self.plot()
        self.save_measurements()
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        self.log = True
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.progress))

    def save_measurements(self):
        measurements = np.array([self.pos, self.vel, self.acc])

        np.save('/home/bencic/measurements', measurements)

    def plot(self):
        pos = np.array(self.pos)
        vel = np.array(self.vel)
        acc = np.array(self.acc)
        size = pos.shape[0]

        fig = plt.figure('Summary', figsize=(15, 15), tight_layout=True)
        gs = gridspec.GridSpec(3, 2)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$x$ [m]')
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_yticks(np.arange(0, 15, 2))
        ax.set_xlim([-0.5, 14.5])
        ax.set_ylim([-0.5, 14.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), pos[:, 0])
        
        ax = fig.add_subplot(gs[0, 1])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_yticks(np.arange(0, 15, 2))
        ax.set_xlim([-0.5, 14.5])
        ax.set_ylim([-0.5, 14.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), pos[:, 1])
        
        ax = fig.add_subplot(gs[1, 0])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$v_x$ [m/step]')
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.set_xlim([-0.5, 14.5])
        ax.set_ylim([-2.5, 2.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), vel[:, 0])

        ax = fig.add_subplot(gs[1, 1])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$v_y$ [m/step]')
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.set_xlim([-0.5, 14.5])
        ax.set_ylim([-2.5, 2.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), vel[:, 1])

        ax = fig.add_subplot(gs[2, 0])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$u_{x}$ [m/step]')
        #ax.set_xlim([-0.5, 14.5])
        #ax.set_ylim([-1.5, 1.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), acc[:, 0])

        ax = fig.add_subplot(gs[2, 1])
        ax.set_xlabel(r'$t$ [step]')
        ax.set_ylabel(r'$u_{y}$ [m/step]')
        #ax.set_xlim([-0.5, 14.5])
        #ax.set_ylim([-1.5, 1.5])
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(np.linspace(0, 14, size), acc[:, 1])

        plt.show()


def main(args=None):
    rclpy.init(args=args)

    trajectory_generator = RestToRest()
    trajectory = trajectory_generator.get_polynomial_path()
    trajectory_generator.send_goal(trajectory)
    rclpy.spin(trajectory_generator)


if __name__ == '__main__':
    main()