# Rest to Rest Trajectory Generation with Q Learning

This is a ROS2 Python package containing a trajectory generation node for rest-to-rest trajectory
generation with Q-Learning as described in "Rest-to-Rest Trajectory Generation with Q-Learning".

To run this node you need to follow the instructions at [px_autonomous](https://github.com/AImotion-Flight/px4_autonomous)
to setup the basic ROS2 PX4 SITL Offboard control.

After that the node can be built and started with:
```bash
colcon build
ros2 run rest_to_rest rest_to_rest
```
