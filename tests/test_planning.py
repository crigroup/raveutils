#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
# Tested package
import raveutils as ru


class Test_planning(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    np.set_printoptions(precision=6, suppress=True)
    scene = 'robots/puma.robot.xml'
    env = orpy.Environment()
    if not env.Load(scene):
      raise Exception('Could not load scene: {0}'.format(scene))
    robot = env.GetRobot('PumaGripper')
    manip = robot.SetActiveManipulator('arm')
    robot.SetActiveDOFs(manip.GetArmIndices())
    # Store the environment and robot in the class
    cls.env = env
    cls.robot = robot
    print('') # dummy line

  @classmethod
  def tearDownClass(cls):
    cls.env.Reset()
    cls.env.Destroy()

  def test_plan_cartesian_twist(self):
    np.random.seed(123)
    env = self.env
    robot = self.robot
    distances = [0.01, 0.02, 0.1, 0.5, 1.]
    for dist in distances:
      while True:
        q = ru.kinematics.random_joint_values(robot)
        with env:
          robot.SetActiveDOFValues(q)
        if not env.CheckCollision(robot):
          break
      twist = np.random.rand(6)*dist
      traj = ru.planning.plan_cartesian_twist(robot, twist, num_waypoints=10)

  def test_plan_to_joint_configuration(self):
    np.random.seed(123)
    robot = self.robot
    qgoal = ru.kinematics.random_joint_values(robot)
    # Test all the available planners
    traj1 = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    self.assertNotEqual(traj1, None)
    traj = ru.planning.plan_to_joint_configuration(robot, qgoal,
                                                              pname='BasicRRT')
    self.assertNotEqual(traj, None)
    # Test swaping option
    traj2 = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT',
                                                                  try_swap=True)
    self.assertNotEqual(traj2, None)
    # Test without post-processing
    traj3 = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT',
                                                                max_ppiters=-1)
    self.assertNotEqual(traj3, None)

  def test_retime_trajectory(self):
    np.random.seed(123)
    robot = self.robot
    qgoal = ru.kinematics.random_joint_values(robot)
    traj = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    # Test all the available retiming methods
    status = ru.planning.retime_trajectory(robot, traj,
                                                      'LinearTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)
    status = ru.planning.retime_trajectory(robot, traj,
                                                  'ParabolicTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)
    status = ru.planning.retime_trajectory(robot, traj,
                                                      'CubicTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)

  def test_ros_trajectory_from_openrave(self):
    np.random.seed(123)
    robot = self.robot
    qgoal = ru.kinematics.random_joint_values(robot)
    traj = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    ros_traj = ru.planning.ros_trajectory_from_openrave(robot.GetName(), traj)
    # Check trajs durations
    ros_traj_duration = ros_traj.points[-1].time_from_start.to_sec()
    np.testing.assert_almost_equal(ros_traj_duration, traj.GetDuration())
    # Check num of waypoints
    self.assertEqual(len(ros_traj.points), traj.GetNumWaypoints())
    # TODO: Send trajectory with repeated waypoints

  def test_trajectory_from_waypoints(self):
    np.random.seed(123)
    robot = self.robot
    waypoints = []
    for i in range(5):
      waypoints.append(ru.kinematics.random_joint_values(robot))
    traj = ru.planning.trajectory_from_waypoints(robot, waypoints)
    self.assertEqual(traj.GetNumWaypoints(), len(waypoints))
