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

  def test_plan_to_joint_configuration(self):
    robot = self.robot
    np.random.seed(123)
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

  def test_retime_trajectory(self):
    robot = self.robot
    np.random.seed(123)
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
    robot = self.robot
    np.random.seed(123)
    qgoal = ru.kinematics.random_joint_values(robot)
    traj = ru.planning.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    ros_traj = ru.planning.ros_trajectory_from_openrave(robot.GetName(), traj)
    # Check trajs durations
    ros_traj_duration = ros_traj.points[-1].time_from_start.to_sec()
    np.testing.assert_almost_equal(ros_traj_duration, traj.GetDuration())
    # Check num of waypoints
    self.assertEqual(len(ros_traj.points), traj.GetNumWaypoints())


  def test_trajectory_from_waypoints(self):
    robot = self.robot
    np.random.seed(123)
    waypoints = []
    for i in range(5):
      waypoints.append(ru.kinematics.random_joint_values(robot))
    traj = ru.planning.trajectory_from_waypoints(robot, waypoints)
    self.assertEqual(traj.GetNumWaypoints(), len(waypoints))
