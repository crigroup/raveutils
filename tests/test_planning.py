#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
from raveutils import kinematics as orkin
from raveutils import planning as orplan


class TestModule(unittest.TestCase):
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
    qgoal = orkin.random_joint_values(robot)
    # Test all the available planners
    traj = orplan.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    self.assertNotEqual(traj, None)
    traj = orplan.plan_to_joint_configuration(robot, qgoal, pname='BasicRRT')
    self.assertNotEqual(traj, None)

  def test_retime_trajectory(self):
    robot = self.robot
    np.random.seed(123)
    qgoal = orkin.random_joint_values(robot)
    traj = orplan.plan_to_joint_configuration(robot, qgoal, pname='BiRRT')
    # Test all the available retiming methods
    status = orplan.retime_trajectory(robot, traj, 'LinearTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)
    status = orplan.retime_trajectory(robot, traj,
                                                  'ParabolicTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)
    status = orplan.retime_trajectory(robot, traj, 'CubicTrajectoryRetimer')
    self.assertEqual(status, orpy.PlannerStatus.HasSolution)

  def test_trajectory_from_waypoints(self):
    robot = self.robot
    np.random.seed(123)
    waypoints = []
    for i in range(5):
      waypoints.append(orkin.random_joint_values(robot))
    traj = orplan.trajectory_from_waypoints(robot, waypoints)
    self.assertEqual(traj.GetNumWaypoints(), len(waypoints))
