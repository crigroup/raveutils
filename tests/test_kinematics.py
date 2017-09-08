#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
from raveutils import kinematics


class TestModule(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    np.set_printoptions(precision=6, suppress=True)
    cls.env = orpy.Environment()
    if not cls.env.Load('data/lab1.env.xml'):
      raise Exception('Could not load scene: data/lab1.env.xml')
    cls.robot = cls.env.GetRobot('BarrettWAM')
    np.random.seed(123)
    q = kinematics.random_joint_values(cls.robot)
    cls.robot.SetActiveDOFValues(q)
    print('') # dummy line

  @classmethod
  def tearDownClass(cls):
    cls.env.Reset()
    cls.env.Destroy()

  def test_compute_jacobian(self):
    robot = self.robot
    # Try the function with its variants
    Jtrans = kinematics.compute_jacobian(robot, translation_only=True)
    J = kinematics.compute_jacobian(robot, translation_only=False)

  def test_compute_yoshikawa_index(self):
    robot = self.robot
    # Try the function with its variants
    idx = kinematics.compute_yoshikawa_index(robot, penalize_jnt_limits=True)
    idx = kinematics.compute_yoshikawa_index(robot, penalize_jnt_limits=False)

  def test_load_ikfast_and_find_ik_solutions(self):
    robot = self.robot
    # Test loading IKFast
    iktype = orpy.IkParameterizationType.Transform6D
    success = kinematics.load_ikfast(robot, iktype, autogenerate=False)
    self.assertTrue(success)
    # Test find IK solutions
    manip = robot.GetActiveManipulator()
    q = robot.GetActiveDOFValues()
    T = manip.GetEndEffectorTransform()
    iktype = orpy.IkParameterizationType.Transform6D
    solutions = kinematics.find_ik_solutions(robot, T, iktype,
                                                          collision_free=False)
    self.assertTrue(len(solutions) > 0)

  def test_load_link_stats(self):
    robot = self.robot
    # Generate the Link Statistics database
    success = kinematics.load_link_stats(robot, autogenerate=True)
    self.assertTrue(success)
