#! /usr/bin/env python
import unittest
import numpy as np
import baldor as br
# PyKDL
import PyKDL
from tf_conversions import posemath
# Tested package
import raveutils as ru


class Test_transforms(unittest.TestCase):
  def test_compute_twist(self):
    for _ in range(100):
      T0 = br.transform.random()
      vel = np.random.rand(3)
      rot = np.random.rand(3)
      kdl_twist = PyKDL.Twist(PyKDL.Vector(*vel), PyKDL.Vector(*rot))
      F0 = posemath.fromMatrix(T0)
      F1 = PyKDL.addDelta(F0, kdl_twist)
      T1 = posemath.toMatrix(F1)
      twist = ru.transforms.compute_twist(F0, F1)
      np.testing.assert_allclose(twist, np.hstack((vel,rot)))
