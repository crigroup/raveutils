#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
from raveutils import conversions
import raveutils.transforms as tr


class TestModule(unittest.TestCase):
  def test_ray_conversions(self):
    # Create the ray
    position = np.array([0.5,  -0.25,  0.1])
    direction = tr.unit_vector([-0.2506, 0.6846, 0.6846])
    ray = orpy.Ray(position, direction)
    # Check that the conversion works in both direction
    transform = conversions.from_ray(ray)
    ray_from_transform = conversions.to_ray(transform)
    np.testing.assert_allclose(ray.pos(), ray_from_transform.pos())
    np.testing.assert_allclose(ray.dir(), ray_from_transform.dir())
