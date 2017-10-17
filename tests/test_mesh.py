#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
import raveutils as ru


class Test_mesh(unittest.TestCase):
  def test_trimesh_from_point_cloud(self):
    cloud = np.random.random_sample((100, 3))
    vertices, faces = ru.mesh.trimesh_from_point_cloud(cloud)
