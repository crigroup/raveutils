#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
# Tested package
import raveutils as ru


class Test_visual(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    np.set_printoptions(precision=6, suppress=True)
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    cls.env = env
    print('') # dummy line

  @classmethod
  def tearDownClass(cls):
    cls.env.Reset()
    cls.env.Destroy()

  def test_draw_axes(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    transform[:3,3] = np.random.randn(3)*0.5
    h = ru.visual.draw_axes(self.env, transform)
    self.assertEqual(len(h), 1)
    self.assertEqual(type(h[0]), orpy.GraphHandle)

  def test_draw_plane(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    transform[:3,3] = np.random.randn(3)*0.5
    h = ru.visual.draw_plane(self.env, transform)
    self.assertEqual(type(h), orpy.GraphHandle)

  def test_draw_point(self):
    np.random.seed(123)
    point = np.random.randn(3)
    h = ru.visual.draw_point(self.env, point)
    self.assertEqual(len(h), 1)
    self.assertEqual(type(h[0]), orpy.GraphHandle)

  def test_draw_ray(self):
    np.random.seed(123)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    position = np.random.randn(3)*0.5
    ray = orpy.Ray(position, direction)
    handles = ru.visual.draw_ray(self.env, ray)
    self.assertEqual(len(handles), 3)
    types = [type(h) for h in handles]
    self.assertEqual(len(set(types)), 1)
    self.assertEqual(set(types), {orpy.GraphHandle})
    # Use negative distance
    handles = ru.visual.draw_ray(self.env, ray, dist=-0.03)
    self.assertEqual(len(handles), 3)
