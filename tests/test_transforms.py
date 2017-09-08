#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
import raveutils.transforms as tr


class TestModule(unittest.TestCase):
  def test_perpendicular_vector(self):
    np.random.seed(123)
    vector = np.random.randn(3)
    vector /= np.linalg.norm(vector)
    pervector = tr.perpendicular_vector(vector)
    np.testing.assert_almost_equal(0, np.dot(vector, pervector))

  def test_transform_between_axes(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    newaxis = np.dot(transform[:3,:3], tr.Z_AXIS)
    est_transform = tr.transform_between_axes(tr.Z_AXIS, newaxis)
    np.testing.assert_allclose(transform[:3,2], est_transform[:3,2])

  def test_transform_inv(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    transform[:3,3] = np.random.randn(3)*0.5
    inv = tr.transform_inv(transform)
    np.testing.assert_array_almost_equal(np.eye(4), np.dot(transform, inv))

  def test_unit_vector(self):
    np.random.seed(123)
    vector = np.random.randn(3)
    unit = vector / np.linalg.norm(vector)
    np.testing.assert_allclose(unit, tr.unit_vector(vector))
