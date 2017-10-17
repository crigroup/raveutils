#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
# Tested package
import raveutils as ru


class Test_transforms(unittest.TestCase):
  def test_perpendicular_vector(self):
    np.random.seed(123)
    vector = np.random.randn(3)
    vector /= np.linalg.norm(vector)
    pervector = ru.transforms.perpendicular_vector(vector)
    np.testing.assert_almost_equal(0, np.dot(vector, pervector))

  def test_transform_between_axes(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    newaxis = np.dot(transform[:3,:3], ru.transforms.Z_AXIS)
    est_transform = ru.transforms.transform_between_axes(ru.transforms.Z_AXIS,
                                                                        newaxis)
    np.testing.assert_allclose(transform[:3,2], est_transform[:3,2])

  def test_transform_inv(self):
    np.random.seed(123)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45)
    transform = orpy.matrixFromAxisAngle(angle*axis)
    transform[:3,3] = np.random.randn(3)*0.5
    inv = ru.transforms.transform_inv(transform)
    np.testing.assert_array_almost_equal(np.eye(4), np.dot(transform, inv))

  def test_unit_vector(self):
    np.random.seed(123)
    vector = np.random.randn(3)
    unit = vector / np.linalg.norm(vector)
    np.testing.assert_allclose(unit, ru.transforms.unit_vector(vector))
