#!/usr/bin/env python
import math
import numpy as np
import openravepy as orpy


X_AXIS = np.array([1., 0., 0.], dtype=np.float64)
Y_AXIS = np.array([0., 1., 0.], dtype=np.float64)
Z_AXIS = np.array([0., 0., 1.], dtype=np.float64)


def counterclockwise_hull(hull):
  """
  Make the edges counterclockwise order

  Parameters
  ----------
  hull: scipy.spatial.ConvexHull
    Convex hull to be re-ordered.
  """
  midpoint = np.sum(hull.points, axis=0) / hull.points.shape[0]
  for i,simplex in enumerate(hull.simplices):
    x, y, z = hull.points[simplex]
    voutward = (x + y + z) / 3 - midpoint
    vccw = np.cross((y - x), (z - y))
    if np.inner(vccw, voutward) < 0:
      hull.simplices[i] = [simplex[0], simplex[2], simplex[1]]

def perpendicular_vector(vector):
  """
  Find an arbitrary perpendicular vector

  Parameters
  ----------
  vector: array_like
    The input vector

  Returns
  -------
  perpendicular: array_like
    The perpendicular vector
  """
  unit = unit_vector(vector)
  if np.allclose(unit[:2], np.zeros(2)):
    if np.isclose(unit[2], 0.):
      # unit is (0, 0, 0)
      raise ValueError('Input vector cannot be a zero vector')
    # unit is (0, 0, Z)
    perpendicular = np.array(Y_AXIS, dtype=np.float64, copy=True)
  perpendicular = np.array([-unit[1], unit[0], 0], dtype=np.float64)
  return perpendicular

def transform_between_axes(axis_a, axis_b):
  """
  Compute the transformation that aligns two vectors/axes.

  Parameters
  ----------
  axis_a: array_like
    The initial axis
  axis_b: array_like
    The goal axis

  Returns
  -------
  transform: array_like
    The transformation that transforms axis ``axis_a`` into axis ``axis_b``
  """
  a_unit = unit_vector(axis_a)
  b_unit = unit_vector(axis_b)
  c = np.dot(a_unit, b_unit)
  angle = np.arccos(c)
  if np.isclose(c, -1.0) or np.allclose(a_unit, b_unit):
    axis = perpendicular_vector(b_unit)
  else:
    axis = unit_vector(np.cross(a_unit, b_unit))
  transform = orpy.matrixFromAxisAngle(angle*axis)
  return transform

def transform_inv(transform):
  """
  Compute the inverse of an homogeneous transformation.

  .. note:: This function is more efficient than :obj:`numpy.linalg.inv` given
    the special properties of homogeneous transformations.

  Parameters
  ----------
  transform: array_like
    The input homogeneous transformation

  Returns
  -------
  transform: array_like
    The resulting homogeneous transformation inverse
  """
  R = transform[:3,:3].T
  p = transform[:3,3]
  inv = np.eye(4)
  inv[:3,:3] = R
  inv[:3,3] = np.dot(-R, p)
  return inv


def unit_vector(vector):
  """
  Return unit vector (normalized)

  Parameters
  ----------
  vector: array_like
    The input vector

  Returns
  -------
  unit: array_like
    The resulting unit vector (normalized)
  """
  unit = np.array(vector, dtype=np.float64, copy=True)
  unit /= math.sqrt(np.dot(vector, vector))
  return unit
