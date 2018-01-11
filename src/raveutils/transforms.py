#!/usr/bin/env python
import math
import numpy as np
import openravepy as orpy
# PyKDL
import PyKDL
from tf_conversions import posemath


def compute_twist(T0, T1):
  """
  Compute the twist between two homogeneous transformations: `twist = T1 - T0`

  Parameters
  ----------
  T0: array_like
    Initial homogeneous transformation
  T1: array_like
    Final homogeneous transformation
  """
  F0 = posemath.fromMatrix(T0)
  F1 = posemath.fromMatrix(T1)
  kdl_twist = PyKDL.diff(F0, F1)
  twist = np.zeros(6)
  twist[:3] = [kdl_twist.vel.x(),kdl_twist.vel.y(),kdl_twist.vel.z()]
  twist[3:] = [kdl_twist.rot.x(),kdl_twist.rot.y(),kdl_twist.rot.z()]
  return twist

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
