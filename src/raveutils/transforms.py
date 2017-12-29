#!/usr/bin/env python
import math
import numpy as np
import openravepy as orpy


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
