#!/usr/bin/env python
import numpy as np
import scipy.spatial
# Local modules
import raveutils as ru


def trimesh_from_point_cloud(cloud):
  """
  Convert a point cloud into a convex hull trimesh

  Parameters
  ----------
  cloud: array_like
    The input point cloud. It can be ``pcl.Cloud`` or :obj:`numpy.array`

  Returns
  -------
  vertices: array_like
    The trimesh vertices
  faces: array_like
    The trimesh faces

  See Also
  --------
  :obj:`scipy.spatial.ConvexHull`: For more details about convex hulls
  """
  points = np.asarray(cloud)
  hull = scipy.spatial.ConvexHull(points)
  hull = scipy.spatial.ConvexHull(points[hull.vertices])
  ru.transforms.counterclockwise_hull(hull)
  vertices = hull.points
  faces = hull.simplices
  return vertices, faces
