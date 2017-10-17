#!/usr/bin/env python
import itertools
import numpy as np
import openravepy as orpy
from scipy.spatial import ConvexHull, Delaunay
# Image geometry
from image_geometry import PinholeCameraModel
# Local modules
import raveutils as ru


class CameraFOV(object):
  """
  Base class for the field-of-view of a Pin hole camera.
  """
  def __init__(self, camera_info, maxdist, transform=None):
    """
    Parameters
    ----------
    camera_info: sensor_msgs/CameraInfo
      Meta information of the camera. For more details check the
      `sensor_msgs/CameraInfo
      <http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html>`_
      documentation.
    maxdist: float
     Maximum distance the FOV covers in the Z direction of the camera frame.
    transform: array_like
     Homogenous transformation of the camera reference frame. If `None` then
     the identity matrix is used.
    """
    if transform is None:
      transform = np.eye(4)
    self.maxdist = maxdist
    self.transform = np.array(transform)
    self.cam_model = PinholeCameraModel()
    self.cam_model.fromCameraInfo(camera_info)
    # Compute FOV corners
    delta_x = self.cam_model.getDeltaX(self.cam_model.width/2, self.maxdist)
    delta_y = self.cam_model.getDeltaY(self.cam_model.height/2, self.maxdist)
    self.corners = np.zeros((5,3))
    self.corners[0,:] = transform[:3,3]
    idx = 1
    for k in itertools.product([-1,1],[-1,1]):
      point = np.array([0, 0, self.maxdist, 1])
      point[:2] = np.array([delta_x, delta_y]) * np.array(k)
      self.corners[idx,:] = np.dot(transform, point)[:3]
      idx += 1

  def get_corners(self):
    """
    Get the five corners of the camera field of view

    Returns
    -------
    corners: array_like
      A 5x3 array with the five corners of the camera field of view
    """
    corners = np.array(self.corners)
    return corners

  def get_trimesh(self):
    """
    Get the convex hull that representes the field of view as a trimesh

    Returns
    -------
    vertices: array_like
      The trimesh vertices
    faces: array_like
      The trimesh faces

    See Also
    --------
    raveutils.mesh.trimesh_from_point_cloud
    """
    corners = self.get_corners()
    vertices, faces = ru.mesh.trimesh_from_point_cloud(corners)
    return vertices, faces

  def contains(self, points):
    """
    Check if all the XYZ points are inside the camera field of view

    Parameters
    ----------
    points: array_like
      List of XYZ points

    Returns
    -------
    all_inside: bool
      True if all the points are inside the FOV. False otherwise.
    """
    hull = ConvexHull(self.corners)
    triangulation = Delaunay(self.corners[hull.vertices])
    all_inside = np.alltrue(triangulation.find_simplex(points)>=0)
    return all_inside

  def random_point_inside(self):
    """
    Generate a random XYZ point inside the camera field of view

    Returns
    -------
    random_point: array_like
      The random XYZ point inside the camera field of view
    """
    z = self.maxdist*np.random.random()
    delta_x = self.cam_model.getDeltaX(self.cam_model.width/2, z)
    delta_y = self.cam_model.getDeltaY(self.cam_model.height/2, z)
    point = np.array([0, 0, z, 1])
    point[:2] = np.array([delta_x,delta_y]) * (2*np.random.random_sample(2)-1.)
    random_point = np.dot(self.transform, point)[:3]
    return random_point
