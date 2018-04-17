#!/usr/bin/env python
import itertools
import numpy as np
import baldor as br
import openravepy as orpy
# Local modules
import raveutils as ru


def enable_body(body, enable):
  """
  Enables all the links of a body.

  Parameters
  ----------
  body: orpy.KinBody
    The OpenRAVE body
  enable: bool
   If true, will enable all the links.
  """
  env = body.GetEnv()
  with env:
    for link in body.GetLinks():
      link.Enable(enable)

def get_bounding_box_corners(body, transform=None, scale=1.):
  """
  Get the bounding box corners (8 corners) for the given body.
  If ``transform`` is given the corners are transformed properly.
  The ``scale`` parameters is a factor used to scale the extents of the
  bounding box.

  Parameters
  ----------
  body: orpy.KinBody
    The OpenRAVE body
  transform: array_like
    Homogeneous transformation of the body. If ``None``, the corners are given
    using the current pose of the body in OpenRAVE.
  scale: float
    The scale factor to modify the extents of the bounding box.

  Returns
  -------
  corners: list
   List containing the 8 box corners. Each corner is a XYZ ``np.array``
  """
  if transform is not None:
    Tinv = br.transform.inverse(body.GetTransform())
  aabb = body.ComputeAABB()
  corners = []
  for k in itertools.product([-1,1],[-1,1],[-1,1]):
    position = aabb.pos() + np.array(k)*aabb.extents()*scale
    if transform is not None:
      homposition = np.hstack((position,1))
      homposition = np.dot(Tinv, homposition)
      position = np.dot(transform, homposition)[:3]
    corners.append(position)
  return corners

def set_body_color(body, diffuse, ambient=None):
  """
  Override diffuse and ambient color of the body

  Parameters
  ----------
  body: orpy.KinBody
    The OpenRAVE body
  diffuse: array_like
    The input diffuse color in RGB format (3 elements array)
  ambient: array_like
    The input ambient color in RGB format (3 elements array)

  Notes
  -----
  The value of each color channel (R, G and B) must be between :math:`[0, 1]`
  """
  env = body.GetEnv()
  is_ambient_available = ambient is not None
  with env:
    for link in body.GetLinks():
      for geom in link.GetGeometries():
        geom.SetDiffuseColor(diffuse)
        if is_ambient_available:
          geom.SetAmbientColor(ambient)

def set_body_transparency(body, transparency=0.0, links=None):
  """
  Set the transparency value of all the body's geometries

  Parameters
  ----------
  body: orpy.KinBody
    The OpenRAVE body
  transparency: float
    The transparency value. If it's out of range [0.0, 1.0], it'll be clipped.
  links: list
   List of links to be modified. By default all the links are.
  """
  if links is None:
    links_to_modify = [link.GetName() for link in body.GetLinks()]
  else:
    links_to_modify = links
  transparency = np.clip(transparency, 0.0, 1.0)
  env = body.GetEnv()
  with env:
    for link in body.GetLinks():
      if link.GetName() in links_to_modify:
        for geom in link.GetGeometries():
          geom.SetTransparency(transparency)
