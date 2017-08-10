#!/usr/bin/env python
import numpy as np
import openravepy as orpy

def draw_point(env, point, linewidth=10, color=(0, 1, 0)):
  """
  Draw a colored point to the OpenRAVE environment.

  Parameters
  ----------
  point: list or array_like
    point: Position of the point.
  linewidth: float
    Size of the point.
  color: list or array_like
    Color of the point.

  Returns
  -------
  handles: orpy.GraphHandle
    Handles holding the plot.
  """
  iktype = orpy.IkParameterizationType.Translation3D
  ikparam = orpy.IkParameterization(point, iktype)
  handles = orpy.misc.DrawIkparam2(env, ikparam, linewidth=linewidth,
                                                                coloradd=color)
  return handles

def draw_axes(env, transform, dist=0.03, linewidth=2):
  """
  Draw an RGB set of axes to the OpenRAVE environment.

  Parameters
  ----------
  transform: string
    Transformation where to draw the axes.
  dist: float
    Length of the axes (meters).
  linewidth: float
    Line width of the axes (pixels).

  Returns
  -------
  hanles: orpy.GraphHandle
    The total axes added to the scene.
  """
  iktype = orpy.IkParameterizationType.Transform6D
  ikparam = orpy.IkParameterization(transform, iktype)
  handles = orpy.misc.DrawIkparam2(env, ikparam, dist=dist, linewidth=linewidth)
  return handles

def draw_plane(env, plane, extents=(4,4), texture=None):
  """
  Draw a plane to the OpenRAVE environment.

  Parameters
  ----------
  plane: criros.spalg.Plane
    Plane object constructed from its origin and normal.
  extents: list or array_like
    The length and width of the drawn plane rectangle.
  texture: array_like
    Texture of the drawn plane. An array of size N*M*4, with [:, :, 0], [:, :, 1], [:, :, 2] and [:, :, 3]
    describe the changes of the transparency and RGB colors of the plane.

  Returns
  -------
  handles: orpy.GraphHandle
    Handles holding the plot.
  """
  if texture is None:
    texture = np.zeros((100,100,4))
    texture[:,:,1] = 0.2
    texture[:,:,2] = 0.2
    texture[:,:,3] = 0.2
  T = plane.get_transform()
  with env:
    handles = env.drawplane(transform=T, extents=extents, texture=texture)
  return handles

def draw_ray(env, ray, dist=0.03, linewidth=2, color=None):
  """
  Draw an arrow with the orpy.linelist heads to the OpenRAVE environment.

  Parameters
  ----------
  ray: orpy.ray
    The ray describes the origin and the direction of the arrow.
  dist: float
    Length of the arrow heads.
  linewidth: float
    Line width of the arrow (pixels).
  color: list or array_like
    Color of the arrow.

  Returns
  -------
  handles: orpy.GraphHandle
    Handles holding the plot.
  """
  if dist < 0:
    newpos = ray.pos() + dist*ray.dir()
    newray = orpy.Ray(newpos, ray.dir())
  else:
    newray = ray
  iktype = orpy.IkParameterizationType.TranslationDirection5D
  ikparam = orpy.IkParameterization(ray, iktype)
  handles = orpy.misc.DrawIkparam2(env, ikparam, dist=dist,
                                          linewidth=linewidth, coloradd=color)
  return handles

def draw_spline(env, spline, num=100, linewidth=1.0, colors=(0,0,1), start=0,
                                                                        stop=1):
  """
  Draw line strips for the sequential segments of a curve to the OpenRAVE environment.

  Parameters
  ----------
  spline: scipy.interpolate.BSpline
    B-spline of the curve.
  num: int
    Number of segments to draw.
  linewidth: float
    Line width of the line strip (pixels).
  colors: list or array_like
    Colors of the line strips.
  start: float
    Start drawing position of the curve, in ``[0, 1]``.
  stop: float
    End drawing position of the curve, in ``[0, 1]``.

  Returns
  -------
  handles: orpy.GraphHandle
    Handles holding the plot.
  """
  if colors is None:
    colors = [0,0,1] * num
  linestrip = []
  for u in np.linspace(start, stop, num=num):
    linestrip.append(spline(u))
  handle = env.drawlinestrip(np.array(linestrip), linewidth=linewidth,
                                                      colors=np.array(colors))
  return handle
