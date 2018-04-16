#!/usr/bin/env python
import numpy as np
import openravepy as orpy


def draw_axes(env, transform, dist=0.03, linewidth=2):
  """
  Draw an RGB set of axes into the OpenRAVE environment.

  Parameters
  ----------
  env: orpy.Environment
    The OpenRAVE environment
  transform: array_like
    The homogeneous transformation where to draw the axes
  dist: float
    Length of the axes (meters)
  linewidth: float
    Line width of the axes (pixels)

  Returns
  -------
  h: orpy.GraphHandle
    The total axes added to the scene.
  """
  iktype = orpy.IkParameterizationType.Transform6D
  ikparam = orpy.IkParameterization(transform, iktype)
  h = orpy.misc.DrawIkparam2(env, ikparam, dist=dist, linewidth=linewidth)
  return h

def draw_plane(env, transform, extents=(4,4), texture=None):
  """
  Draw a plane into the OpenRAVE environment.

  Parameters
  ----------
  env: orpy.Environment
    The OpenRAVE environment
  transform: array_like
    Transform that represents the plane. The origin of the plane corresponds to
    the position of the transform. Axis Z of the transform is used as the plane
    normal
  extents: list
    The length and width of the plane (meters)
  texture: array_like
    Texture of the drawn plane. It must have a size `N*M*4`. The last dimension
    corresponds to the RGBA channels.

  Returns
  -------
  h: orpy.GraphHandle
    Handles holding the plot.
  """
  if texture is None:
    texture = np.zeros((100,100,4))
    texture[:,:,1] = 0.2
    texture[:,:,2] = 0.2
    texture[:,:,3] = 0.2
  with env:
    h = env.drawplane(transform, extents=extents, texture=texture)
  return h

def draw_point(env, point, size=10, color=(0, 1, 0)):
  """
  Draw a colored point into the OpenRAVE environment.

  Parameters
  ----------
  env: orpy.Environment
    The OpenRAVE environment
  point: array_like
    XYZ position of the point
  size: float
    Size of the point (pixels)
  color: array_like
    Color of the point in the format `(red, green, blue, alpha)`

  Returns
  -------
  h: orpy.GraphHandle
    Handles of the plot. This is require for the point to stay on the
    environment
  """
  iktype = orpy.IkParameterizationType.Translation3D
  ikparam = orpy.IkParameterization(point, iktype)
  h = orpy.misc.DrawIkparam2(env, ikparam, linewidth=size, coloradd=color)
  return h

def draw_ray(env, ray, dist=0.03, linewidth=2, color=None):
  """
  Draw a ray as an arrow + line into the OpenRAVE environment.

  Parameters
  ----------
  env: orpy.Environment
    The OpenRAVE environment
  ray: orpy.ray
    The input ray  with the position and direction of the arrow
  dist: float
    Length of the line
  linewidth: float
    Linewidth of the arrow and line (pixels)
  color: array_like
    Color of the arrow in the format `(red, green, blue, alpha)`

  Returns
  -------
  h: orpy.GraphHandle
    Handles holding the plot.
  """
  if dist < 0:
    newpos = ray.pos() + dist*ray.dir()
    newray = orpy.Ray(newpos, ray.dir())
  else:
    newray = ray
  iktype = orpy.IkParameterizationType.TranslationDirection5D
  ikparam = orpy.IkParameterization(ray, iktype)
  h = orpy.misc.DrawIkparam2(env, ikparam, dist=dist, linewidth=linewidth,
                                                                coloradd=color)
  return h

def draw_spline(env, spline, num=100, linewidth=1.0, colors=(0,0,1),
                                                      start=0, stop=1):
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
  linestrip = []
  for u in np.linspace(start, stop, num=num):
    linestrip.append(spline(u))
  h = env.drawlinestrip(np.array(linestrip), linewidth=linewidth,
                                                        colors=np.array(colors))
  return h
