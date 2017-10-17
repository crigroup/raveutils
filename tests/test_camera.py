#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
from sensor_msgs.msg import CameraInfo
# Tested package
import raveutils as ru


class Test_camera(unittest.TestCase):
  def test_CameraFOV(self):
    msg = CameraInfo()
    msg.height = 1024
    msg.width = 1280
    msg.distortion_model = 'plumb_bob'
    msg.D = [-0.5506515572367885, 0.16918149333674903, -0.0005494252446900035,
                                    -0.003574460971943457, 0.08824797068343779]
    msg.K = [1547.3611792786492, 0.0, 645.7946620597459, 0.0,
                            1546.5965535476455, 512.489834878375, 0.0, 0.0, 1.0]
    msg.R = [0.9976063902119301, 0.0008462845042432227, 0.06914314145929477,
            -0.0007608300534628908, 0.9999989139538485, -0.0012622316559149822,
                -0.06914413457374326, 0.0012066041858555048, 0.997605950644034]
    msg.P = [1445.628365834274, 0.0, 521.7993656184378, 0.0, 0.0,
                  1445.628365834274, 514.7422812912976, 0.0, 0.0, 0.0, 1.0, 0.0]
    camfov = ru.camera.CameraFOV(msg, maxdist=2.)
    corners = camfov.get_corners()
    vertices, faces = camfov.get_trimesh()
    self.assertTrue(camfov.contains(corners))
    points_inside = [camfov.random_point_inside() for _ in range(100)]
    self.assertTrue(camfov.contains(points_inside))
