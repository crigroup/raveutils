#! /usr/bin/env python
import unittest
import numpy as np
import openravepy as orpy
# Tested package
import raveutils as ru


class Test_body(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    np.set_printoptions(precision=6, suppress=True)
    env = orpy.Environment()
    makita = env.ReadKinBodyXMLFile('objects/makita.kinbody.xml')
    env.AddKinBody(makita)
    cls.env = env
    print('') # dummy line

  @classmethod
  def tearDownClass(cls):
    cls.env.Reset()
    cls.env.Destroy()

  def test_enable_body(self):
    env = self.env
    makita = env.GetBodies()[0]
    # Enable
    ru.body.enable_body(makita, True)
    link_status = [l.IsEnabled() for l in makita.GetLinks()]
    self.assertTrue(all(link_status))
    # Disable
    ru.body.enable_body(makita, False)
    link_status = [l.IsEnabled() for l in makita.GetLinks()]
    self.assertFalse(any(link_status))

  def test_get_bounding_box_corners(self):
    # TODO: Write a proper test for this function
    env = self.env
    makita = env.GetBodies()[0]
    corners = ru.body.get_bounding_box_corners(makita)
    self.assertEqual(len(corners), 8)
    # Transform given
    makita = env.GetBodies()[0]
    transform = makita.GetTransform()
    corners = ru.body.get_bounding_box_corners(makita, transform)
    self.assertEqual(len(corners), 8)

  def test_set_body_color(self):
    env = self.env
    makita = env.GetBodies()[0]
    for _ in range(10):
      diffuse = np.random.sample(3)
      ambient = np.random.sample(3)
      ru.body.set_body_color(makita, diffuse, ambient)
      dcolors = [g.GetDiffuseColor() for l in makita.GetLinks()
                                                    for g in l.GetGeometries()]
      dcolor = np.unique(dcolors, axis=0).reshape(diffuse.shape)
      np.testing.assert_allclose(diffuse, dcolor)
      acolors = [g.GetAmbientColor() for l in makita.GetLinks()
                                                    for g in l.GetGeometries()]
      acolor = np.unique(acolors, axis=0).reshape(ambient.shape)
      np.testing.assert_allclose(ambient, acolor)

  def test_set_body_transparency(self):
    env = self.env
    makita = env.GetBodies()[0]
    # Modify only a given link
    ru.body.set_body_transparency(makita, 0.5, links=['body'])
    for _ in range(10):
      expected = np.random.sample(1)
      ru.body.set_body_transparency(makita, expected)
      values = [g.GetTransparency() for l in makita.GetLinks()
                                                    for g in l.GetGeometries()]
      transparency = float(np.unique(values))
      np.testing.assert_allclose(transparency, expected)
