import sys
sys.path.append("../")
from geo_img import *
import unittest
import random

class TestGI(unittest.TestCase):

	def test_mu_m_v(self):
		im_arr = np.ones((400, 400, 3))*2
		mu_old = im_arr
		m_old = im_arr**2
		mu, m, v = mu_m_v(mu_old, m_old, 2)
		self.assertEqual(np.max(mu), 2)
		self.assertEqual(np.min(mu), 2)
		self.assertEqual(np.max(m), 4)
		self.assertEqual(np.min(m), 4)
		self.assertEqual(np.max(v), 0)
		self.assertEqual(np.min(v), 0)

	def test_mu_m_v_list(self):
		r = 9
		g = 11
		b = 13
		im_arr = np.ones((150, 150, 3))
		im_arr[:,:,0] = r*im_arr[:,:,0]
		im_arr[:,:,1] = g*im_arr[:,:,1]
		im_arr[:,:,2] = b*im_arr[:,:,2]
		mu_old = im_arr
		m_old = im_arr**2
		mu_list, m_list, v_list = mu_m_v_list(im_arr)
		for k in range(len(mu_list)):
			mu = mu_list[k]
			m = m_list[k]
			v = v_list[k]
			self.assertEqual(np.max(mu[:,:,0]), r)
			self.assertEqual(np.min(mu[:,:,0]), r)
			self.assertEqual(np.max(mu[:,:,1]), g)
			self.assertEqual(np.min(mu[:,:,1]), g)
			self.assertEqual(np.max(mu[:,:,2]), b)
			self.assertEqual(np.min(mu[:,:,2]), b)
			self.assertEqual(np.max(m[:,:,0]), r**2)
			self.assertEqual(np.min(m[:,:,0]), r**2)
			self.assertEqual(np.max(m[:,:,1]), g**2)
			self.assertEqual(np.min(m[:,:,1]), g**2)
			self.assertEqual(np.max(m[:,:,2]), b**2)
			self.assertEqual(np.min(m[:,:,2]), b**2)
			self.assertEqual(np.max(v[:,:]), 0)
			self.assertEqual(np.min(v[:,:]), 0)

	def test_circles(self):
		img_dims = (200, 139)
		max_radius = 10
		circles = Circles(max_radius, img_dims)
		for k in range(10000):
			r = random.randint(1, max_radius)
			x = random.randint(0, img_dims[0] - 2*r)
			y = random.randint(0, img_dims[1] - 2*r)
			circle = Circle(x, y, r)
			circles.add(circle)
		draw_svg(circles, img_dims, './no_overlap.svg')

	def test_debug(self):
		img_dims = (200, 139)
		max_radius = 10
		circles = Circles(max_radius, img_dims)
		
		r1 = 4
		x1 = 35
		y1 = 18
		c1 = Circle(x1, y1, r1)

		r2 = 9
		x2 = 41
		y2 = 15
		c2 = Circle(x2, y2, r2)


if __name__ == '__main__':
	unittest.main()