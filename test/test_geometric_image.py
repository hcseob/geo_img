import sys
sys.path.append("../")
from geometric_image import *
import unittest
import random

class TestGI(unittest.TestCase):

    def test_circles(self):
    	for k in range(100):
			r1 = random.randint(1,10)
			r2 = random.randint(1,10)
			x1 = random.randint(-50, 50)
			x2 = x1 + 2*r1 + 1e-12
			y1 = -r1
			y2 = -r2
			c1 = Circle(x1, y1, r1)
			c2 = Circle(x2, y2, r2)
			self.assertFalse(c1.overlaps(c2))
			self.assertFalse(c2.overlaps(c1))
			c2.x -= 2e-12
			self.assertTrue(c1.overlaps(c2))
			self.assertTrue(c2.overlaps(c1))

if __name__ == '__main__':
	unittest.main()