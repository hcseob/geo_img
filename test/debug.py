import sys
sys.path.append("../")
from geo_img import *

img_dims = (200, 139)
max_radius = 10
circles = Circles(max_radius, img_dims)

r1 = 9
x1 = 4
y1 = 37
c1 = Circle(x1, y1, r1)
circles.add(c1)

r2 = 9
x2 = 7
y2 = 51
c2 = Circle(x2, y2, r2)
circles.add(c2)
draw_svg(circles, img_dims, './no_overlap.svg')
