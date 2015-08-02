import re
import sys
sys.path.append("../")
from geo_img import *

img_dims = (200, 139)
circles = Circles(img_dims)

svgs = []
svgs.append('<circle xmlns="http://www.w3.org/2000/svg" cx="60" cy="95" r="4" style="fill:rgb(255,255,255)"/>')
svgs.append('<circle xmlns="http://www.w3.org/2000/svg" cx="52" cy="91" r="8" style="fill:rgb(255,255,255)"/>')
for svg in svgs:
	match = re.search(r'cx="(.+?)".+cy="(.+?)".+r="(.+?)"', svg)
	r = int(match.group(3))
	x = int(match.group(2)) - r
	y = int(match.group(1)) - r
	c = Circle(x, y, r)
	circles.add(c)

draw_svg(circles, img_dims, './no_overlap_debug.svg')
