import math
import numpy as np
import os
from PIL import Image
import sys
import time

#TODO: py_lint
#TODO: Comment everything
#TODO: one function that takes input and output location
#TODO: underscore functions 
#TODO: main() handles command line but can also import geo_img
#TODO: default output same location in svg
#TODO: add to github, license, readme, clean up everything
#TODO: more efficient circle lookup

class Circle(object):
    def __init__(self, x, y, radius, rgb=[255, 255, 255]):
        self.x = x
        self.y = y
        self.r = radius
        self.xc = x + radius
        self.yc = y + radius
        self.rgb = rgb

    def overlaps(self, circle):
        if (self.xc - circle.xc)**2 + (self.yc - circle.yc)**2 < (self.r + circle.r )**2:
            return True
        return False


class Circles(object):
    def __init__(self, max_radius, dims):
        self.max_diameter = 2 * max_radius
        self.x_len = int(math.ceil(dims[0] * 1.0 / self.max_diameter))
        self.y_len = int(math.ceil(dims[1] * 1.0 / self.max_diameter))
        self.circle_partitions = [[[] for y in range(self.y_len)] for x in range(self.x_len)]

    def __iter__(self):
        for col in self.circle_partitions:
            for box in col:
                for circle in box:
                    yield circle

    def add(self, circle):
        if not self.overlaps(circle):
            x_box, y_box = self.box_coords(circle)
            self.circle_partitions[x_box][y_box].append(circle)

    def box_coords(self, circle):
        return (int(math.floor(circle.x / self.max_diameter)), int(math.floor(circle.y / self.max_diameter)))

    def overlaps(self, circle_candidate):
        x_box0, y_box0 = self.box_coords(circle_candidate)
        x_min = max(x_box0 - 1, 0)
        x_max = min(x_box0 + 2, self.x_len)
        y_min = max(y_box0 - 1, 0)
        y_max = min(y_box0 + 2, self.y_len)
        for x_box in range(x_min, x_max):
            for y_box in range(y_min, y_max):
                for circle in self.circle_partitions[x_box][y_box]:
                    if circle_candidate.overlaps(circle):
                        return True
        return False


def mu_m_v(mu_in, m_in, sq_len):
    sq_os = sq_len/2
    j_len, k_len, c_len = mu_in.shape
    sz = (j_len-sq_os, k_len-sq_os, c_len)
    mu = np.zeros(sz)
    m = np.zeros(sz)
    v = np.zeros((j_len-sq_os, k_len-sq_os))
    for j in range(j_len-sq_os):
        for k in range(k_len-sq_os):
            for c in range(c_len):
                mu[j, k, c] = 0.25*(mu_in[j, k, c] + mu_in[j+sq_os, k, c] +
                                    mu_in[j, k+sq_os, c] + mu_in[j+sq_os, k+sq_os, c])
                m[j, k, c] = 0.25*(m_in[j, k, c] + m_in[j+sq_os, k, c] +
                                   m_in[j, k+sq_os, c] + m_in[j+sq_os, k+sq_os, c])
            v[j, k] = np.sum(m[j, k, :] - mu[j, k, :]**2)
    return mu, m, v

def mu_m_v_list(im_arr):
    mu_old = np.array(im_arr, dtype=np.uint32)
    m_old = mu_old**2
    sq_len = 1
    mu_list = [mu_old]
    m_list = [m_old]
    v_list = [m_old[:,:,0] - mu_old[:,:,0]**2 + 
              m_old[:,:,1] - mu_old[:,:,1]**2 + 
              m_old[:,:,2] - mu_old[:,:,2]**2]
    for k in range(6):
        sq_len *= 2
        mu, m, v = mu_m_v(mu_old, m_old, sq_len)
        mu_list.append(mu)
        m_list.append(m)
        v_list.append(v)
        mu_old = mu
        m_old = m
    return mu_list, m_list, v_list

def resize_im(im, max_side):
    scale = max_side/max(im.size)
    s = (int(im.size[0]*scale), int(im.size[1]*scale))
    return im.resize(s)

def find_circles(im_arr):
    tstart = time.time()
    mu_list, m_list, v_list = mu_m_v_list(im_arr)
    num_sizes = 6
    radius = 2**(num_sizes-2)
    circles = Circles(radius, im_arr.shape[:-1])
    v_max = 1000
    for i in np.arange(num_sizes-1,-1,-1):
        if i == 1: v_max = 1e9
        mu = mu_list[i]
        v = v_list[i]
        j_len, k_len = v.shape
        for j in range(j_len):
            for k in range(k_len):
                if v[j, k] < v_max:
                    circle = Circle(j, k, radius, mu[j, k, :])
                    circles.add(circle) # Circles object checks for overlapping before adding
        radius = radius/2.0
    print "Found circles in %i seconds." % (time.time() - tstart)
    return circles

def draw_svg(circles, img_dims, path):
    scale = 1
    height = img_dims[0] * scale
    width = img_dims[1] * scale
    with open(path, 'w') as fout:
        fout.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" style="stroke-width: 0px; background-color: black;" viewBox="0 0 %i %i">\n' % (width, height))
        for circle in circles:
            radius = circle.r*scale
            cy = circle.x*scale + radius
            cx = circle.y*scale + radius
            r = circle.rgb[0]
            g = circle.rgb[1]
            b = circle.rgb[2]
            fout.write('<circle cx="%i" cy="%i" r="%i" style="fill:rgb(%i,%i,%i)" />\n' % (cx, cy, radius, r, g, b))
        fout.write('</svg>')

def main():
    im = Image.open(sys.argv[1])
    im = resize_im(im, 200.0)
    im_arr = np.array(im)
    print "Finding circles in image..."
    circles = find_circles(im_arr)
    draw_svg(circles, im_arr.shape, sys.argv[2])


if __name__ == '__main__':
    main()