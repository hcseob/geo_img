"""Create a representation of an image using circles.

Module usage:
import geo_img
geo_img.create('my_image.jpg', 'my_image.svg')

Command line:
python geo_img.py my_image.jpg
"""

import math
import ntpath
import numpy as np
from PIL import Image
import sys
import time

class Circle(object):
    """A circle object with a method to check for overlap."""
    def __init__(self, x, y, radius, rgb=None):
        self.x = x
        self.y = y
        self.r = radius
        # self.xc = x + radius
        # self.yc = y + radius
        self.rgb = [255, 255, 255]
        if rgb is not None:
            self.rgb = rgb

    @property
    def xc(self):
        """Circle center in x direction."""
        return self.x + self.r

    @property
    def yc(self):
        """Circle center in y direction."""
        return self.y + self.r

    def overlaps(self, c):
        """Checks for overlap between two circles."""
        if (self.xc - c.xc)**2 + (self.yc - c.yc)**2 < (self.r + c.r)**2:
            return True
        return False


class Circles(object):
    """A object of circle objects.

    Holds a dictionary which partitions circles based on radius for efficient
    collision detection. The overlaps() method checks all the circles in +/-
    one partion in each direction to check for collision. It is only safe for
    adding circles with a radius less or equal to the current minimum radius.
    The add() method calls overlaps() to check for collision before adding.
    """
    def __init__(self, dims):
        self.dims = dims
        self.circle_partitions = {}

    def __iter__(self):
        for partition_radius in self.circle_partitions:
            for col in self.circle_partitions[partition_radius]:
                for box in col:
                    for circle in box:
                        yield circle

    def __len__(self):
        len_sum = 0
        for partition_radius in self.circle_partitions:
            for col in self.circle_partitions[partition_radius]:
                for box in col:
                    len_sum += len(box)
        return len_sum

    def add(self, circle):
        """Add circle to appropriate partition if there is no overlap."""
        if not self.overlaps(circle):
            x_box, y_box = self.box_coords(circle)
            if circle.r not in self.circle_partitions:
                diameter = 2 * circle.r
                x_len = self.x_len(diameter)
                y_len = self.y_len(diameter)
                self.circle_partitions[circle.r] = [[[] for y in range(y_len)]
                                                    for x in range(x_len)]
            self.circle_partitions[circle.r][x_box][y_box].append(circle)

    def x_len(self, diameter):
        """Return the total number of partition boxes in the x direction."""
        return int(math.ceil(self.dims[0] * 1.0 / diameter))

    def y_len(self, diameter):
        """Return the total number of partition boxes in the y direction."""
        return int(math.ceil(self.dims[1] * 1.0 / diameter))

    def overlaps(self, circle_candidate):
        """Check if circle_candidate overlaps any other circle.

        This method is only safe for circle_candidates that have radius less
        than or equal to the minimum radius already added.
        """
        for partition_radius in self.circle_partitions:
            if self.overlaps_partition(circle_candidate, partition_radius):
                return True
        return False

    def overlaps_partition(self, circle_candidate, partition_radius):
        """Check if circle_candidate overlaps any circle in the partition."""
        diameter = 2*partition_radius
        x_box0, y_box0 = self.box_coords(circle_candidate, diameter)
        x_min = max(x_box0 - 1, 0)
        x_max = min(x_box0 + 2, self.x_len(diameter))
        y_min = max(y_box0 - 1, 0)
        y_max = min(y_box0 + 2, self.y_len(diameter))
        for x_box in range(x_min, x_max):
            for y_box in range(y_min, y_max):
                circle_partitions = self.circle_partitions[partition_radius]
                for circle in circle_partitions[x_box][y_box]:
                    if circle_candidate.overlaps(circle):
                        return True
        return False

    @staticmethod
    def box_coords(circle, diameter=None):
        """Return the coordinates of the box contained the circle corner."""
        if diameter is None:
            diameter = 2 * circle.r
        x_box = int(math.floor(circle.x / diameter))
        y_box = int(math.floor(circle.y / diameter))
        return (x_box, y_box)


def mu_m_v(mu_in, m_in, sq_len):
    """Calculate mean, moment, and variance of 2x larger regions."""
    sq_os = sq_len/2
    j_len, k_len, c_len = mu_in.shape
    sz = (j_len-sq_os, k_len-sq_os, c_len)
    mu = np.zeros(sz)
    m = np.zeros(sz)
    v = np.zeros((j_len-sq_os, k_len-sq_os))
    for j in range(j_len-sq_os):
        for k in range(k_len-sq_os):
            for c in range(c_len):
                mu[j, k, c] = 0.25*(mu_in[j, k, c] +
                                    mu_in[j+sq_os, k, c] +
                                    mu_in[j, k+sq_os, c] +
                                    mu_in[j+sq_os, k+sq_os, c])
                m[j, k, c] = 0.25*(m_in[j, k, c] +
                                   m_in[j+sq_os, k, c] +
                                   m_in[j, k+sq_os, c] +
                                   m_in[j+sq_os, k+sq_os, c])
            v[j, k] = np.sum(m[j, k, :] - mu[j, k, :]**2)
    return mu, m, v

def mu_m_v_list(im_arr):
    """Calculate mean, moment, and variance for region sizes."""
    mu_old = np.array(im_arr, dtype=np.uint32)
    m_old = mu_old**2
    sq_len = 1
    mu_list = [mu_old]
    m_list = [m_old]
    v_list = [m_old[:, :, 0] - mu_old[:, :, 0]**2 +
              m_old[:, :, 1] - mu_old[:, :, 1]**2 +
              m_old[:, :, 2] - mu_old[:, :, 2]**2]
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
    """Resize image."""
    scale = max_side/max(im.size)
    s = (int(im.size[0]*scale), int(im.size[1]*scale))
    return im.resize(s)

def find_circles(im_arr):
    """Find circles in im_arr where the region variance is low."""
    tstart = time.time()
    mu_list, _, v_list = mu_m_v_list(im_arr)
    num_sizes = 6
    radius = 2**(num_sizes-2)
    circles = Circles(im_arr.shape[:-1])
    v_max = 1000
    for i in np.arange(num_sizes-1, -1, -1):
        if i == 1:
            v_max = 1e9
        mu = mu_list[i]
        v = v_list[i]
        j_len, k_len = v.shape
        for j in range(j_len):
            for k in range(k_len):
                if v[j, k] < v_max:
                    circle = Circle(j, k, radius, mu[j, k, :])
                    circles.add(circle)
        radius = radius/2.0
    tend = time.time() - tstart
    print "Found circles %i in %i seconds." % (len(circles), tend)
    return circles

def draw_svg(circles, img_dims, path):
    """Create an svg image from circles object."""
    scale = 1
    height = img_dims[0] * scale
    width = img_dims[1] * scale
    with open(path, 'w') as fout:
        fout.write(('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
                    'style="stroke-width: 0px; background-color: black;" '
                    'viewBox="0 0 %i %i">\n') % (width, height))
        for circle in circles:
            radius = circle.r*scale
            cy = circle.x*scale + radius
            cx = circle.y*scale + radius
            r = circle.rgb[0]
            g = circle.rgb[1]
            b = circle.rgb[2]
            fout.write(('<circle cx="%i" cy="%i" r="%i" style="fill:rgb'
                        '(%i,%i,%i)" />\n') % (cx, cy, radius, r, g, b))
        fout.write('</svg>')

def create(path_in, path_out=None):
    """Create a geo_img from image at path path_in.

    The output path defaults to image path with svg extension.
    """
    if path_out is None:
        path_out = path_in.rsplit('.', 1)[0] + '.svg'
    im = Image.open(path_in)
    im = resize_im(im, 200.0)
    im_arr = np.array(im)
    print "Finding circles in image %s..." % ntpath.basename(path_in)
    circles = find_circles(im_arr)
    draw_svg(circles, im_arr.shape, path_out)

def main():
    """Handle command line arguments to geo_img."""
    if len(sys.argv) > 2:
        create(sys.argv[1], sys.argv[2])
    else:
        create(sys.argv[1])

if __name__ == '__main__':
    main()
