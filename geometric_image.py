import numpy as np
from PIL import Image


class Circle(object):
    def __init__(self, x, y, radius, rgb=[0, 0, 0]):
        self.x = x
        self.y = y
        self.radius = radius
        self.rgb = rgb

    def distance_to(self, circle):
        center_distance = (self.x - circle.x)**2.0 + (self.y - circle.y)**2.0
        return center_distance - self.radius - circle.radius


def convert_to_circles(path, radius_array, step_size, var_limit, scale):
    img = Image.open(path)
    img_array = np.array(img)
    dims = img_array.shape
    scaled_dims = [dims[0]*scale, dims[1]*scale, dims[2]]
    out_img_array = np.zeros(scaled_dims)
    # TODO initialize to mean color
    x_len = len(img_array[:, 0, 0])
    y_len = len(img_array[0, :, 0])
    num_outs = 0
    circle_list = []  # list of circle objects
    for radius in radius_array:
        length = 2.0 * radius
        circle_array = make_circle_array((length, length))
        big_circle = make_circle_array((length * scale, length * scale))
        while True:
            found_circle = False
            x = 0
            while x < x_len - length:
                print x
                y = 0
                while y < y_len - length:
                    sub_img_array = get_sub_img_array(img_array, x, y, length)
                    var_tot = tot_var_in_circle(sub_img_array, circle_array)
                    if var_tot < var_limit:
                        opt_rgb = get_mean_rgb(sub_img_array, circle_array)
                        if np.mean(opt_rgb)/3 > 1:
                            print var_tot, var_limit
                            circle_list.append(Circle(x, y, radius, opt_rgb))
                            found_circle = True
                            opt_sub_img_array = np.copy(sub_img_array)
                            opt_circle = color_circle(circle_array, opt_rgb)
                            img_array[x:x+length, y:y+length, :] -= opt_circle
                            img_array[img_array < 0] = 0
                            # create output scaled up by scale
                            opt_big_circle = color_circle(big_circle, opt_rgb)
                            out_img_array[x*scale:x*scale+length*scale,
                                          y*scale:y*scale+length*scale,
                                          :] += opt_big_circle
                    y = y + step_size
                x = x + step_size
            if not found_circle:
                break
            else:
                # save images for debug purposes
                save_image(opt_sub_img_array, './debug/opt_sub.png')
                save_image(opt_circle, './debug/opt_circle.png')
                save_image(img_array, './debug/image_residual.png')
                save_image(out_img_array, './debug/out/out%03d.png' % num_outs)
                num_outs += 1


def get_sub_img_array(img_array, x_start, y_start, length):
    return np.copy(img_array[x_start:x_start+length, y_start:y_start+length, :])


def get_mean_rgb(img_array, circle_array):
    return [mean_in_circle(img_array[:, :, 0], circle_array),
            mean_in_circle(img_array[:, :, 1], circle_array),
            mean_in_circle(img_array[:, :, 2], circle_array)]


def get_image_from_array(img_array):
    return Image.fromarray((img_array).astype(np.uint8))


def save_image(img_array, path):
    img = get_image_from_array(img_array)
    img.save(path)


def color_circle(circle_array, rgb_list):
    dimensions = circle_array.shape
    x_len = dimensions[0]
    y_len = dimensions[1]
    color_circle = np.empty((x_len, y_len, 3))
    for x in range(x_len):
        for y in range(y_len):
            if ~np.isnan(circle_array[x, y]):
                color_circle[x, y, 0] = rgb_list[0]
                color_circle[x, y, 1] = rgb_list[1]
                color_circle[x, y, 2] = rgb_list[2]
            else:  # outside the circle is black
                color_circle[x, y, 0] = 0
                color_circle[x, y, 1] = 0
                color_circle[x, y, 2] = 0
    return color_circle


def mean_in_circle(img_array_X, circle_array):
    error_array = img_array_X * circle_array
    error_array = error_array[~np.isnan(error_array)]  # Discard nan terms.
    return np.mean(error_array)


def var_in_circle(sub_img_array_X, circle_array):
    error_array = sub_img_array_X * circle_array
    error_array = error_array[~np.isnan(error_array)]  # Discard nan terms.
    return np.var(error_array)


def tot_var_in_circle(sub_img_array, circle_array):
    var_R = var_in_circle(np.squeeze(sub_img_array[:, :, 0]), circle_array)
    var_G = var_in_circle(np.squeeze(sub_img_array[:, :, 1]), circle_array)
    var_B = var_in_circle(np.squeeze(sub_img_array[:, :, 2]), circle_array)
    return var_R + var_G + var_B


def make_circle_array(dimensions):
    x_len = dimensions[0]
    y_len = dimensions[1]
    x0 = (x_len - 1.0)/2.0
    y0 = (y_len - 1.0)/2.0
    x = np.arange(0, x_len - 1)
    y = np.arange(0, y_len - 1)
    yv, xv = np.meshgrid(y, x)
    circle_array = np.empty((x_len, y_len))
    circle_array.fill(np.nan)
    radius = x0 if x_len < y_len else y0
    circle_array[radius**2.0 > (xv - x0)**2.0 + (yv - y0)**2.0] = 1
    return circle_array


def main():
    path_to_file = './hawaii_tent.jpg'
    radius_array = [60, 40, 20, 10, 5]
    step_size = 1
    var_limit = 600
    scale = 4
    convert_to_circles(path_to_file, radius_array, step_size, var_limit, scale)


if __name__ == '__main__':
    main()
