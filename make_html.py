import geo_img
import os

path = './inputs/'

def is_img(fname):
    return fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png')

with open('%sgeo_img.html' % path, 'w') as fout:
    fout.write("<!DOCTYPE html>\n<html>\n<body>\n")
    for fname in os.listdir(path):
        if is_img(fname):
            geo_img.create(path+fname)
            name = os.path.splitext(fname)[0]
            fout.write('<h1>%s</h1>' % fname)
            fout.write('<img src="%s.svg" style="width:50%%"></img>' % name)
            fout.write('<img src="%s" style="width:50%%"></img>' % fname)
    fout.write("</body>\n</html>")
