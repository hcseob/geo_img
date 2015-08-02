import os
with open('geo_img.html', 'w') as fout:
    fout.write("<!DOCTYPE html>\n<html>\n<body>\n")
    for fname in os.listdir('./inputs/'):
        if fname.endswith('.jpg'):
            name = os.path.splitext(fname)[0]
            fout.write('<img src="./outputs/%s.svg" style="width:49%%"></img>' % name)   
            fout.write('<img src="./inputs/%s.jpg" style="width:49%%"></img>' % name)    
    fout.write("</body>\n</html>")
