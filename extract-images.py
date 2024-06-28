# Python script to extract a whole bunch of 2D images from
# a Cloud volume.  Uses cloud_export.py to do the hard work.

from random import randint
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.geometry import Vec3D
from cloud_export import *
from time import sleep

cvpath = "https://storage.googleapis.com/dacey-human-retina-001-alignment/img/v1"
sample_resolution = Vec3D(10, 10, 50)

cvl = build_cv_layer(path=cvpath)

xrange = [5000, 22000]
yrange = [5000, 22000]
zrange = [24, 3000]

output_dir = "/home/joe/Documents/datasets/dacey-retina/"
export_count = 990

for i in range(0, export_count):
    x = randint(*xrange)
    y = randint(*yrange)
    z = randint(*zrange)
    export_XY(cvl, sample_resolution,
              x, y, z, 
              output_dir + f"{x}_{y}_{z}.png")
    fig.canvas.manager.set_window_title(f'{i} / {export_count}')
    fig.canvas.flush_events()
    sleep(0.01)
