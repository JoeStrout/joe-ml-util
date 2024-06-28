# Python script to extract images from a CloudVolume dataset.

from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.geometry import Vec3D
import cc3d
import numpy as np
import zetta_utils.tensor_ops.convert as convert
from PIL import Image
import matplotlib.pyplot as plt
import time

# Enable interactive mode
plt.ion()
fig, ax = plt.subplots()
image_display = None

def show_slice(image_slice):
    """
    Show the given image slice (a 2D tensor or np array)
    in a matplotlib window.  (Without blocking.)
    """
    global image_display
    if image_display is None:
        # Initial plot setup
        image_display = ax.imshow(image_slice, cmap='gray', vmin=np.min(image_slice), vmax=np.max(image_slice))        #plt.colorbar(image_display, ax=ax)
    else:
        # Update the plot with new data
        image_display.set_data(image_slice)
        image_display.set_clim(vmin=np.min(image_slice), vmax=np.max(image_slice))
    fig.canvas.draw()
    fig.canvas.flush_events()

def save_slice(image_slice, path='image.png'):
    """
    Write the given image slice (a 2D tensor or np array)
    out to disk in PNG format.
    """
    if image_slice.dtype == np.uint8:
        image = image_slice
    else:
        image = (image_slice * 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def export_XY(cvl, resolution, x, y, z, path=None, width=256, height=256, show=True):
    """
    Extract an XY slice from the given cloud volume, centered
    on x,y,z and of the given width and height.  Save to disk at
    the given path (unless that is None).

    If show is True, also show it in a MatPlotLib window (without
    blocking).
    """
    global wtf
    data = cvl[resolution, 
               x - width/2 : x + width/2,
               y - height/2: y + height/2,
               z : z+1]
    wtf = data
    data = data[0, :, :, 0]
    if show:
        show_slice(data)
    if path != None:
        save_slice(data, path)


if __name__ == "__main__":
    #cvpath = "https://storage.googleapis.com/dkronauer-ant-001-synapse/test/inference20240329023545"
    #sample_resolution = Vec3D(16, 16, 42)

    cvpath = "https://storage.googleapis.com/dacey-human-retina-001-alignment/img/v1"
    sample_resolution = Vec3D(10, 10, 50)

    cvl = build_cv_layer(path=cvpath)

    # display an image from the volume
    # (add the `path` parameter to save to a PNG file)
    export_XY(cvl, sample_resolution, 12986, 12820, 1577)
    while True:
        fig.canvas.flush_events()
        time.sleep(0.01)

