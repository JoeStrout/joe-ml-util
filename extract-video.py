"""
This script extracts a Z-stack of images from a cloud volume,
and exports them as a video.
"""

from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.geometry import Vec3D
import numpy as np
import cv2

cvpath = "https://storage.googleapis.com/dacey-human-retina-001-alignment/img/v1"
sample_resolution = Vec3D(10, 10, 50)

cvl = build_cv_layer(path=cvpath)

xrange = [12288, 12288 + 512]
yrange = [14336, 14336 + 512]
zrange = [1400, 1500]

# Video parameters
output_filename = "output_video.mp4"
fps = 10  # frames per second
frame_size = (xrange[1] - xrange[0], yrange[1] - yrange[0])  # Width x Height

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=False)

for z in range(zrange[0], zrange[1]):
    data = cvl[sample_resolution, xrange[0]:xrange[1], yrange[0]:yrange[1], z:z+1][0, :, :, 0]
    data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the data to 8-bit
    data_uint8 = data_normalized.astype(np.uint8)  # Convert to 8-bit image
    print(f'Writing frame for z={z}')
    out.write(data_uint8)

# Release the video writer
out.release()
print(f"Video saved as {output_filename}")
