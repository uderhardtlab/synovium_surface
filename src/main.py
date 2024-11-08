import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import sobel
from skimage import img_as_float
from scipy.ndimage import binary_closing
from scipy.interpolate import griddata
import json
import tifffile as tf
import seaborn as sns
import sys
sys.path.append("../../network_extraction/src/")
from real_data import write_tiff
sys.path.append("../src/")
from surface_extraction import *

configs_path = "../configs.json"
with open(configs_path, "r") as f:
    configs = json.load(f)

image = tf.imread(configs["CD44"])
print("read image")

smoothed_image = smooth_image_in_xy(image)
binary_image = image > 6 # TODO thresholding?
top_layer = get_top_layer(binary_image)
top_layer_3d = get_layer_in_3d(top_layer, image.shape)
print("saving top layer")
write_tiff(configs, "column-wise-first-gradient.tif", top_layer_3d)

piece_wise_smoothed_top_layer = piecewiese_smooth_layer_in_xy(top_layer, sigma=5)
piece_wise_smoothed_top_layer_3d = get_layer_in_3d(piece_wise_smoothed_top_layer, image.shape)
print("saving piece-wise smoothed top layer")
write_tiff(configs, "piecewise-smoothed-column-wise-first-gradient.tif", piece_wise_smoothed_top_layer_3d)

top_layer_interp = interpolate_top_layer(top_layer, image.shape)
interp_top_layer_3d = get_layer_in_3d(top_layer_interp, image.shape)
print("saving interpolated top layer")
write_tiff(configs, "interpolated_surface.tif", interp_top_layer_3d)

smoothed_top_layer = smooth_layer_in_xy(top_layer_interp, sigma=5)
smoothed_top_layer_3d = get_layer_in_3d(smoothed_top_layer, image.shape)
print("saving smoothed interpolated top layer")
write_tiff(configs, "smoothed_interpolated_surface.tif", smoothed_top_layer_3d)