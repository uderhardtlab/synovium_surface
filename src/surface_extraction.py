import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filters import sobel
from skimage import img_as_float
from scipy.ndimage import binary_closing
from scipy.interpolate import griddata
import json
import tifffile as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, label
from tqdm import tqdm


def smooth_image_in_xy(image):
    return median_filter(image.astype(np.uint8), size=(1, 3, 3))


def get_top_layer(binary_image):
    gradient_z = np.diff(binary_image, axis=0)  # estimates gradient along z
    top_layer_threshold = 0.1
    top_layer_mask = gradient_z > top_layer_threshold
    top_layer = np.argmax(top_layer_mask, axis=0)  # Array of z-coordinates for the top layer
    return top_layer


def get_layer_in_3d(top_layer, image_shape):
    x, y = np.indices(top_layer.shape) 
    top_layer_3d = np.zeros(image_shape)
    top_layer = np.clip(top_layer, 0, image_shape[0] - 1).astype(int)
    top_layer_3d[top_layer, x, y] = 1
    top_layer_3d[0, :, :] = 0
    return top_layer_3d


def interpolate_top_layer(top_layer, image_shape):
    x, y = np.indices(top_layer.shape)  # Create x, y grid
    valid_points = top_layer > 0  # Points where a valid top layer was detected
    known_points = (x[valid_points], y[valid_points])
    known_values = top_layer[valid_points]

    top_layer_interp = griddata(known_points, known_values, (x, y), method='cubic', fill_value=0)
    top_layer_interp = np.clip(top_layer_interp, 0, image_shape[0] - 1).astype(int)
    return top_layer_interp


def smooth_layer_in_xy(top_layer_interp, sigma=2):
    return gaussian_filter(top_layer_interp, sigma=sigma)


def piecewiese_smooth_layer_in_xy(top_layer, sigma=2):
    valid_mask = np.isfinite(top_layer).astype(float)
    image_filled = np.nan_to_num(top_layer, nan=0)
    smoothed_image = gaussian_filter(image_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(valid_mask, sigma=sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed_top_layer = smoothed_image / smoothed_mask
        smoothed_top_layer[smoothed_mask == 0] = np.nan  

    smoothed_top_layer = np.round(smoothed_top_layer).astype(int)
    return smoothed_top_layer