import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from skimage.measure import regionprops, label
import tifffile as tf
from tqdm import tqdm

cuda_device = True
import scipy as sc

if cuda_device:
    from cupyx.scipy.ndimage import median_filter
    from cupy import diff, argmax, where, nan_to_num, indices, zeros, clip, nanmedian, isnan
    import cupy as cp
    
    
else:
    from scipy.ndimage import median_filter
    from numpy import diff, argmax, where, nan_to_num, indices, zeros, clip, nanmedian, isnan

def smooth_image_in_xyz(image):
    if cuda_device:
        if isinstance(image, np.ndarray):
            image = cp.array(image)
    return median_filter(image.astype(int), size=(3, 3, 3))


def get_top_layer(binary_image):
    gradient_z = diff(binary_image, axis=0)  # estimates gradient along z
    top_layer_threshold = 0.1
    top_layer_mask = gradient_z > top_layer_threshold
    top_layer = argmax(top_layer_mask, axis=0) # Array of z-coordinates for the top layer
    top_layer[where(top_layer) == 0] = np.nan
    return top_layer


def get_layer_in_3d(top_layer, image_shape):
    if isinstance(top_layer, np.ndarray):
        top_layer = cp.array(top_layer)
    top_layer = nan_to_num(top_layer, 0)
    x, y = indices(top_layer.shape) 
    top_layer_3d = zeros(image_shape)
    top_layer = clip(top_layer, 0, image_shape[0] - 1).astype(int)
    top_layer_3d[top_layer, x, y] = 1
    top_layer_3d[0, :, :] = 0
    return cp.asarray(top_layer_3d)


def patch(top_layer, median_filtered_interp_top_layer):
    if cuda_device:
        if isinstance(top_layer, np.ndarray):
            top_layer = cp.array(top_layer)
        if isinstance(median_filtered_interp_top_layer, np.ndarray):
            median_filtered_interp_top_layer = cp.array(median_filtered_interp_top_layer)
    return where(top_layer > 0, median_filtered_interp_top_layer, top_layer)


def interpolate_top_layer(top_layer, z_max, method):
    if not isinstance(top_layer, np.ndarray):
        top_layer = cp.asnumpy(top_layer)
    x, y = np.indices(top_layer.shape)  # Create x, y grid
    valid_points = top_layer > 0  # Points where a valid top layer was detected
    known_points = (x[valid_points], y[valid_points])
    known_values = top_layer[valid_points]
    print("interpolating")
    top_layer_interp = sc.interpolate.griddata(known_points, known_values, (x, y), method=method, fill_value=0)
    print("done")
    top_layer_interp = np.clip(top_layer_interp, 0, z_max - 1).astype(int)
    return top_layer_interp



def median_filtering_in_xy(top_layer, size):
    if isinstance(top_layer, np.ndarray):
        top_layer = cp.array(top_layer)
    return median_filter(top_layer, size=size)
