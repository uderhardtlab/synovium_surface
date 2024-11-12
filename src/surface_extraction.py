import matplotlib.pyplot as plt
import numpy as np
from cupyx.scipy.ndimage import median_filter, gaussian_filter, label, morphological_gradient
from skimage.filters import sobel
from skimage.segmentation import expand_labels
from skimage import img_as_float
from scipy.ndimage import binary_closing
from scipy.interpolate import griddata
import json
import tifffile as tf
import seaborn as sns
import sys
import cupy as cp

def smooth_image_in_xyz(image, size):
    image = cp.array(image)
    smoothed = median_filter(image.astype(int), size=(size, size, size))
    del image
    return smoothed


def get_biggest_connected_component(binary_image):
    labelled, num_labels = label(cp.array(binary_image.astype(bool)), structure=cp.ones((3, 3, 3)))
    del binary_image
    v, c = cp.unique(labelled, return_counts=True)
    v_max = v[1:][cp.argmax(c[1:])]
    biggest_connected_volume = labelled == v_max
    del labelled
    return biggest_connected_volume


def get_morph_gradient(biggest_connected_volume):
    kernel = cp.zeros((3, 3, 3))
    kernel[1,1,:] = 1
    kernel[:,1,1] = 1
    kernel[1,:,1] = 1
    morph_grad = morphological_gradient(biggest_connected_volume.astype(int), structure=kernel)
    return morph_grad


def get_layer_in_3d(top_layer, image_shape):
    if isinstance(top_layer, np.ndarray):
        top_layer = cp.array(top_layer)
    top_layer = cp.nan_to_num(top_layer, 0)
    x, y = cp.indices(top_layer.shape) 
    top_layer_3d = cp.zeros(image_shape)
    top_layer = cp.clip(top_layer, 0, image_shape[0] - 1).astype(int)
    top_layer_3d[top_layer, x, y] = 1
    top_layer_3d[0, :, :] = 0
    return cp.asarray(top_layer_3d)


def get_top_layer(morph_grad, img_shape, bottom=False):
    if not bottom:
        top_layer = cp.argmax(morph_grad, axis=0)
        top_layer_3d = get_layer_in_3d(top_layer, img_shape)
        del top_layer
        return top_layer_3d
    else:
        bottom_layer = cp.argmax(morph_grad[::-1], axis=0)
        bottom_layer_3d = get_layer_in_3d(bottom_layer, img_shape)[::-1]
        del bottom_layer
        return bottom_layer_3d


def expand_labels_on_morph(morph_grad, img_shape):
    labels = get_top_layer(morph_grad, img_shape, bottom=False) + 2 * get_top_layer(morph_grad, img_shape, bottom=True)
    expanded_labels = expand_labels(cp.asnumpy(labels), distance=20, spacing=(10, 1, 1)) # this corresponds to expanding by 2 in z dimension and by 20 in x and y dimensions, so it should definetely cover all pertrusions
    expanded_labels_on_morph = cp.array(expanded_labels == 1).astype(int) * (morph_grad == np.max(morph_grad))
    return expanded_labels_on_morph


def process_patch(image, binarization_threshold=6):
    smoothed_image = smooth_image_in_xyz(image, size=5)
    binary_image = smoothed_image > binarization_threshold
    del smoothed_image

    biggest_connected_volume = get_biggest_connected_component(binary_image)
    shape_3d = biggest_connected_volume.shape
    morph_grad = get_morph_gradient(biggest_connected_volume)
    surface = expand_labels_on_morph(morph_grad, shape_3d)
    del morph_grad
    del biggest_connected_volume
    return cp.asnumpy(surface)
    

"""
old

def get_top_layer(binary_image):
    gradient_z = diff(binary_image, axis=0)  # estimates gradient along z
    top_layer_threshold = 0.1
    top_layer_mask = gradient_z > top_layer_threshold
    top_layer = argmax(top_layer_mask, axis=0) # Array of z-coordinates for the top layer
    top_layer[where(top_layer) == 0] = np.nan
    return top_layer



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
"""