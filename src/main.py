import numpy as np
import json
import tifffile as tf
import sys
sys.path.append("../../network_extraction/src/")
from real_data import write_tiff
sys.path.append("../src/")
from surface_extraction import *

# THIS SCRIPT REQUIRES A CUDA COMPATIBLE GPU
# CONSIDER SPECIFIYING CUDA_VISIBLE_DEVICES=0 

configs_path = "../configs.json"
with open(configs_path, "r") as f:
    configs = json.load(f)


    

image = tf.imread(configs["CD44"])
if "exp2" in configs["data"]:
    print("flipping image in z")
    image = image[::-1,:,:]

window_size = int(configs["window_size"]) # increasing this constant removes intensity detail of the binarization
C = int(configs["C"]) # increasing this constant removes spatial detail of the binarization
z_expansion = int(configs["z_expansion"]) # increasing this constant adds more signal to the top layer

x = image.shape[2]
y = image.shape[1]



if x > 1500 or y > 1500:
    surface = np.zeros_like(image)  
    surface[:, y // 2:, x // 2:] = process_patch(image[:, y // 2:, x // 2:], window_size, C, z_expansion)
    surface[:, y // 2:, :x // 2] = process_patch(image[:, y // 2:, :x // 2], window_size, C, z_expansion)
    surface[:, :y // 2, x // 2:] = process_patch(image[:, :y // 2, x // 2:], window_size, C, z_expansion)
    surface[:, :y // 2:, :x // 2] = process_patch(image[:, :y // 2:, :x // 2], window_size, C, z_expansion)
else:
    surface = process_patch(image)
print("saving")
write_tiff(configs, f"surface_with_adaptive_thresholding_C={C}_window_size={window_size}_z_expansion={z_expansion}.tif", surface[::-1,:,:])

