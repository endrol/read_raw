import cv2
import numpy as np
import pdb
import json
import pandas as pd



raw_path = "/workspace/dataset/data_raw_image/iso12233_dro_on_iso125.raw"
json_path = "/workspace/dataset/data_raw_image/iso12233_dro_on_iso125.json"


"""step one, read from file"""
raw_data = np.fromfile(raw_path, dtype=np.uint16)


with open(json_path) as f:
    config_json = json.load(f)

"""step 2, normalize to unit8"""
height, width = config_json["sensor_input"]["height"], config_json["sensor_input"]["stride"]
# reshape
raw_data = raw_data.reshape([height, width])
# normalize
img_data = raw_data / (2**(config_json["opcode"]["ct"]["acc"][1] - 8))

"""# step 3
# blc operation"""
img_data = img_data - config_json["opcode"]["blc"]["black_level"]["data"][0]

"""# step 4 wlc"""
assert config_json["sensor_input"]["bayer_format"] == "RGGB", "this only for RGGB pattern"
# R channel
img_data[0::2, 0::2] *= config_json["opcode"]["wbc"]["acc"][0]
# G channel
img_data[0::2, 1::2] *= config_json["opcode"]["wbc"]["acc"][1]
img_data[1::2, 0::2] *= config_json["opcode"]["wbc"]["acc"][1]

"""# step 5 clip"""
img = np.clip(img_data, 0, 255)
img = img.astype(np.uint8)

# R channel
r_channel = np.zeros(img.shape, dtype=np.uint8)
r_channel[0::2, 0::2] = img[0::2, 0::2]

# b_channel
b_channel = np.zeros(img.shape, dtype=np.uint8)
b_channel[1::2, 1::2] = img[1::2, 1::2]

# g_channel
g_channel = np.zeros(img.shape, dtype=np.uint8)
g_channel[0::2, 1::2] = img[0::2, 1::2]
g_channel[1::2, 0::2] = img[1::2, 0::2]

full_image = np.concatenate(([b_channel], [g_channel], [r_channel]), axis=0)
full_image = np.transpose(full_image, (1,2,0))

cv2.imwrite("bayer_image.png",full_image)
cv2.imwrite("pure_bayer.png", img)

print("ok")