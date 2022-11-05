import os
import cv2
import re
import tensorflow as tf
import numpy as np
from PIL import Image

def remove_files_from_dir(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            remove_files_from_dir(f"{path}/{d}")
            if os.path.isdir(f"{path}/{d}") == True:
                os.removedirs(f"{path}/{d}")
        for f in files:
            os.remove(f"{path}/{f}")
    if os.path.isdir(path) == True:
        os.removedirs(path)



def crop_annotations(source_dir, target_dir, ann, fi):
    [name, ext] = ann["file_name"].split(".") 
    im = Image.open(f"{source_dir}/{ann['file_name']}")
    
    i = 0
    for a in ann["annotations"]:
        box = a["bbox"]
        part = im.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        # print(name)
        for c in a["classes"]:
            if os.path.isdir(f"{target_dir}/{c}") == False:
                os.mkdir(f"{target_dir}/{c}")

            part.save(f"{target_dir}/{c}/{fi}_{i}.{ext}")
            
        i += 1

def image_to_np(path):
    img = cv2.imread(path, 3)
    b,g,r = cv2.split(img)           # get b, g, r
    rgb_img = cv2.merge([r,g,b])     # switch it to r, g, b
    image_np = np.array(rgb_img)
    return image_np

def image_to_input_tensor(path):
    image_np = image_to_np(path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor

def set_config_value(key, value, model):
    path = f"./out/{model}.config"
    with open(path) as f:
        config = f.read()

    with open(path, 'w') as f:
        config = re.sub(f'{key}: ".*?"', f'{key}: "{value}"', config)
        f.write(config)

def get_last_checkpoint_name(model_dir):
    files = os.listdir(model_dir)
    filtered = []

    for f in files:
        if re.search('ckpt-[0-9]{1,2}\.i.+', f):
            filtered.append(f.replace('.index', ''))
    
    filtered.sort()
    last_checkpoint = filtered[len(filtered) - 1]
    return last_checkpoint