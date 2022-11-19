import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from io import BytesIO

def parse_record(record):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    result = {}

    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)
    return result

def binary_to_img(bin, path):
    image = np.array(Image.open(BytesIO(bin)))
    img = Image.fromarray(image)
    img.save(path)
    return img

def extract_dataset(record_path, out_dir):
    raw_dataset = tf.data.TFRecordDataset(record_path)
    rows = {} 
    i = 1
    for rec in raw_dataset:
        result = parse_record(rec)
        binary_to_img(result['input'], f'{out_dir}/images')
        
        rows[f'{i}.jpg'] = {
            "cooling": result['output/cooling_output'][0],
            "pollen": result['output/pollen_output'][0],
            "varroa": result['output/varroa_output'][0],
            "wasps": result['output/wasps_output'][0]
        }

        i += 1
        
    with open(f'{out_dir}/annotations.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

def generate_dataset(record_path, out_dir):
    extract_dataset(record_path, out_dir)

    
    