import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from io import BytesIO
from src.processors import csv as csv_processor, labels as labels_processor

def get_labels_list():
    return list(["Bee", "Varroa", "Cooling", "Wasp", "Pollen"])

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
        if 'bee_dataset' in record_path:
            binary_to_img(result['input'], f'{out_dir}/images/{i}.jpg')
            
            rows[f'{i}.jpg'] = {
                "cooling": result['output/cooling_output'][0],
                "pollen": result['output/pollen_output'][0],
                "varroa": result['output/varroa_output'][0],
                "wasps": result['output/wasps_output'][0]
            }

        i += 1
        
    with open(f'{out_dir}/record_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

def generate_csv_from_bee_annotation(filename, ann, out_dir):
    [name, ext] = filename.split(".")
    i = Image.open(f"{out_dir}/images/{filename}")
    rows = []

    if ann['varroa'] == 1.:
        rows.append([filename, "Varroa", i.width, i.height, 0, 0, i.width, i.height])
    if ann['cooling'] == 1.:
        rows.append([filename, "Cooling", i.width, i.height, 0, 0, i.width, i.height])
    if ann['wasps'] == 1.:
        rows.append([filename, "Wasp", i.width, i.height, 0, 0, i.width, i.height])
    else:
        rows.append([filename, "Bee", i.width, i.height, 0, 0, i.width, i.height])
    if ann['pollen'] == 1.:
        rows.append([filename, "Pollen", i.width, i.height, 0, 0, i.width, i.height])

    csv_processor.save_rows(rows, f"{out_dir}/csvs/{name}.csv")
    return rows

def bee_dataset_to_csv(out_dir):
    f = open(f'{out_dir}/record_annotations.json')
    data = json.load(f)
    ds_rows = list()
    for filename in data:
        rows = generate_csv_from_bee_annotation(filename, data[filename], out_dir)
        ds_rows.extend(rows)
    csv_processor.save_rows(ds_rows, f"{out_dir}/record_annotations.csv")
    
    bee_rows = list()
    for row in ds_rows:
        if row[0] == "Bee":
            bee_rows.append(row[0])

    csv_processor.save_rows(ds_rows, f"{out_dir}/annotations.csv")

def generate_dataset(record_path, out_dir):
    os.mkdir(out_dir)
    os.mkdir(f"{out_dir}/csvs")
    os.mkdir(f"{out_dir}/images")
    extract_dataset(record_path, out_dir)
    labels_processor.generate_labels_file(get_labels_list(), f'{out_dir}/labels.pbtxt')
    
    if 'bee_dataset' in record_path:
        bee_dataset_to_csv(out_dir)
