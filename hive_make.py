from src.processors import model as model_processor, csv as csv_processor, labels as labels_processor
from src.datasets import remo, record
from src import util, record_csv

import argparse
import shutil
import os
import json
import time

# python hive_make.py 
#   --model=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
#   --out=./out/myhive
#   --dataset=remo:./source/remo
#   --num_steps=10000
#   --batch_size=16
#   --test_ratio=0.2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a hive folder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_steps', type=int, default=1000, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False)
    parser.add_argument('--test_ratio', type=float, default=0.2, required=False)
    
    args = parser.parse_args()
    
    HIVE_DIR_PATH=args.out
    HIVE_MODEL_DIR=f"{HIVE_DIR_PATH}/{args.model}"
    HIVE_DIR_DATASET=f"{HIVE_DIR_PATH}/dataset"
    HIVE_DIR_DATASET_IMAGES = f"{HIVE_DIR_DATASET}/images"
    HIVE_DIR_DATASET_CSV = f"{HIVE_DIR_DATASET}/annotations.csv"
    HIVE_DIR_CSV=f"{HIVE_DIR_PATH}/annotations.csv"
    HIVE_DIR_LABELS=f"{HIVE_DIR_PATH}/labels.pbtxt"
    HIVE_DIR_TFRECORD=f"{HIVE_DIR_PATH}/record.tfrecord"
    HIVE_DIR_IMAGES=f"{HIVE_DIR_PATH}/images"
    HIVE_DIR_CONFIG=f"{HIVE_DIR_PATH}/config.json"

    util.remove_files_from_dir(HIVE_DIR_PATH)
    os.mkdir(HIVE_DIR_PATH)

    ds_type, ds_path = args.dataset.split(":")

    print("Dataset parse... ", end='')
    if ds_type == 'remo':
        remo.generate_dataset(f'{ds_path}/remo.json', ds_path, HIVE_DIR_DATASET)
    elif ds_type == 'record':
        record.generate_dataset(ds_path, HIVE_DIR_DATASET)
    print("OK")

    # Download and unpack model
    print("Downloading model... ", end='')
    model_processor.download_model(args.model)
    model_processor.decompress_model(args.model, HIVE_DIR_PATH)
    print("OK")

    print("Transfer files...", end='')
    # Move images
    shutil.copytree(HIVE_DIR_DATASET_IMAGES, HIVE_DIR_IMAGES)
    # Copy annotations
    shutil.copy(HIVE_DIR_DATASET_CSV, HIVE_DIR_CSV)
    print("OK")

    print("Create hive_config...", end='')
    # Create a Config file
    default_config = {
        "created_at": str(time.gmtime()),
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "model": args.model,
        "test_train_ratio": args.test_ratio,
        # "labels": labels
    }

    with open(HIVE_DIR_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=4)

    print("OK")

    print("Cleaning package folder...", end='')
    # Remove dataset folder
    shutil.rmtree(HIVE_DIR_DATASET)
    print("OK")