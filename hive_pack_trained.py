from src.processors import config as config_processor, csv as csv_processor, labels as labels_processor
from src import record_csv
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import argparse
import shutil
import random
import json

# python hive_pack.py 
#   --hive_dir=./out/myhive
#   --name=myhive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split and pack a hive folder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--hive_dir', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    HIVE_DIR_PATH=args.hive_dir
    HIVE_DIR_CONFIG=f"{HIVE_DIR_PATH}/config.json"
    HIVE_DIR_CSV=f"{HIVE_DIR_PATH}/annotations.csv"
    HIVE_DIR_LABELS=f"{HIVE_DIR_PATH}/labels.pbtxt"
    HIVE_DIR_TFRECORD=f"{HIVE_DIR_PATH}/record.tfrecord"
    HIVE_DIR_IMAGES=f"{HIVE_DIR_PATH}/images"

    with open(HIVE_DIR_CONFIG, 'r', encoding='UTF8') as f:
        config = json.load(f)

    HIVE_MODEL_DIR=f"{HIVE_DIR_PATH}/{config['model']}"
    HIVE_DIR_PIPELINE=f"{HIVE_MODEL_DIR}/pipeline.config"

    HIVE_DIR_TEST_CSV=f"{HIVE_DIR_PATH}/test.csv"
    HIVE_DIR_TEST_TFRECORD=f"{HIVE_DIR_PATH}/test.tfrecord"

    HIVE_DIR_TRAIN_CSV=f"{HIVE_DIR_PATH}/train.csv"
    HIVE_DIR_TRAIN_TFRECORD=f"{HIVE_DIR_PATH}/train.tfrecord"

    # Generate config
    print("Updating the config...")
    config_processor.fill_config(
        config["model"],
        HIVE_MODEL_DIR,
        HIVE_DIR_LABELS,
        HIVE_DIR_TRAIN_TFRECORD,
        HIVE_DIR_TEST_TFRECORD,
        config["num_steps"],
        config["batch_size"],
    )

    # Pack
    print(f"Packing: {args.name}")
    shutil.make_archive(f"./{args.name}.hive", 'zip', args.hive_dir)
    shutil.move(f"./{args.name}.hive.zip", f"./{args.name}.hive")
