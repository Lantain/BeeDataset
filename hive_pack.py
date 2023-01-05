from src.processors import config as config_processor, csv as csv_processor
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

    df = pd.read_csv(HIVE_DIR_CSV)
    
    test = list()
    train = list()

    labels = df["class"].unique().tolist()
    for label in list(config["labels"]):
        dft = df.query(f"class == {label}")
        df_train, df_test = train_test_split(dft, test_size=config["test_train_ratio"])
        train.extend(df_train)
        test.extend(df_test)
    
    random.seed(0)
    random.shuffle(train)
    random.shuffle(test)

    print(f"Packing {len(train)} train rows")
    print(f"Packing {len(test)} test rows")

    # Save Split to CSVs
    csv_processor.save_rows(train, HIVE_DIR_TRAIN_CSV)
    csv_processor.save_rows(test, HIVE_DIR_TEST_CSV)

    # Generate Split records
    print("Generating records...")
    record_csv.create_record_csv(
        HIVE_DIR_TRAIN_CSV, 
        HIVE_DIR_IMAGES, 
        HIVE_DIR_TRAIN_TFRECORD, 
        HIVE_DIR_LABELS
    )

    record_csv.create_record_csv(
        HIVE_DIR_TEST_CSV, 
        HIVE_DIR_IMAGES, 
        HIVE_DIR_TEST_TFRECORD, 
        HIVE_DIR_LABELS
    )

    # Generate config
    print("Filling the config...")
    config_processor.fill_config(
        config["model"],
        HIVE_DIR_LABELS,
        HIVE_DIR_TRAIN_TFRECORD,
        HIVE_DIR_TEST_TFRECORD,
        config["num_steps"],
        config["batch_size"],
        HIVE_DIR_PIPELINE
    )

    # Pack
    print(f"Packing: {args.name}")
    shutil.make_archive(f"./{args.name}.hive", 'zip', args.hive_dir)
