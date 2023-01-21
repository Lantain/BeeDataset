from src.processors import config as config_processor
import os
import argparse
import shutil
import json
from src.util import get_last_checkpoint_name

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
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

    HIVE_DIR_PATH=f"./{args.name}"
    HIVE_DIR_CONFIG=f"{HIVE_DIR_PATH}/config.json"
    HIVE_DIR_CSV=f"{HIVE_DIR_PATH}/annotations.csv"
    HIVE_DIR_LABELS=f"{HIVE_DIR_PATH}/labels.pbtxt"
    HIVE_DIR_TFRECORD=f"{HIVE_DIR_PATH}/record.tfrecord"
    HIVE_DIR_IMAGES=f"{HIVE_DIR_PATH}/images"

    with open(HIVE_DIR_CONFIG, 'r', encoding='utf-8') as f:
        config = json.load(f)

    HIVE_MODEL_DIR=f"{HIVE_DIR_PATH}/{config['model']}"
    HIVE_DIR_PIPELINE=f"{HIVE_MODEL_DIR}/pipeline.config"

    HIVE_DIR_TEST_CSV=f"{HIVE_DIR_PATH}/test.csv"
    HIVE_DIR_TEST_TFRECORD=f"{HIVE_DIR_PATH}/test.tfrecord"

    HIVE_DIR_TRAIN_CSV=f"{HIVE_DIR_PATH}/train.csv"
    HIVE_DIR_TRAIN_TFRECORD=f"{HIVE_DIR_PATH}/train.tfrecord"
    HIVE_DIR_TRAINED = f"{HIVE_DIR_PATH}/trained"
    HIVE_DIR_INFERENCE = f"{HIVE_DIR_PATH}/inference"

    name = get_last_checkpoint_name(args.hive_dir)
    config["last_checkpoint"] = name
    
    with open(HIVE_DIR_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Generate config
    print("Updating the config...")
    config_processor.set_config_value(
        'fine_tune_checkpoint', 
        f"{args.hive_dir}/trained/{name}", 
        HIVE_MODEL_DIR
    )
    print("Generating inference...")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(HIVE_DIR_PIPELINE, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    exporter_lib_v2.export_inference_graph(
        'image_tensor', 
        pipeline_config, 
        HIVE_DIR_TRAINED,
        HIVE_DIR_INFERENCE
    )

    # Pack
    print(f"Packing: {args.name}")
    shutil.make_archive(f"./t_{args.name}.hive", 'zip', HIVE_DIR_PATH)
    shutil.move(f"./t_{args.name}.hive.zip", f"./t_{args.name}.hive")
    shutil.rmtree(HIVE_DIR_PATH)
