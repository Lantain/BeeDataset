import argparse
import shutil
import os
import json
import time
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

def run(pipeline_config_path, model_dir, num_train_steps):
    tf.config.set_soft_device_placement(True)
    strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            train_steps=num_train_steps,
            use_tpu=False,
            checkpoint_every_n=250,
            record_summaries=True
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str)
    parser.add_argument('--num_steps', type=int, required=False)
    args = parser.parse_args()

    name = os.path.basename(args.hive)
    model_dir = f"./out/{str(time.time())}-{name.replace('.hive', '')}"
    shutil.unpack_archive(args.hive, model_dir, 'zip')
    
    with open(f"{model_dir}/config.json", 'r', encoding='UTF8') as f:
        config = json.load(f)
    print(f"Model: {model_dir}")
    run(
        f"{model_dir}/{config['model']}/pipeline.config",
        model_dir,
        num_train_steps=args.num_steps or config.num_steps
    )
    print(f"Model: {model_dir}")