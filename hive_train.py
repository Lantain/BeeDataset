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
            checkpoint_every_n=1000,
            record_summaries=True,
            checkpoint_max_to_keep=20
        )

def main(unused_argv):
    parser = argparse.ArgumentParser(description='Train a hive')
    parser.add_argument('--hive', type=str)
    parser.add_argument('--dir', type=str, required=False, default=".")
    parser.add_argument('--num_steps', type=int, required=False)
    args = parser.parse_args()

    name = os.path.basename(args.hive)
    model_dir = f"{args.dir}/{name.replace('.hive', '')}"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    
    shutil.copy(args.hive, f"{args.hive}.zip")
    shutil.unpack_archive(args.hive, model_dir, 'zip')
    os.remove(f"{args.hive}.zip")
    
    with open(f"{model_dir}/config.json", 'r', encoding='UTF8') as f:
        config = json.load(f)
    print(f"Model: {model_dir}")
    os.chdir(f"{model_dir}/..")
    run(
        f"{model_dir}/{config['model']}/pipeline.config",
        f"{model_dir}/trained",
        num_train_steps=args.num_steps or config.num_steps
    )
    print(f"Model: {model_dir}")

if __name__ == '__main__':
    tf.compat.v1.app.run()