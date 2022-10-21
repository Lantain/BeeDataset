import os
import io
import re
import pandas as pd
import tensorflow as tf
import argparse
import requests
import tarfile


def download_model(model_name):
    model_url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"
    r = requests.get(model_url, allow_redirects=True)
    open(f"out/{model_name}.tar.gz", 'wb').write(r.content)


def download_config(model_name):
    model_url = f"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/{model_name}.config"
    r = requests.get(model_url, allow_redirects=True)
    open(f"out/{model_name}.config", 'wb').write(r.content)


def decompress_model(model_name):
    file = tarfile.open(f'out/{model_name}.tar.gz')
    file.extractall(f'out/{model_name}')
    file.close()


def fill_config(model_name, base_path):
    base_config_path = f'{model_name}.config'
    labelmap_path = f'{base_path}/assets/labels.pbtxt'
    fine_tune_checkpoint = f'{base_path}/out/{model_name}/checkpoint/ckpt-0'
    train_record_path = f'{base_path}/out/train/train_csv.tfrecord'
    test_record_path = f'{base_path}/out/test/test_csv.tfrecord'
    num_classes = 1
    batch_size = 32
    num_steps = 30000

    with open(f'{base_path}/out/{base_config_path}') as f:
        config = f.read()

    with open(f'{base_path}/out/{base_config_path}', 'w') as f:

        # Set labelmap path
        config = re.sub('label_map_path: ".*?"',
                        'label_map_path: "{}"'.format(labelmap_path), config)

        # Set fine_tune_checkpoint path
        config = re.sub('fine_tune_checkpoint: ".*?"',
                        'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)

        # Set train tf-record file path
        config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
                        'input_path: "{}"'.format(train_record_path), config)

        # Set test tf-record file path
        config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
                        'input_path: "{}"'.format(test_record_path), config)

        # Set number of classes.
        config = re.sub('num_classes: [0-9]+',
                        'num_classes: {}'.format(num_classes), config)

        # Set batch size
        config = re.sub('batch_size: [0-9]+',
                        'batch_size: {}'.format(batch_size), config)

        # Set training steps
        config = re.sub('num_steps: [0-9]+',
                        'num_steps: {}'.format(num_steps), config)

        # Set fine-tune checkpoint type to detection
        config = re.sub('fine_tune_checkpoint_type: "classification"',
                        'fine_tune_checkpoint_type: "{}"'.format('detection'), config)

        f.write(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate config & checkpoint for the given model',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', metavar='model', type=str, help='Model name')
    parser.add_argument('base_path', metavar='base_path', type=str, help='Base beedataset path', default=os.getcwd())

    args = parser.parse_args()
    download_config(args.model)
    download_model(args.model)
    decompress_model(args.model)
    fill_config(args.model, args.base_path)

    print(f'{args.base_path}/out/{args.model}.config')
