import argparse
import os
from src import util

from src.processors import bash as bash_processor

OUT_DIR = "./out"
DATASETS_DIR = f"{OUT_DIR}/datasets"
MODELS_DIR = f"{OUT_DIR}/models"
MODELS_LIST = f"{OUT_DIR}/models.txt"
HIVES_DIR= f"{OUT_DIR}/hives"

def flush_workspace():
    util.remove_files_from_dir(OUT_DIR)
    os.mkdir(OUT_DIR)
    # os.mkdir(DATASETS_DIR)
    os.mkdir(MODELS_DIR)
    # os.mkdir(HIVES_DIR)

def main(models: list):
    flush_workspace()
    util.save_models_list(models, MODELS_LIST)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Main', description = 'Prepare workspace', epilog = 'Huh')
    parser.add_argument('-m', '--models', nargs="+")

    args = parser.parse_args()
    models = args.models
    if len(models) == 1:
        models = models[0].split(',')

    main(list(models))

# python.exe generate_main.py --models ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
# python.exe generate_models.py
# python.exe generate_bash.py
# python.exe generate_dataset.py --type remo --src_dir=./source/remo --out_dir=./out/datasets/remo
# python.exe generate_dataset.py --type record --file=./source/records/bee_dataset-train.tfrecord-00000-of-00001 --out_dir=./out/datasets/bee_datasets
# python.exe 
