import argparse
import os
from src import util

from src.processors import bash as bash_processor

OUT_DIR = "./out"
MODELS_DIR = f"{OUT_DIR}/models"
MODELS_LIST = f"{OUT_DIR}/models.txt"

def flush_workspace():
    util.remove_files_from_dir(OUT_DIR)
    os.mkdir(OUT_DIR)
    os.mkdir(MODELS_DIR)

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