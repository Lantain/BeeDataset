import argparse
import os
from src import util

from src.processors import bash as bash_processor

OUT_DIR = "./out"
DATASETS_DIR = f"./out/datasets"
MODELS_LIST = f"{OUT_DIR}/models.txt"

def flush_workspace():
    util.remove_files_from_dir(OUT_DIR)
    os.mkdir(OUT_DIR)
    os.mkdir(DATASETS_DIR)

def main(models: list):
    flush_workspace()
    util.save_models_list(models, MODELS_LIST)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Main', description = 'Prepare workspace', epilog = 'Huh')
    parser.add_argument('-m', '--models', nargs="+")
    parser.add_argument('-l', '--labels', nargs="+")
    parser.add_argument('-t', '--trained_path')

    args = parser.parse_args()
    
    main(list(args.models))
    