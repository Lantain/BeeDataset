import argparse
from src.processors import config as config_processor

OUT_DIR = "./out"
MODELS_LIST = f"{OUT_DIR}/models.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Main', description = 'Prepare workspace', epilog = 'Huh')

    args = parser.parse_args()
    with open(MODELS_LIST, mode='r') as f:
        line = f.readline()
        models = line.split(',')
        for model in models:
            config_processor.fill_config(model)
    