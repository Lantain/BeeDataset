from src.processors import config as config_processor
from util import load_models_list

OUT_DIR = "./out"
MODELS_LIST = f"{OUT_DIR}/models.txt"

if __name__ == '__main__':
    models = load_models_list("./out/models.txt")
    for model in models:
        config_processor.fill_config(model)
    