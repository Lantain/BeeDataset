from src.processors import model as model_processor
from tqdm import tqdm

from util import load_models_list

if __name__ == '__main__':
    models = load_models_list()
    for model in tqdm(models, desc="models"):
        model_processor.download_model(model)
        model_processor.decompress_model(model)