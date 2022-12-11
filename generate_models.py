from src.processors import model as model_processor
from src.util import load_models_list
from tqdm import tqdm

if __name__ == '__main__':
    models = load_models_list("./out/models.txt")
    for model in tqdm(models, desc="models"):
        model_processor.download_model(model)
        model_processor.decompress_model(model)