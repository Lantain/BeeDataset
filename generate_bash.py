from src.processors import bash as bash_processor
from src.util import load_models_list
import os

if __name__ == '__main__':
    models = load_models_list("./out/models.txt")
    out_dir = f"{os.getcwd()}/out/models"

    for model in models:
        bash_processor.train_sh(model, out_dir)
        bash_processor.inference_graph_sh(model, out_dir)
        bash_processor.save_sh(model, out_dir)
    bash_processor.train_env_sh(out_dir)
    bash_processor.inference_graph_env_sh(out_dir)
    bash_processor.save_env_sh(out_dir)