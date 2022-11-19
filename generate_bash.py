from src.processors import bash as bash_processor
from util import load_models_list

if __name__ == '__main__':
    models = load_models_list()
    for model in models:
        bash_processor.train_sh(model)
        bash_processor.inference_graph_sh(model)
        bash_processor.save_sh(model)
    bash_processor.train_env_sh()
    bash_processor.inference_graph_env_sh()
    bash_processor.save_env_sh()