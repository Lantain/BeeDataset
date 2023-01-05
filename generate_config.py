from src.processors import config as config_processor
from src.util import load_models_list, find_record_in_dir, find_labels_in_dir
import argparse

OUT_DIR = "./out"
MODELS_LIST = f"{OUT_DIR}/models.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fill models configs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ds_path')

    args = parser.parse_args()

    models = load_models_list("./out/models.txt")

    for model in models:
        config_processor.fill_config(
            model,
            f"./out/models/{model}",
            find_labels_in_dir(args.ds_path), 
            find_record_in_dir(args.ds_path, 'train'), 
            find_record_in_dir(args.ds_path, 'test'), 
            args.num_steps, 
            args.batch_size
        )
    