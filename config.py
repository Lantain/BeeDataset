import argparse
import os
from src.util import get_last_checkpoint_name
from src.processors import config as config_processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('key')
    parser.add_argument('-m', '--model')
    args = parser.parse_args()
    
    out = f"{os.getcwd()}/out/models/{args.model}/trained"

    name = get_last_checkpoint_name(out)
    print(f"Updating {args.key} config value for model: {args.model}...")
    config_processor.update_config_values_regex(args.model, list([
            {
                "regex": f'{args.key}: ".*?"',
                "value": f'{args.key}: "{out}/{name}"'
            }
        ])
    )