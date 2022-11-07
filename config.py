import argparse
import os
from src.util import get_last_checkpoint_name
from src.processors import config as config_processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--model')
    parser.add_argument('-o', '--out')

    args = parser.parse_args()
    name = get_last_checkpoint_name(args.out)
    key = 'fine_tune_checkpoint'
    config_processor.update_config_values_regex(args.model, list([
            {
                "regex": '{key}: ".*?"',
                "value": f"{args.out}/{name}"
            }
        ])
    )