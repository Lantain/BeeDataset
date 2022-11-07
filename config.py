import argparse
import os
from src.util import get_last_checkpoint_name, set_config_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--model')
    parser.add_argument('-o', '--out')

    args = parser.parse_args()
    name = get_last_checkpoint_name(args.out)

    set_config_value('fine_tune_checkpoint', f"{os.getcwd()}/{args.out}/{name}", args.model)