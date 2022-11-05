import argparse
import os
from src.util import get_last_checkpoint_name, set_config_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Updates config values',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', metavar='model', type=str, help='Model name')
    parser.add_argument('dir', metavar='dir', type=str, help='dir path')

    args = parser.parse_args()
    name = get_last_checkpoint_name(args.dir)

    set_config_value('fine_tune_checkpoint', f"{os.getcwd()}/{args.dir}/{name}", args.model)