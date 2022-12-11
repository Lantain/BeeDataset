import shutil
import argparse
import os

from src.util import get_n_files_from, remove_files_from_dir
# from processors.csv import transfer_n_fields

DS_OUT = f"./out/ds"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform folder into protoc dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source')
    args = parser.parse_args()
    try:
        remove_files_from_dir(DS_OUT)

        os.mkdir(f"{DS_OUT}")
        os.mkdir(f"{DS_OUT}/images")
        os.mkdir(f"{DS_OUT}/images/Bee")
    except:
        print("Whatever")
        
    files = get_n_files_from("./out/datasets/remo/crop/Bee", 1000)
    for file in files:
        shutil.copyfile(f"./out/datasets/remo/crop/Bee/{file}", f"{DS_OUT}/images/Bee/{file}")
    