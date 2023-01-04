import argparse
import time
import os
import zipfile
import pandas as pd
import shutil

def unpack_hive(path, out):
    with zipfile.ZipFile(path,"r") as zip_ref:
        zip_ref.extractall(out)

# python hive_merge.py 
#   --src_hive=./myhive.hive
#   --target_hive=./myhive2.hive
#   --fraction=0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split and pack a hive folder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--src_hive', type=str)
    parser.add_argument('--target_hive', type=str)
    parser.add_argument('--fraction', type=str)
    args = parser.parse_args()
    timestamp = time.time()
    UNPACKED_SOURCE_DIR = f"./out/s{timestamp}"
    UNPACKED_SOURCE_DIR_CSV = f"{UNPACKED_SOURCE_DIR}/annotation.csv"
    UNPACKED_SOURCE_DIR_IMAGES = f"{UNPACKED_SOURCE_DIR}/images"

    UNPACKED_TARGET_DIR = f"./out/t{timestamp}"
    UNPACKED_TARGET_DIR_CSV = f"{UNPACKED_TARGET_DIR}/annotation.csv"
    UNPACKED_TARGET_DIR_IMAGES = f"{UNPACKED_TARGET_DIR}/images"

    os.mkdir(UNPACKED_SOURCE_DIR)
    os.mkdir(UNPACKED_TARGET_DIR)

    src_path = unpack_hive(args.src_hive, UNPACKED_SOURCE_DIR)
    trg_path = unpack_hive(args.target_hive, UNPACKED_TARGET_DIR)

    s_df = pd.read_csv(UNPACKED_SOURCE_DIR_CSV)
    t_df = pd.read_csv(UNPACKED_TARGET_DIR_CSV)

    slice_df = t_df.sample(frac = args.fraction)

    res_df = s_df + slice_df

    res_df.to_csv(UNPACKED_SOURCE_DIR_CSV)

    for filename in slice_df["filename"].to_list():
        try:
            shutil.copy(f"{UNPACKED_TARGET_DIR_IMAGES}/{filename}", f"{UNPACKED_SOURCE_DIR_IMAGES}/{filename}")
        except:
            print("Oh no")
    
    print(f"Merged to {UNPACKED_SOURCE_DIR}")
