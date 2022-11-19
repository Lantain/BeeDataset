import shutil
from src.processors import csv as csv_processor

OUT_DIR = "./out"

TRAIN_DIR =          f"{OUT_DIR}/train"
TRAIN_IMAGES_DIR =   f"{OUT_DIR}/train/images"
TRAIN_CSV_DIR =      f"{OUT_DIR}/train/annotations"

def process_set(rows: list, ds_dir: str, out_dir: str):
    csv_processor.save_rows(rows, f"{out_dir}/annotations.csv")
    for r in rows:
        filename = r['filename']
        [name, ext] = filename.split(".") 
        shutil.copyfile(f"{ds_dir}/images/{filename}", f"{out_dir}/images/{filename}")
        shutil.copyfile(f"{ds_dir}/csvs/{name}.csv", f"{out_dir}/csvs/{name}.csv")
        
def split_sets(rows: list):
    split_x = int(len(rows) * 0.8)
    training_set, test_set = rows[:split_x], rows[split_x:]
    print(f"Training len: {len(training_set)}; Test len: {len(test_set)}")
    return training_set, test_set