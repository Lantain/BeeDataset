import csv
import pandas as pd
import os
import shutil

header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
def save_rows(rows: list, out_file: str):
    with open(out_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row in rows:
            writer.writerow(row)

def transfer_n_fields(csv_path, source_dir, target_dir, n):
    data = pd.read_csv(csv_path, nrows=n)
    for row in data:
        shutil.copyfile(f"{source_dir}/{row['filename']}", f"{target_dir}/{row['filename']}")

    return data
    