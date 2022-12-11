import argparse
import csv
import os
import shutil
from src.datasets import remo
from src.datasets import record
from src.util import remove_files_from_dir
from src.processors import csv as csv_processor

SPLIT_RATIO = .8

def copy_rows(rows, from_dir, to_dir):
    os.mkdir(to_dir)
    os.mkdir(f"{to_dir}/images")
    os.mkdir(f"{to_dir}/csvs")

    is_header = True
    for row in rows:
        if len(row) == 0 or is_header is True:
            if is_header:
                is_header = False
            continue
        filename = row[0]
        [name, ext] = filename.split('.')

        shutil.copyfile(f"{from_dir}/images/{filename}", f"{to_dir}/images/{filename}")
        shutil.copyfile(f"{from_dir}/csvs/{name}.csv", f"{to_dir}/csvs/{name}.csv")

    csv_processor.save_rows(rows, f"{to_dir}/annotations.csv")

def split_dir(out_dir):
    with open(f"{out_dir}/annotations.csv") as csvfile:
        rows = list(csv.reader(csvfile))

        split_x = int(len(rows) * 0.8)
        training_set, test_set = rows[1:split_x], rows[1:split_x:]
        
        copy_rows(training_set, out_dir, f"{out_dir}/train")
        copy_rows(test_set, out_dir, f"{out_dir}/test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Process available datasets',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-t', '--type')
    parser.add_argument('-f', '--file', required=False)
    parser.add_argument('-s', '--src_dir', required=False)
    parser.add_argument('-o', '--out_dir')
    
    args = parser.parse_args()

    remove_files_from_dir(args.out_dir)
    remove_files_from_dir('./out/datasets/train')
    remove_files_from_dir('./out/datasets/test')

    if args.type == 'remo':
        remo.generate_dataset(f'{args.src_dir}/remo.json', args.src_dir, args.out_dir)
    elif args.type == 'record':
        record.generate_dataset(args.file, args.out_dir)
    
    split_dir(args.out_dir)