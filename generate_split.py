import shutil
import argparse
import os
from src.processors.labels import generate_labels_file
from src.record_csv import create_record_csv
from src.processors import csv as csv_proccessor

from src.util import get_n_files_from, remove_files_from_dir

FILES_LIMIT = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform folder into protoc dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source')
    parser.add_argument('--out')
    parser.add_argument('--class_name')
    args = parser.parse_args()
    IMG_DIR = f"{args.out}/images"
    try:
        remove_files_from_dir(args.out)
        # os.rmdir(args.out)
        os.mkdir(args.out)
        os.mkdir(f"{args.out}/images")
        # os.mkdir(f"{args.out}/images/{args.class_name}")
    except:
        print("Whatever")
        
    files = get_n_files_from(f"./out/datasets/remo/crop/{args.class_name}", FILES_LIMIT)
    for file in files:
        shutil.copyfile(f"./out/datasets/remo/crop/{args.class_name}/{file}", f"{args.out}/images/{file}")

    TRAIN_CSV = f'{args.out}/train.csv'
    TRAIN_RECORD = f'{args.out}/train.tfrecord'

    TEST_CSV = f'{args.out}/test.csv'
    TEST_RECORD = f'{args.out}/test.tfrecord'
    
    LABELS_PBTXT = f'{args.out}/labels.pbtxt'
    
    rows = list()
    labels = list()
    
    labels.append(args.class_name)
    features = csv_proccessor.dir_to_features(args.class_name, IMG_DIR)
    rows.extend(features)

    generate_labels_file(labels, LABELS_PBTXT)
    split_x = int(len(rows) * 0.8)
    training_set, test_set = rows[:split_x], rows[split_x:]
    print(f"Training len: {len(training_set)}; Test len: {len(test_set)}")

    csv_proccessor.save_rows(training_set, TRAIN_CSV)
    csv_proccessor.save_rows(test_set, TEST_CSV)

    create_record_csv(TRAIN_CSV, IMG_DIR, TRAIN_RECORD, LABELS_PBTXT)
    create_record_csv(TEST_CSV, IMG_DIR, TEST_RECORD, LABELS_PBTXT)