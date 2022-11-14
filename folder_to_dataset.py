import argparse
import os
from src import annotations_csv
from src.processors.labels import generate_labels_file
from src.record_csv import create_record_csv
from PIL import Image


def dir_to_features(label, path):
    rows = list()
    for root, dirs, files in os.walk(args.dir):
        for f in files:
            image = Image.open(f"{path}/{f}")
            rows.append([f, label, image.width, image.height, 0, 0, image.width, image.height])
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Transform folder into protoc dataset',
    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--dir')
    parser.add_argument('-c', '--class_name')
    args = parser.parse_args()
    rows = list()
    labels = list()

    TRAIN_CSV = f'{args.dir}/{args.class_name}_train.csv'
    TRAIN_RECORD = f'{args.dir}/{args.class_name}_train.record'

    TEST_CSV = f'{args.dir}/{args.class_name}_test.csv'
    TEST_RECORD = f'{args.dir}/{args.class_name}_test.record'
    
    LABELS_PBTXT = f'{args.dir}/{args.class_name}_labels.pbtxt'
    IMG_DIR = f"{args.dir}/{args.class_name}"

    for root, dirs, files in os.walk(args.dir):
        for d in dirs:
            labels.append(d)
            features = dir_to_features(d, f"{args.dir}/{d}")
            rows.extend(features)

    generate_labels_file(labels, LABELS_PBTXT)

    split_x = int(len(rows) * 0.8)
    training_set, test_set = rows[:split_x], rows[split_x:]
    print(f"Training len: {len(training_set)}; Test len: {len(test_set)}")
    
    annotations_csv.save_rows(training_set, TRAIN_CSV)
    annotations_csv.save_rows(test_set, TEST_CSV)

    create_record_csv(TRAIN_CSV, IMG_DIR, TRAIN_RECORD, LABELS_PBTXT)
    create_record_csv(TEST_CSV, IMG_DIR, TEST_RECORD, LABELS_PBTXT)