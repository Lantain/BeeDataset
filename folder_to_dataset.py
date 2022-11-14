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

    for root, dirs, files in os.walk(args.dir):
        for d in dirs:
            print(f'found dir: {d}')
            features = dir_to_features(d, f"{args.dir}/{d}")
            rows.extend(features)
        generate_labels_file(dirs, 'out/folder.pbtxt')

    split_x = int(len(rows) * 0.8)
    training_set, test_set = rows[:split_x], rows[split_x:]
    print(f"Training len: {len(training_set)}; Test len: {len(test_set)}")
    
    annotations_csv.save_rows(training_set, './out/folder_training.csv')
    annotations_csv.save_rows(test_set, './out/folder_test.csv')

    create_record_csv('./out/folder_training.csv', f"{args.dir}/{args.class_name}", './out/folder_train.record', './out/folder.pbtxt')
    create_record_csv('./out/folder_test.csv', f"{args.dir}/{args.class_name}", './out/folder_test.record', './out/folder.pbtxt')