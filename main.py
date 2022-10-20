import json
import os
import shutil
import util
import image 
import csv_processor
import record_json

SOURCE_DIR = "./source"
ASSETS_DIR = "./assets"
OUT_DIR = "./out"

JSON_INPUT =         f"{ASSETS_DIR}/remo.json"
PBTXT_INPUT =        f"{ASSETS_DIR}/labels.pbtxt"

TRAIN_DIR =          f"{OUT_DIR}/train"
TRAIN_IMAGES_DIR =   f"{OUT_DIR}/train/images"
TRAIN_CSV_DIR =      f"{OUT_DIR}/train/annotations"

TEST_DIR =          f"{OUT_DIR}/test"
TEST_IMAGES_DIR =   f"{OUT_DIR}/test/images"
TEST_CSV_DIR =      f"{OUT_DIR}/test/annotations"

USED_DIR =           f"{OUT_DIR}/used"
CROPPED_DIR =        f"{OUT_DIR}/crop"
CSVS_DIR =           f"{OUT_DIR}/used_csvs"
RECORD_OUTPUT_PATH = f"{OUT_DIR}/record.tfrecord"

def start():
    print("[0] Pipeline started")

    f = open(JSON_INPUT)
    data = json.load(f)

    print("[1] Clearing...")
    util.remove_files_from_dir(OUT_DIR)

    os.mkdir(OUT_DIR)
    os.mkdir(USED_DIR)
    os.mkdir(CROPPED_DIR)
    print("[1] Done")

    print("[2] Generated images...")
    file_i = 0
    for a in data:
        shutil.copyfile(f"{SOURCE_DIR}/{a['file_name']}", f"{USED_DIR}/{a['file_name']}")
        image.crop_annotations(USED_DIR, CROPPED_DIR, a, file_i)
        file_i += 1
    print("[2] Done")

    print("[3] Generate csvs...")
    os.mkdir(CSVS_DIR)

    for a in data:
        csv_processor.generate_csv_from_annotation(a, CSVS_DIR)
    print("[3] Done")

    print("[4] Splitting on sets")
    split_x = int(len(data) * 0.8)
    training_set, test_set = data[:split_x], data[split_x:]
    print(f"[4] Done. Training len: {len(training_set)}; Test len: {len(test_set)}")

    print("[5] Generating training & test folders...")
    try:
        os.mkdir(TRAIN_DIR)
        os.mkdir(TRAIN_IMAGES_DIR)
        os.mkdir(TRAIN_CSV_DIR)

        os.mkdir(TEST_DIR)
        os.mkdir(TEST_IMAGES_DIR)
        os.mkdir(TEST_CSV_DIR)
    except:
        print("[5] Failed to create directories")

    for t in training_set:
        filename = t['file_name']
        [name, ext] = filename.split(".") 
        shutil.copyfile(f"{USED_DIR}/{filename}", f"{TRAIN_IMAGES_DIR}/{filename}")
        shutil.copyfile(f"{CSVS_DIR}/{name}.csv", f"{TRAIN_CSV_DIR}/{name}.csv")
        csv_processor.generate_csv_from_annotation_set(training_set, f"{TRAIN_DIR}/annotations.csv")
        json_object = json.dumps(training_set, indent=4)
        with open(f"{TRAIN_DIR}/annotations.json", "w") as outfile:
            outfile.write(json_object)

    for t in test_set:
        filename = t['file_name']
        [name, ext] = filename.split(".") 
        shutil.copyfile(f"{USED_DIR}/{filename}", f"{TEST_IMAGES_DIR}/{filename}")
        shutil.copyfile(f"{CSVS_DIR}/{name}.csv", f"{TEST_CSV_DIR}/{name}.csv")
        csv_processor.generate_csv_from_annotation_set(test_set, f"{TEST_DIR}/annotations.csv")
        json_object = json.dumps(test_set, indent=4)
        with open(f"{TEST_DIR}/annotations.json", "w") as outfile:
            outfile.write(json_object)

    print("[5] Done")

    print("[6] Generate records")
    record_json.create_record_json(
        f"{TRAIN_DIR}/annotations.json", 
        TRAIN_IMAGES_DIR, 
        f"{TRAIN_DIR}/train.record", 
        PBTXT_INPUT
    )
    record_json.create_record_json(
        f"{TEST_DIR}/annotations.json", 
        TEST_IMAGES_DIR, 
        f"{TEST_DIR}/train.record",
        PBTXT_INPUT
    )
    print("[6] Done")
start()