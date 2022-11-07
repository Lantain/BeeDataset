import json
import os
from tqdm import tqdm
import shutil
import argparse
from src import util, annotations_csv, record_csv
from src.processors import model as model_processor
from src.processors import config as config_processor
from src.processors import bash as bash_processor
from src.util import save_models_list

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

MODELS_LIST =        f"{OUT_DIR}/models.txt"

def main(models, trained_path):
    print("[0] Pipeline started")

    f = open(JSON_INPUT)
    data = json.load(f)

    print("[1] Clearing...", end=' ')
    util.remove_files_from_dir(OUT_DIR)

    os.mkdir(OUT_DIR)
    os.mkdir(USED_DIR)
    os.mkdir(CROPPED_DIR)
    print("Done")

    print("[2] Generating images...", end=' ')
    file_i = 0
    for a in data:
        shutil.copyfile(f"{SOURCE_DIR}/{a['file_name']}", f"{USED_DIR}/{a['file_name']}")
        util.crop_annotations(USED_DIR, CROPPED_DIR, a, file_i)
        file_i += 1
    print("Done")

    print("[3] Generate csvs...", end='')
    os.mkdir(CSVS_DIR)

    for a in data:
        annotations_csv.generate_csv_from_annotation(a, CSVS_DIR)
    print("Done")

    print("[4] Splitting on sets...", end=' ')
    split_x = int(len(data) * 0.8)
    training_set, test_set = data[:split_x], data[split_x:]
    print(f"Done. Training len: {len(training_set)}; Test len: {len(test_set)}")

    print("[5] Generating training & test folders... ", end=' ')
    try:
        os.mkdir(TRAIN_DIR)
        os.mkdir(TRAIN_IMAGES_DIR)
        os.mkdir(TRAIN_CSV_DIR)

        os.mkdir(TEST_DIR)
        os.mkdir(TEST_IMAGES_DIR)
        os.mkdir(TEST_CSV_DIR)
    except:
        print("Failed to create directories")

    for t in training_set:
        filename = t['file_name']
        [name, ext] = filename.split(".") 
        shutil.copyfile(f"{USED_DIR}/{filename}", f"{TRAIN_IMAGES_DIR}/{filename}")
        shutil.copyfile(f"{CSVS_DIR}/{name}.csv", f"{TRAIN_CSV_DIR}/{name}.csv")
        annotations_csv.generate_csv_from_annotation_set(training_set, f"{TRAIN_DIR}/annotations.csv")
        json_object = json.dumps(training_set, indent=4)
        with open(f"{TRAIN_DIR}/annotations.json", "w") as outfile:
            outfile.write(json_object)

    for t in test_set:
        filename = t['file_name']
        [name, ext] = filename.split(".") 
        shutil.copyfile(f"{USED_DIR}/{filename}", f"{TEST_IMAGES_DIR}/{filename}")
        shutil.copyfile(f"{CSVS_DIR}/{name}.csv", f"{TEST_CSV_DIR}/{name}.csv")
        annotations_csv.generate_csv_from_annotation_set(test_set, f"{TEST_DIR}/annotations.csv")
        json_object = json.dumps(test_set, indent=4)
        with open(f"{TEST_DIR}/annotations.json", "w") as outfile:
            outfile.write(json_object)

    print("Done")

    print("[6] Generate records")
    record_csv.create_record_csv(
        f"{TRAIN_DIR}/annotations.csv", 
        TRAIN_IMAGES_DIR, 
        f"{TRAIN_DIR}/train_csv.tfrecord", 
        PBTXT_INPUT
    )
    record_csv.create_record_csv(
        f"{TEST_DIR}/annotations.csv", 
        TEST_IMAGES_DIR, 
        f"{TEST_DIR}/test_csv.tfrecord", 
        PBTXT_INPUT
    )
    print("[6] Done")

    print("[7] Save models list...", end=' ')
    save_models_list(models, MODELS_LIST)
    print("Done")

    print("[8] Download & decompress model archives")
    for model in tqdm(models, desc="models"):
        model_processor.download_model(model)
        model_processor.decompress_model(model)
    print("[8] Done")

    print("[9] Fill config defaults", end=' ')
    for model in models:
        config_processor.fill_config_defaults(model)
    print("Done")

    print("[10] Prepare bash...", end=' ')
    for model in models:
        bash_processor.train_sh(model, trained_path)
        bash_processor.inference_graph_sh(model, trained_path)
        bash_processor.save_sh(model, trained_path)
    bash_processor.train_env_sh(trained_path)
    bash_processor.inference_graph_env_sh(trained_path)
    bash_processor.save_env_sh(trained_path)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Main', description = 'Prepare workspace', epilog = 'Huh')
    parser.add_argument('-m', '--models', nargs="+")
    parser.add_argument('-t', '--trained_path')

    args = parser.parse_args()
    
    main(list(args.models), args.trained_path)
    