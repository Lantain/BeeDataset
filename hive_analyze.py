import src.processors.config as config_processor
import detect as validate
import argparse
import shutil
import os
import re
import tensorflow as tf
import json
import datetime
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze trained hive',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--hive', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--images', nargs="+")
    
    args = parser.parse_args()
    print(f"Images {args.images}")
    hive = args.hive
    name = os.path.basename(hive)
    model_dir = f"{args.out}/{name.replace('.hive', '')}"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    shutil.unpack_archive(hive, model_dir, 'zip')
    with open(f"{model_dir}/config.json", 'r', encoding='UTF8') as f:
        config = json.load(f)

    checkpoints_dir = f"{model_dir}/trained"
    pipeline_config = f"{model_dir}/{config['model']}/pipeline.config"

    files = os.listdir(checkpoints_dir)
    filtered = []
    for f in files:
        if re.search('ckpt-[0-9]{1,3}\.i.+', f):
            filtered.append(f.replace('.index', ''))

    filtered.sort()
    
    
    for ckpt in filtered:
        print(f"Processing {ckpt}")
        pipeline_ckpt_dir = f"{model_dir}/{ckpt}"
        ckpt_pipeline =  f"{pipeline_ckpt_dir}/pipeline.config"
        os.mkdir(pipeline_ckpt_dir)

        shutil.copy(pipeline_config, ckpt_pipeline)
        # shutil.copy(f"{pipeline_ckpt_dir}/{ckpt}.data-00000-of-00001", pipeline_ckpt_dir)
        # shutil.copy(f"{pipeline_ckpt_dir}/{ckpt}.index", pipeline_ckpt_dir)

        config_processor.set_checkpoint_value(
            f"{pipeline_ckpt_dir}/pipeline.config", 
            f"{checkpoints_dir}/{ckpt}", 
            pipeline_ckpt_dir
        )
        config_processor.export_inference_graph(
            ckpt_pipeline, 
            checkpoints_dir, 
            f"{pipeline_ckpt_dir}",
            ckpt
        )
    analisys = list()
    for ckpt in filtered:
        print(f"Analyzing {ckpt}")
        ckpt_dir = f"{model_dir}/{ckpt}"
        saved_model_dir = f"{ckpt_dir}/saved_model"
        model_fn = validate.get_model(saved_model_dir)
        
        img_detections_set = list()
        i = 0
        diffs = list()
        for img in list(args.images):
            name = os.path.basename(img)
            print(f"Processing {i} - {name}")
            start = datetime.datetime.now()
            detections = validate.get_detections(model_fn, img)
            img_detections = validate.get_processed_image(detections, f"{model_dir}/labels.pbtxt", img)
            img_detections_set.append(img_detections)
            end = datetime.datetime.now()
            diff = (start - end).microseconds / 1000
            print(f"Time diff: {diff} millis")
            diffs.append({ 'img': name, 'time': diff })
            
            img = cv2.imread(img, 3)
            b,g,r = cv2.split(img)           # get b, g, r
            rgb_img = cv2.merge([r,g,b])     # switch it to r, g, b

            fig = plt.figure(figsize=(20, 14))
            fig.suptitle(name, fontsize=16)

            rows = 1
            columns = 2

            fig.add_subplot(rows, columns, 1)
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.title("Original")

            fig.add_subplot(rows, columns, 2)
            plt.imshow(img_detections)
            plt.axis('off')
            plt.title("Detections")

            print(f"Saving {model_dir}/{ckpt}--{name}.png")
            plt.savefig(f"{model_dir}/{ckpt}--{name}.png")
            plt.close()

        analisys.append({
            'items': diffs,
            'name': ckpt
        })

    with open(f"{model_dir}/analyze.json", 'w', encoding='UTF8') as f:
        json.dumps({ 'data': analisys })

