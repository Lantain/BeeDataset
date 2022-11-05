import os

def save_to_file(path, str):
    with open(path, 'w') as file:
        file.write(str)

def models_generate_sh(model_names):
    str = '#!/bin/bash'
    for name in model_names:
        str += f"\npython3 model_generate.py {name} $PWD"

    save_to_file('./out/generate.sh', str)
        

def inference_graph_sh(model_names):
    str = '#!/bin/bash'
    str += '\ncd ./models/research/object_detection\n'

    for name in model_names:
        str += f'''\npython3 exporter_main_v2.py \
            --trained_checkpoint_dir={os.getcwd()}/out/trained_{name} \
            --pipeline_config_path={os.getcwd()}/out/{name}.config \
            --output_directory={os.getcwd()}/out/inference_{name}'''
    save_to_file('./out/inference.sh', str)

def train_sh(model, out_dir, steps=3000):
    str = f'''#!/bin/bash
        cd {os.getcwd()}/models/research/object_detection
        rm -rf {out_dir}/{model}
        mkdir {out_dir}/{model}

        python model_main_tf2.py \
        --model_dir={out_dir}/{model} \
        --num_train_steps=$1 \
        --pipeline_config_path={os.getcwd()}/out/{model}.config \
        --alsologtostderr'''
    save_to_file(f'./out/train_{model}.sh', str)
#ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8