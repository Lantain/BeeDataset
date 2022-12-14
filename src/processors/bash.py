import os

def save_to_file(path, str):
    with open(path, 'w') as file:
        file.write(str)

def inference_graph_str(model, out_dir):
    str = '#!/bin/bash'
    str += f'\ncd {os.getcwd()}/models/research/object_detection\n'
    str += f'''
    python3 exporter_main_v2.py \
        --trained_checkpoint_dir={out_dir}/{model}/trained \
        --pipeline_config_path={out_dir}/{model}/pipeline.config \
        --output_directory={out_dir}/{model}/inference'''
    return str

def train_str(model, out_dir):
    str = f'''#!/bin/bash
        cd {os.getcwd()}/models/research/object_detection
        rm -rf {out_dir}/{model}
        mkdir {out_dir}/{model}

        python model_main_tf2.py \
            --model_dir={out_dir}/{model} \
            --num_train_steps=$1 \
            --pipeline_config_path={out_dir}/{model}/pipeline.config \
            --trained_checkpoint_dir={out_dir}/{model}
            --alsologtostderr'''
    return str

def save_str(model, out_dir):
    str = f'''#!/bin/bash
        export LAST_SAVED_PATH=$1/{model}-$(date '+%m%d%H%M')
        mkdir -p $LAST_SAVED_PATH
        cp -r {out_dir}/{model}/* $LAST_SAVED_PATH
        echo $LAST_SAVED_PATH'''
    return str

def inference_graph_sh(model, out_dir):
    str = inference_graph_str(model, out_dir)
    save_to_file(f'{out_dir}/{model}/inference.sh', str)

def inference_graph_env_sh(out_dir):
    str = inference_graph_str('$MODEL', out_dir)
    save_to_file(f'./out/models/inference.sh', str)

def train_sh(model, out_dir):
    str = train_str(model, out_dir)
    save_to_file(f'{out_dir}/{model}/train.sh', str)

def train_env_sh(out_dir):
    str = train_str('$MODEL', out_dir)
    save_to_file(f'./out/models/train.sh', str)

def save_sh(model, out_dir):
    str = save_str(model, out_dir)
    save_to_file(f'{out_dir}/{model}/save.sh', str)

def save_env_sh(out_dir):
    str = save_str('$MODEL', out_dir)
    save_to_file(f'./out/models/save.sh', str)