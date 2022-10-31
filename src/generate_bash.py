def models_generate_sh(model_names):
    str = '#!/bin/bash'
    for name in model_names:
        str += f"python3 model_generate.py {name}"

    with open(f'./out/generate.sh', 'w') as file:
        file.write(str)

def inference_graph_sh(model_names):
    str = '#!/bin/bash'
    str += 'cd /content/models/research/object_detection'

    for name in model_names:
        str += f'''
        python3 exporter_main_v2.py \
            --trained_checkpoint_dir=/content/{name} \
            --pipeline_config_path=/content/models/research/dataset/out/{name}.config \
            --output_directory /content/interference_{name}
        '''
#ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8