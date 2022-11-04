def save_to_file(path, str):
    with open(path, 'w') as file:
        file.write(str)

def models_generate_sh(model_names):
    str = '#!/bin/bash'
    for name in model_names:
        str += f"\npython3 model_generate.py {name}"

    save_to_file('./out/generate.sh', str)
        

def inference_graph_sh(model_names):
    str = '#!/bin/bash'
    str += '\ncd ./models/research/object_detection\n'

    for name in model_names:
        str += f'''\npython3 exporter_main_v2.py \
            --trained_checkpoint_dir=../../../out/trained_{name} \
            --pipeline_config_path=../../../out/{name}.config \
            --output_directory ../../../out/interference_{name}'''
    save_to_file('./out/inference.sh', str)


#ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8