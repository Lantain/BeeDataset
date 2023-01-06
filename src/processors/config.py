import re
import os
from object_detection.utils import config_util
from src.util import get_last_checkpoint_name

def update_config_values_regex(config_path, values):
    with open(config_path) as f:
        config = f.read()
    with open(config_path, 'w') as f:
        for obj in values:
            config = re.sub(obj['regex'],  obj['value'], config)
        f.write(config)

def set_config_value(key, value, model_dir):
    path = f"{model_dir}/pipeline.config"
    with open(path) as f:
        config = f.read()

    with open(path, 'w') as f:
        config = re.sub(f'{key}: ".*?"', f'{key}: "{value}"', config)
        f.write(config)

def get_train_record_path():
    cwd = os.getcwd().replace('\\', "\\\\")
    return f'{cwd}/out/train/train_csv.tfrecord'

def get_test_record_path():
    cwd = os.getcwd().replace('\\', "\\\\")
    return f'{cwd}/out/test/test_csv.tfrecord'

def get_fine_tune_checkpoint(model):
    cwd = os.getcwd().replace('\\', "\\\\")
    return f'{cwd}/out/models/{model}/checkpoint/ckpt-0'

def fill_config_defaults(checkpoint_path, pipeline_config_path):
    cwd = os.getcwd().replace('\\', "\\\\");
    labelmap_path = f'{cwd}/assets/labels.pbtxt'

    train_record_path = get_train_record_path()
    test_record_path = get_test_record_path()
    num_classes = 1
    batch_size = 32
    num_steps = 30000

    values = list([
        {
            "regex": 'label_map_path: ".*?"',
            "value": 'label_map_path: "{}"'.format(labelmap_path)
        },
        {
            "regex": 'fine_tune_checkpoint: ".*?"',
            "value": 'fine_tune_checkpoint: "{}"'.format(checkpoint_path)
        },
        {
            "regex": '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
            "value": 'input_path: "{}"'.format(train_record_path)
        },
        {
            "regex": '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
            "value": 'input_path: "{}"'.format(test_record_path)
        },
        {
            "regex": 'num_classes: [0-9]+',
            "value": 'num_classes: {}'.format(num_classes)
        },
        {
            "regex": 'batch_size: [0-9]+',
            "value": 'batch_size: {}'.format(batch_size)
        },
        {
            "regex": 'num_steps: [0-9]+',
            "value": 'num_steps: {}'.format(num_steps)
        },
        {
            "regex": 'fine_tune_checkpoint_type: "classification"',
            "value": 'fine_tune_checkpoint_type: "{}"'.format('detection')
        }
    ])
    update_config_values_regex(pipeline_config_path, values)


def set_checkpoint_value(model_dir, checkpoint_name):
    configs = config_util.get_configs_from_pipeline_file(f"{model_dir}/pipeline.config")


def fill_config(model, model_dir, labels_path, train_rec_path, test_rec_path, num_steps, batch_size):
    pipeline_config_path = f"{model_dir}/pipeline.config"
    checkpoint_name = get_last_checkpoint_name(f"{model_dir}/trained")
    checkpoint_path = get_fine_tune_checkpoint(model)

    if checkpoint_name is not None:
        checkpoint_path = f"{model_dir}/trained/{checkpoint_name}"

    fill_config_defaults(checkpoint_path, pipeline_config_path)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    configs['train_input_config'].label_map_path = labels_path
    configs['train_input_config'].tf_record_input_reader.input_path[:] = [train_rec_path]

    configs['eval_input_config'].label_map_path = labels_path
    configs['eval_input_config'].tf_record_input_reader.input_path[:] = [test_rec_path]

    configs['train_config'].num_steps = num_steps
    configs['train_config'].batch_size = batch_size
    configs['train_config'].fine_tune_checkpoint = checkpoint_path

    # if re.match('ssd_mobilenet_v2_fpnlite.+', model):
    #     configs['train_input_config'].tf_record_input_reader.input_path[:] = [get_train_record_path()]
    #     configs['eval_input_config'].tf_record_input_reader.input_path[:] = [get_test_record_path()]

    if re.match('faster_rcnn_inception_resnet.+', model):
        configs['train_config'].batch_size = 2
        configs['eval_config'].batch_size = 2

    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, model_dir)
