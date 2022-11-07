import re
import os

def update_config_values_regex(model, values):
    config_path = f'./out/{model}/pipeline.config'
    with open(config_path) as f:
        config = f.read()
    with open(config_path, 'w') as f:
        for obj in values:
            config = re.sub(obj['regex'],  obj['value'], config)
        f.write(config)

def set_config_value(key, value, model):
    path = f"./out/{model}.config"
    with open(path) as f:
        config = f.read()

    with open(path, 'w') as f:
        config = re.sub(f'{key}: ".*?"', f'{key}: "{value}"', config)
        f.write(config)

def fill_config_defaults(model_name):
    cwd = os.getcwd().replace('\\', "\\\\");
    labelmap_path = f'{cwd}/assets/labels.pbtxt'
    fine_tune_checkpoint = f'{cwd}/out/{model_name}/checkpoint/ckpt-0'
    train_record_path = f'{cwd}/out/train/train_csv.tfrecord'
    test_record_path = f'{cwd}/out/test/test_csv.tfrecord'
    num_classes = 1
    batch_size = 32
    num_steps = 30000

    values = list([
        {
            "regex": r'label_map_path: ".*?"',
            "value": 'label_map_path: "{}"'.format(labelmap_path)
        },
        {
            "regex": r'fine_tune_checkpoint: ".*?"',
            "value": 'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint)
        },
        {
            "regex": r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
            "value": 'input_path: "{}"'.format(train_record_path)
        },
        {
            "regex": r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
            "value": 'input_path: "{}"'.format(test_record_path)
        },
        {
            "regex": r'num_classes: [0-9]+',
            "value": 'num_classes: {}'.format(num_classes)
        },
        {
            "regex": r'batch_size: [0-9]+',
            "value": 'batch_size: {}'.format(batch_size)
        },
        {
            "regex": r'num_steps: [0-9]+',
            "value": 'num_steps: {}'.format(num_steps)
        },
        {
            "regex": r'fine_tune_checkpoint_type: "classification"',
            "value": 'fine_tune_checkpoint_type: "{}"'.format('detection')
        }
    ])

    update_config_values_regex(model_name, values)