from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import json
import os
import io
from tkinter.tix import IMAGE
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

JSON_INPUT='./bees.json'
CSV_INPUT='./bees.csv'
PBTXT_INPUT='./bees.pbtxt'
IMAGE_DIR='./used'
OUTPUT_PATH='./bees.tfrecord'

def create_tf_example(group, path, class_dict):
   with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group['file_name'])), 'rb') as fid:
      encoded_jpg = fid.read()

   width = group["width"]
   height = group["height"]

   filename = group['file_name'].encode('utf8')
   image_format = b'jpg'
   xmins = []
   xmaxs = []
   ymins = []
   ymaxs = []
   classes_text = []
   classes = []

   for row in group['annotations']: 
      xmin = row['bbox']['xmin']
      xmax = row['bbox']['xmax']
      ymin = row['bbox']['ymin']
      ymax = row['bbox']['ymax']
      
      xmins.append(xmin)
      xmaxs.append(xmax)
      ymins.append(ymin)
      ymaxs.append(ymax)

      if len(row['classes']) > 0:  
        class_name = row['classes'][0]

        classes_text.append(str(class_name).encode('utf8'))
        classes.append(class_dict[str(class_name)])

   tf_example = tf.train.Example(features=tf.train.Features(
       feature={
           'image/height': dataset_util.int64_feature(height),
           'image/width': dataset_util.int64_feature(width),
           'image/filename': dataset_util.bytes_feature(filename),
           'image/source_id': dataset_util.bytes_feature(filename),
           'image/encoded': dataset_util.bytes_feature(encoded_jpg),
           'image/format': dataset_util.bytes_feature(image_format),
           'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
           'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
           'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
           'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
           'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
           'image/object/class/label': dataset_util.int64_list_feature(classes), }))
   return tf_example


def class_dict_from_pbtxt(pbtxt_path):
   # open file, strip \n, trim lines and keep only
   # lines beginning with id or display_name

   with open(pbtxt_path, 'r', encoding='utf-8-sig') as f:
      data = f.readlines()

   name_key = None
   if any('display_name:' in s for s in data):
      name_key = 'display_name:'
   elif any('name:' in s for s in data):
      name_key = 'name:'

   if name_key is None:
      raise ValueError(
          "label map does not have class names, provided by values with the 'display_name' or 'name' keys in the contents of the file"
      )

   data = [l.rstrip('\n').strip() for l in data if 'id:' in l or name_key in l]

   ids = [int(l.replace('id:', '')) for l in data if l.startswith('id')]
   names = [
       l.replace(name_key, '').replace('"', '').replace("'", '').strip() for l in data
       if l.startswith(name_key)]

   # join ids and display_names into a single dictionary
   class_dict = {}
   for i in range(len(ids)):
      class_dict[names[i]] = ids[i]

   return class_dict


def create_record_json(JSON_INPUT, IMAGE_DIR, OUTPUT_PATH, PBTXT_INPUT):
   class_dict = class_dict_from_pbtxt(PBTXT_INPUT)

   writer = tf.compat.v1.python_io.TFRecordWriter(OUTPUT_PATH)
   path = os.path.join(IMAGE_DIR)
   f = open(JSON_INPUT)
   
   examples_json = json.load(f)

   for group in examples_json:
      tf_example = create_tf_example(group, path, class_dict)
      writer.write(tf_example.SerializeToString())


   writer.close()
   output_path = os.path.join(os.getcwd(), OUTPUT_PATH)
   print('Successfully created the JSON TFRecords: {}'.format(output_path))