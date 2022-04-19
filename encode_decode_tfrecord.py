import tensorflow as tf
import re
import os
import orjson
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm




feature_description = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "height": tf.io.FixedLenFeature([], tf.string),
    "width": tf.io.FixedLenFeature([], tf.string),
    "bbox_amount": tf.io.FixedLenFeature([], tf.string),
    "class": tf.io.FixedLenFeature([], tf.string),
    "xmax": tf.io.FixedLenFeature([], tf.string),
    "xmin": tf.io.FixedLenFeature([], tf.string),
    "ymax": tf.io.FixedLenFeature([], tf.string),
    "ymin": tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def serialize_example(feature0, feature1,feature2, feature3,feature4, feature5,feature6, feature7,feature8):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """


    
    feature = {
        "id":  _bytes_array(feature0),
        "height":  _bytes_array(feature1),
        "width": _bytes_array(feature2),
        "bbox_amount": _bytes_array(feature3),
        "class":  _bytes_array(feature4),
        "xmax":  _bytes_array(feature5),
        "xmin":  _bytes_array(feature6),
        "ymax":  _bytes_array(feature7),
        "ymin":  _bytes_array(feature8),
    }
    # Create a Features message using tf.train.Example.


    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _bytes_array(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.numpy()]))


def np_to_tf_tensor(nparray, dtype = tf.int32):
    tf_mat = tf.convert_to_tensor(nparray, dtype=dtype)
    return tf_mat

def pretrain_tfrecord_generation(tfr_fname, id_array, height_array,width_array,bbox_amount_array, class_array,xmax_array,xmin_array,ymax_array,ymin_array):
    # load data2
    with tf.io.TFRecordWriter(tfr_fname) as writer:
        data_length = len(id_array)
        image_root_path = "/home/data_c/finnweng/coco/train2017/"
        # image_root_path = "/home/workspace/yolov2/train2017/"
        for i in range(data_length):
        # for i in range(100):
            print("data_generate_step:",i)


            # id_tensor = np_to_tf_tensor(id_array[i], tf.int32)
            id_tensor = tf.convert_to_tensor(image_root_path + str(id_array[i]).zfill(12)+".jpg",dtype=tf.string)
            height_tensor = np_to_tf_tensor(height_array[i], tf.int16)
            width_tensor = np_to_tf_tensor(width_array[i], tf.int16)
            bbox_amount_tensor = np_to_tf_tensor(bbox_amount_array[i], tf.int16)
            class_tensor = np_to_tf_tensor(class_array[i], tf.uint8)
            xmax_tensor = np_to_tf_tensor(xmax_array[i], tf.float32)
            xmin_tensor = np_to_tf_tensor(xmin_array[i], tf.float32)
            ymax_tensor = np_to_tf_tensor(ymax_array[i], tf.float32)
            ymin_tensor = np_to_tf_tensor(ymin_array[i], tf.float32)

            id_tensor  = tf.io.serialize_tensor(id_tensor)
            height_tensor  = tf.io.serialize_tensor(height_tensor)
            width_tensor  = tf.io.serialize_tensor(width_tensor)
            bbox_amount_tensor = tf.io.serialize_tensor(bbox_amount_tensor)
            class_tensor  = tf.io.serialize_tensor(class_tensor)
            xmax_tensor  = tf.io.serialize_tensor(xmax_tensor)
            xmin_tensor   = tf.io.serialize_tensor(xmin_tensor)
            ymax_tensor   = tf.io.serialize_tensor(ymax_tensor)
            ymin_tensor   = tf.io.serialize_tensor(ymin_tensor)


            example = serialize_example(id_tensor, height_tensor, width_tensor, bbox_amount_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor)
            writer.write(example)

    # load tfrecord
    raw_dataset = tf.data.TFRecordDataset(tfr_fname)

    parsed_dataset = raw_dataset.map(_parse_function)
    for parsed_record in parsed_dataset.take(2):
        height_tensor = tf.io.parse_tensor(parsed_record["height"],out_type = tf.int16)
        class_tensor = tf.io.parse_tensor(parsed_record["class"],out_type = tf.uint8)
    
    # cv2.imshow('image',tf.cast(img, tf.uint8).numpy())
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

def deserialized(input_dict):


    # id_tensor = tf.io.parse_tensor(input_dict["id"],out_type = tf.int32)
    id_tensor = tf.io.parse_tensor(input_dict["id"],out_type = tf.string)
    
    height_tensor = tf.io.parse_tensor(input_dict["height"],out_type = tf.int16)
    width_tensor  = tf.io.parse_tensor(input_dict["width"],out_type = tf.int16)
    bbox_amount_tensor  = tf.io.parse_tensor(input_dict["bbox_amount"],out_type = tf.int16)
    class_tensor  = tf.io.parse_tensor(input_dict["class"],out_type = tf.uint8)
    xmax_tensor  = tf.io.parse_tensor(input_dict["xmax"],out_type = tf.float32)
    xmin_tensor   = tf.io.parse_tensor(input_dict["xmin"],out_type = tf.float32)
    ymax_tensor   = tf.io.parse_tensor(input_dict["ymax"],out_type = tf.float32)
    ymin_tensor   = tf.io.parse_tensor(input_dict["ymin"],out_type = tf.float32)

    
    return id_tensor, height_tensor, width_tensor, bbox_amount_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor