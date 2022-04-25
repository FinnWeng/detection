import numpy as np
import tensorflow as tf

import orjson

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image

import cv2
import math
import os
import pandas as pd
from functools import partial
import time

from keras.utils import tf_utils
from keras.utils import io_utils

from kmeans_utils import kmeans, get_best_bbox_setting
from encode_decode_tfrecord import pretrain_tfrecord_generation, _parse_function, deserialized
from preprocess_utils import  tf_load_image, tf_resize_image, tf_crop_and_resize_image, batch_data_preprocess_v3, random_flip,image_only_aug
from utils import plot_image_with_grid_cell_partition, plot_grid, OutputRescaler, find_high_class_probability_bbox, nonmax_suppression, draw_boxes
# from loss_utils import get_cell_grid, custom_loss, yolov3_custom_loss
from loss_utils import yolov3_custom_loss



import model_config

# from net.detection import Detection_Net, YOLOV2_Net
from net.detection import Detection_Net, YOLOV3_Net, Swin_Encoder, Swin_YOLOV3_Net, Decoder_Net
from gradient_accumulating_model import Gradient_Accumulating_Model







class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    config = AttrDict()
    # config.shuffle_buffer = 1000
    config.shuffle_buffer = 100
    config.batch_size = 64
    config.base_lr = 1e-4
    # config.end_lr = 1e-5
    config.end_lr = 0
    config.warmup_steps = 3000
    config.epochs = 100
    config.log_dir = "./great_batch_tf_log/"
    config.model_path = './model/detection.ckpt'
    config.tfr_fname = "./coco_without_img.tfrecord"
    config.box_buffer = 100
    config.image_root_path = "/home/data_c/finnweng/coco/train2017/"

    # anchors = tf.convert_to_tensor(
    #         np.array([0.04076599, 0.061204,
    #                 0.69486092, 0.74961789,
    #                 0.13173388, 0.181611,
    #                 0.48747882 ,0.29158246,
    #                 0.22299274, 0.49581151]), dtype=tf.float32)

    anchors = tf.convert_to_tensor(
        np.array([1.25,1.625, 2.0,3.75, 4.125,2.875, 
                1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 
                3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]), dtype=tf.float32) # for anchors1: 1/8, for anchors2: 1/16, for anchors3: 1/32

    anchors = tf.reshape(anchors, [3,3,2])

    


    

    anchor_zeros = tf.zeros(anchors.shape)
    config.zero_and_anchors = tf.concat([anchor_zeros, anchors], axis = -1) # [3,3,4]

    config.strides= [8, 16, 32]
    
    config.IMAGE_H = 224
    config.IMAGE_W = 224
    config.GRID_H = config.IMAGE_H//(32)
    config.GRID_W = config.IMAGE_W//(32)
    config.num_of_labels = 80 # there're missing id in json_data["categories"]
    # config.BOX = int(len(config.anchors)/2)
    config.BOX = anchors.shape[1]

    anchors = anchors*config.IMAGE_H/416 # for image size now 

    config.anchors = anchors

    '''
    0 for index use for traning ( NOT +1 for 0 is for padding ANY more!!!)
    2 for ture label
    '''
    # config.cls_label = [[1, 'person', 1], [2, 'bicycle', 2], [3, 'car', 3], [4, 'motorcycle', 4], [5, 'airplane', 5], [6, 'bus', 6], [7, 'train', 7], [8, 'truck', 8], [9, 'boat', 9], [10, 'traffic light', 10], [11, 'fire hydrant', 11], [12, 'stop sign', 13], [13, 'parking meter', 14], [14, 'bench', 15], [15, 'bird', 16], [16, 'cat', 17], [17, 'dog', 18], [18, 'horse', 19], [19, 'sheep', 20], [20, 'cow', 21], [21, 'elephant', 22], [22, 'bear', 23], [23, 'zebra', 24], [24, 'giraffe', 25], [25, 'backpack', 27], [26, 'umbrella', 28], [27, 'handbag', 31], [28, 'tie', 32], [29, 'suitcase', 33], [30, 'frisbee', 34], [31, 'skis', 35], [32, 'snowboard', 36], [33, 'sports ball', 37], [34, 'kite', 38], [35, 'baseball bat', 39], [36, 'baseball glove', 40], [37, 'skateboard', 41], [38, 'surfboard', 42], [39, 'tennis racket', 43], [40, 'bottle', 44], [41, 'wine glass', 46], [42, 'cup', 47], [43, 'fork', 48], [44, 'knife', 49], [45, 'spoon', 50], [46, 'bowl', 51], [47, 'banana', 52], [48, 'apple', 53], [49, 'sandwich', 54], [50, 'orange', 55], [51, 'broccoli', 56], [52, 'carrot', 57], [53, 'hot dog', 58], [54, 'pizza', 59], [55, 'donut', 60], [56, 'cake', 61], [57, 'chair', 62], [58, 'couch', 63], [59, 'potted plant', 64], [60, 'bed', 65], [61, 'dining table', 67], [62, 'toilet', 70], [63, 'tv', 72], [64, 'laptop', 73], [65, 'mouse', 74], [66, 'remote', 75], [67, 'keyboard', 76], [68, 'cell phone', 77], [69, 'microwave', 78], [70, 'oven', 79], [71, 'toaster', 80], [72, 'sink', 81], [73, 'refrigerator', 82], [74, 'book', 84], [75, 'clock', 85], [76, 'vase', 86], [77, 'scissors', 87], [78, 'teddy bear', 88], [79, 'hair drier', 89], [80, 'toothbrush', 90]]
    config.cls_label = [[0, 'person', 1], [1, 'bicycle', 2], [2, 'car', 3], [3, 'motorcycle', 4], [4, 'airplane', 5], [5, 'bus', 6], [6, 'train', 7], [7, 'truck', 8], [8, 'boat', 9], [9, 'traffic light', 10], [10, 'fire hydrant', 11], [11, 'stop sign', 13], [12, 'parking meter', 14], [13, 'bench', 15], [14, 'bird', 16], [15, 'cat', 17], [16, 'dog', 18], [17, 'horse', 19], [18, 'sheep', 20], [19, 'cow', 21], [20, 'elephant', 22], [21, 'bear', 23], [22, 'zebra', 24], [23, 'giraffe', 25], [24, 'backpack', 27], [25, 'umbrella', 28], [26, 'handbag', 31], [27, 'tie', 32], [28, 'suitcase', 33], [29, 'frisbee', 34], [30, 'skis', 35], [31, 'snowboard', 36], [32, 'sports ball', 37], [33, 'kite', 38], [34, 'baseball bat', 39], [35, 'baseball glove', 40], [36, 'skateboard', 41], [37, 'surfboard', 42], [38, 'tennis racket', 43], [39, 'bottle', 44], [40, 'wine glass', 46], [41, 'cup', 47], [42, 'fork', 48], [43, 'knife', 49], [44, 'spoon', 50], [45, 'bowl', 51], [46, 'banana', 52], [47, 'apple', 53], [48, 'sandwich', 54], [49, 'orange', 55], [50, 'broccoli', 56], [51, 'carrot', 57], [52, 'hot dog', 58], [53, 'pizza', 59], [54, 'donut', 60], [55, 'cake', 61], [56, 'chair', 62], [57, 'couch', 63], [58, 'potted plant', 64], [59, 'bed', 65], [60, 'dining table', 67], [61, 'toilet', 70], [62, 'tv', 72], [63, 'laptop', 73], [64, 'mouse', 74], [65, 'remote', 75], [66, 'keyboard', 76], [67, 'cell phone', 77], [68, 'microwave', 78], [69, 'oven', 79], [70, 'toaster', 80], [71, 'sink', 81], [72, 'refrigerator', 82], [73, 'book', 84], [74, 'clock', 85], [75, 'vase', 86], [76, 'scissors', 87], [77, 'teddy bear', 88], [78, 'hair drier', 89], [79, 'toothbrush', 90]]



    config.take_upper_threshold = 0.3



    config.LAMBDA_NO_OBJECT = 1.0
    config.LAMBDA_OBJECT    = 5.0
    config.LAMBDA_COORD     = 1.0
    config.LAMBDA_CLASS     = 1.0

    config.IOU_LOSS_THRESH = 0.5

    config.model_dim = 32


    return config

def define_model_config():
    config = AttrDict()
    config.mlp_dim = 512
    config.layer_num = 2
    config.state_dim = 512
    config.out_dim = 1
    config.time_steps = 30




    return config




def get_json_data(json_path):
    with open(json_path, "rb") as f:
        json_data = orjson.loads(f.read())
        annotation = json_data["annotations"]


    return json_data


def plot_bbox_on_image(img, bbox_coord_list):
    # Create figure and axes

    '''
    # img = cv2.imread(img_path)  
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # img = plot_bbox_on_image(img, [bbox])
    # img = plot_bbox_on_image(img, bboxs)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('./check_bbox.jpg', img)
    '''


    
    for bbox_coord in bbox_coord_list:
        """
        bbox = [x,y, w,h] the origin is upper-left corner.
        """
        # Create a Rectangle patch 
        x,y, w,h = bbox_coord
        x,y, w,h = math.trunc(x),math.trunc(y),math.trunc(w),math.trunc(h)
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 5)

    # cv2.imwrite("img_with_bbox.png",img)

    return img



def plot_bbox_on_image2(one_train_data):
    
    img,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor = one_train_data

    def plot_bbox_on_image2(img, xmax, xmin, ymax, ymin):
        # Create figure and axes


        for j in range(100):
            """
            bbox = [x,y, w,h] the origin is upper-left corner.
            """
            
            cv2.rectangle(img, (math.trunc(float(xmin[j])), math.trunc(float(ymin[j]))), (math.trunc(float(xmax[j])), math.trunc(float(ymax[j]))), (255,0,0), 2)

        # cv2.imwrite("img_with_bbox.png",img)

        return img


    for i in range(config.batch_size):
        this_img = plot_bbox_on_image2(img[i], xmax_tensor[i], xmin_tensor[i], ymax_tensor[i], ymin_tensor[i])
        this_img = cv2.cvtColor(this_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./tf_check_bbox_{}.jpg'.format(i), this_img)


def check_all_img_exists():
    annotation_path = "/home/data_c/finnweng/coco/annotations/instances_train2017.json"
    image_root_path = "/home/data_c/finnweng/coco/train2017/"

    json_data = get_json_data(annotation_path)
    annotation = json_data["annotations"]



    '''
    check by annnotations
    '''
    # img_path = image_root_path + str(json_data["annotations"][0]["image_id"]).zfill(12)+".jpg"
    # bbox = json_data["annotations"][0]['bbox']

    '''
    check by images
    '''

    # img_path = image_root_path + str(json_data["images"][0]['id']).zfill(12)+".jpg"
    bboxs = [  anno["bbox"]  for anno in json_data["annotations"] if anno["image_id"] == json_data["images"][0]['id']]
    # print(bboxs)



    anno_list = json_data["annotations"]
    img_info_list = json_data["images"]

    

    image_bbox_list = []



    print("now sorting...")
    anno_list = sorted(anno_list, key=lambda anno: anno["image_id"])
    img_info_list = sorted(img_info_list, key=lambda img_info: img_info["id"])

    anno_by_image_id_list = []
    this_id = 0
    this_anno_by_image_id_list = []
    no_missing_img = True
    for anno in anno_list:
        print("this_id:",this_id)
        if anno["image_id"] == this_id:
            # this_anno_by_image_id_list.append(anno)
            essential_anno = {'bbox':anno['bbox'],'category_id':anno['category_id']}
            this_anno_by_image_id_list.append(essential_anno)
        else:
            if len(this_anno_by_image_id_list) >0:
                anno_by_image_id_list.append(this_anno_by_image_id_list)
            this_anno_by_image_id_list = []
            this_id = anno["image_id"]
            # img_info = [  img_info for img_info in img_info_list if img_info['id'] == this_id]

            while True:
                if img_info_list[0]['id'] == this_id:
                    img_info = img_info_list[0]
                    essential_img_info = {'height':img_info['height'],'width':img_info['width']}
                    break
                elif img_info_list[0]['id'] < this_id:
                    img_info_list.pop(0)
                else:
                    assert False, "order of anno and image not the same!!"

            
            this_anno_by_image_id_list.append(essential_img_info)
            essential_anno = {'bbox':anno['bbox'],'category_id':anno['category_id']}
            this_anno_by_image_id_list.append(essential_anno)
            


        print(len(anno_by_image_id_list))
        img_path = image_root_path + str(this_id).zfill(12)+".jpg"

        img_existing = os.path.isfile(img_path)
        no_missing_img = no_missing_img&img_existing
    
    print("len(img_info_list)", len(img_info_list))
    print("no_missing_img:",no_missing_img)



def get_anno_by_image_id_list(json_data):
    anno_list = json_data["annotations"]
    img_info_list = json_data["images"]

    print("now sorting...")
    anno_list = sorted(anno_list, key=lambda anno: anno["image_id"])
    img_info_list = sorted(img_info_list, key=lambda img_info: img_info["id"])

    anno_by_image_id_list = []
    this_id = 0
    this_anno_by_image_id_list = []
    no_missing_img = True
    for anno in anno_list:
        print("this_id:",this_id)
        if anno["image_id"] == this_id:
            '''
            collect all same image_id anno
            '''
            essential_anno = {"bbox":anno["bbox"],"category_id":anno["category_id"]}
            this_anno_by_image_id_list.append(essential_anno)
        else:
            '''
            switch this_id
            '''
            if len(this_anno_by_image_id_list) >0: # avoid this_id = 0
                anno_by_image_id_list.append(this_anno_by_image_id_list) # acturally put info into anno_by_image_id_list
            this_anno_by_image_id_list = []
            this_id = anno["image_id"]

            '''
            collect id = image_id img info
            '''
            while True:
                if img_info_list[0]["id"] == this_id:
                    img_info = img_info_list[0]
                    essential_img_info = {"height":img_info["height"],"width":img_info["width"], "id":this_id}
                    break
                elif img_info_list[0]["id"] < this_id:
                    img_info_list.pop(0)
                else:
                    assert False, "order of anno and image not the same!!"

            
            this_anno_by_image_id_list.append(essential_img_info)
            essential_anno = {"bbox":anno["bbox"],"category_id":anno["category_id"]}
            this_anno_by_image_id_list.append(essential_anno)
        

        print(len(anno_by_image_id_list))
        img_path = image_root_path + str(this_id).zfill(12)+".jpg"
        img_existing = os.path.isfile(img_path)
        no_missing_img = no_missing_img&img_existing


    if len(this_anno_by_image_id_list) >0:
        anno_by_image_id_list.append(this_anno_by_image_id_list) # acturally put info into anno_by_image_id_list
        print("save the last id!!")

    '''
    check correctness
    '''

    print("len(img_info_list)", len(img_info_list))
    print("no_missing_img:",no_missing_img)

    return anno_by_image_id_list


def make_data_list_for_kmeans(anno_by_image_id_list):

    '''
    post process for make data we want
    '''

    data_list = []
    max_len = 0
    for anno_by_image_id in anno_by_image_id_list:
        this_data = {}
        this_img_info = anno_by_image_id[0]
        this_bboxs = anno_by_image_id[1:]

        """
        bbox = [x,y, w,h] the origin is upper-left corner.
        """
        this_bboxs = [ {"class":bbox["category_id"], 
                        "xmax":bbox['bbox'][0]+bbox['bbox'][2], 
                        "xmin":bbox['bbox'][0],
                        "ymax":bbox['bbox'][1]+bbox['bbox'][3], 
                        "ymin":bbox['bbox'][1]} for bbox in anno_by_image_id[1:]]

        if len(this_bboxs)>max_len:
            max_len = len(this_bboxs)


        this_data["id"] = this_img_info["id"]
        print(this_img_info)
        this_data["height"] = this_img_info["height"]
        this_data["width"] = this_img_info["width"]

        this_data["bbox"] = this_bboxs

        data_list.append(this_data)
        print("len(data_list):",len(data_list))
    print("max_len:",max_len) # max_len: 93
    
    return data_list

def make_data_list_for_tfrecord(anno_by_image_id_list, config):

    '''
    post process for make data we want
    Do not save img to tfrecord since they have different size
    '''

    data_list = []
    max_len = 0
    for anno_by_image_id in anno_by_image_id_list:
        this_data = {}
        this_img_info = anno_by_image_id[0]
        this_bboxs = anno_by_image_id[1:]
        
        """
        bbox = [x,y, w,h] the origin is upper-left corner.
        """
        # this_bboxs = [ {"class":bbox["category_id"], 
        #                 "xmax":bbox['bbox'][0]+bbox['bbox'][2], 
        #                 "xmin":bbox['bbox'][0],
        #                 "ymax":bbox['bbox'][1]+bbox['bbox'][3], 
        #                 "ymin":bbox['bbox'][1]} for bbox in anno_by_image_id[1:]]

        class_base = np.zeros([config.box_buffer])
        xmax_base = np.zeros([config.box_buffer])
        xmin_base = np.zeros([config.box_buffer])
        ymax_base = np.zeros([config.box_buffer])
        ymin_base = np.zeros([config.box_buffer])

        for idx, value in enumerate(this_bboxs):
            # class_base[idx] = value["category_id"]
            class_base[idx] = [item for item in config.cls_label if item[2] == value["category_id"]][0][0]
            this_bbox =  value['bbox']
            xmax_base[idx] = this_bbox[0] + this_bbox[2]
            xmin_base[idx] = this_bbox[0]
            ymax_base[idx] = this_bbox[1] + this_bbox[3]
            ymin_base[idx] = this_bbox[1]
        


        this_data["id"] = this_img_info["id"]
        this_data["height"] = this_img_info["height"]
        this_data["width"] = this_img_info["width"]
        this_data["bbox_amount"] = len(this_bboxs)

        this_data["class"] = class_base
        this_data["xmax"] = xmax_base
        this_data["xmin"] = xmin_base
        this_data["ymax"] = ymax_base
        this_data["ymin"] = ymin_base

        data_list.append(this_data)
    
    data_df = pd.DataFrame(data_list)

    id_array = data_df["id"].to_numpy()
    height_array = data_df["height"].to_numpy()
    width_array = data_df["width"].to_numpy()
    bbox_amount_array = data_df["bbox_amount"].to_numpy()


    class_array = np.stack(data_df["class"].to_numpy())
    xmax_array = np.stack(data_df["xmax"].to_numpy())
    xmin_array = np.stack(data_df["xmin"].to_numpy())
    ymax_array = np.stack(data_df["ymax"].to_numpy())
    ymin_array = np.stack(data_df["ymin"].to_numpy())
    
    return id_array, height_array,width_array,bbox_amount_array, class_array,xmax_array,xmin_array,ymax_array,ymin_array


def make_tfrecord(annotation_path):
    json_data = get_json_data(annotation_path)
    # import pdb
    # pdb.set_trace()

    '''
    json_data['categories']: (1,2,...90)
    max bbox num = 93

    #  5 clusters: mean IoU = 0.4942
    #     [[0.04076599 0.061204  ]
    #     [0.69486092 0.74961789]
    #     [0.13173388 0.181611  ]
    #     [0.48747882 0.29158246]
    #     [0.22299274 0.49581151]]
    '''



    anno_by_image_id_list = get_anno_by_image_id_list(json_data)


    '''
    make tfrecord
    
    '''

    id_array, height_array,width_array,bbox_amount_array, class_array,xmax_array,xmin_array,ymax_array,ymin_array = make_data_list_for_tfrecord(anno_by_image_id_list,config)
    print("id_array.shape:", id_array.shape)
    print("xmax_array.shape:",xmax_array.shape)

    pretrain_tfrecord_generation(config.tfr_fname, id_array, height_array,width_array,bbox_amount_array, class_array,xmax_array,xmin_array,ymax_array,ymin_array)

def get_proper_anchors_by_kmeans(annotation_path):
    json_data = get_json_data(annotation_path)
    # import pdb
    # pdb.set_trace()

    '''
    json_data['categories']: (1,2,...90)
    max bbox num = 93

    #  5 clusters: mean IoU = 0.4942
    #     [[0.04076599 0.061204  ]
    #     [0.69486092 0.74961789]
    #     [0.13173388 0.181611  ]
    #     [0.48747882 0.29158246]
    #     [0.22299274 0.49581151]]
    '''



    anno_by_image_id_list = get_anno_by_image_id_list(json_data)


    '''
    get best_bbox_setting(run only once)
    '''
    data_list = make_data_list_for_kmeans(anno_by_image_id_list)
    get_best_bbox_setting(data_list)





class Wrapping_Loss(tf.keras.losses.Loss):

    def __init__(self, config, loss_fn):
        super(Wrapping_Loss, self).__init__()
        self.config = config
        self.loss_fn = loss_fn

    def call(self, y_true, true_boxes, y_pred):
        loss = self.loss_fn(self.config, y_true, true_boxes, y_pred)
        return loss

class Gradient_Accumulating_Custom_Model(tf.keras.Model):
    def __init__(self, config, inputs, outputs,n_gradients):
        super( Gradient_Accumulating_Custom_Model, self).__init__(inputs,outputs)
        self.config = config

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]


    def compile(self, optimizer, loss_fn):
        super( Gradient_Accumulating_Custom_Model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def apply_accu_gradients(self):
        '''
        For solution of https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras,
        there's a obvious mistake that it simply assign_add the gradients and will make it self.n_gradients times greater(maximum case).
        So I feel it should be mean of gradients, not sum of gradient.
        '''
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(self.gradient_accumulation[i]/tf.cast(self.n_gradients, tf.float32))

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))
        print("accumulated Gradient applied!")

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


    def train_step(self, data):
        x, y = data
        x_batch = x["x"]
        # y_batch, b_batch = y["y"], y["true_boxes"]
        '''
        For each of small_y, middle_y, large_y
        y_batch:
        center_x, center_y, center_w, center_h, this is box or not(update_index_indicator)
        '''
        y_batch, b_batch = [y["small_y"], y["middle_y"],y["large_y"]], [y["small_true_boxes"], y["middle_true_boxes"],y["large_true_boxes"]]

        with tf.GradientTape() as tape:
            y_pred_small_bbox, y_pred_middle_bbox, y_pred_large_bbox = self(x_batch)  # Forward pass
            y_pred = [y_pred_small_bbox,y_pred_middle_bbox,y_pred_large_bbox]
            # print("y_pred:",y_pred[0].shape)
            # Compute the loss valuese
            # (the loss function is configured in `compile()`)
            # loss = self.loss_fn(self.config, y_batch,b_batch, y_pred)
            loss, giou_loss, conf_loss,  prob_loss, coord_loss = self.loss_fn(self.config, y_batch,b_batch, y_pred)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: print("accum step:", self.n_acum_step))


        # Update weights
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        result = {"loss":loss, "giou_loss":giou_loss, "conf_loss":conf_loss,  "prob_loss":prob_loss, "coord_loss":coord_loss}
        return result


class Warmup_Cos_Decay_Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, cos_initial_learning_rate, warmup_steps, cos_decay_steps, alpha = 0):
        self.cos_initial_learning_rate = tf.cast(cos_initial_learning_rate, tf.float32)
        self.warmup_steps = warmup_steps
        self.cos_decay_steps = tf.cast(cos_decay_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)

        '''
        there's only one step
        the step of warm up and decay step counts separatly.
        when step > warmup step, switch to cos decay model
        '''
    
    def decayed_learning_rate(self, step):
        inner_step = tf.math.minimum(tf.cast(step - self.warmup_steps,  tf.float32), self.cos_decay_steps) # here i deal with problem of step that count warm up step.
        inner_step = tf.cast(inner_step, tf.float32)
        cosine_decay = 0.5 * (1 + tf.math.cos(tf.constant(math.pi, dtype=tf.float32) * inner_step /  self.cos_decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.cos_initial_learning_rate * decayed

    def __call__(self, step):
        
        # if step <= self.warmup_steps:
        #     lr = self.cos_initial_learning_rate*(step/self.warmup_steps)
        # else:
        #     lr = self.decayed_learning_rate(step)
        
        con_1 = self.cos_initial_learning_rate*(tf.cast(step, tf.float32)/tf.cast(self.warmup_steps, tf.float32))
        con_2 = self.decayed_learning_rate(step)
        # lr = tf.where(step <= self.warmup_steps, self.cos_initial_learning_rate*(step/self.warmup_steps), self.decayed_learning_rate(step))
        lr = tf.where(step <= self.warmup_steps, con_1, con_2)
             
        return lr



class Custom_Save_Model_Callback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
            options=None,
            initial_value_threshold=None,
            resume_epoch = 0,
            **kwargs):
        super(Custom_Save_Model_Callback, self).__init__(
            filepath = filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            initial_value_threshold=initial_value_threshold)
        self.resume_epoch = resume_epoch
        self.config = config
    
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch + self.resume_epoch
        # print("self._current_epoch:",self._current_epoch)
        # print("self.resume_epoch:",self.resume_epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch + self.resume_epoch, batch=None, logs=logs)



# class TerminateOn_Begin_NaN(tf.keras.callbacks.Callback):
#     """Callback that terminates training when a NaN loss is encountered.
#     """

#     def __init__(self):
#         super(TerminateOn_Begin_NaN, self).__init__()
#         self._supports_tf_logs = True

#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}
#         loss = logs.get('loss')
#         if loss is not None:
#             loss = tf_utils.sync_to_numpy_or_python_type(loss)
#             if np.isnan(loss) or np.isinf(loss):
#                 io_utils.print_msg(f'Batch {batch}: Invalid loss, terminating training')
#                 self.model.stop_training = True
    
#     def on_train_begin(self, logs=None):
#         if self.load_weights_on_restart:
#             filepath_to_load = (
#                 self._get_most_recently_modified_file_matching_pattern(self.filepath))
#             if (filepath_to_load is not None and
#                 self._checkpoint_exists(filepath_to_load)):
#                 try:
#                     # `filepath` may contain placeholders such as `{epoch:02d}`, and
#                     # thus it attempts to load the most recently modified file with file
#                     # name matching the pattern.
#                     self.model.load_weights(filepath_to_load)
#                 except (IOError, ValueError) as e:
#                     raise ValueError(
#                         f'Error loading file from {filepath_to_load}. Reason: {e}')

if __name__ == "__main__":

    # tf.data.experimental.enable_debug_mode()
    # tf.config.experimental_run_functions_eagerly(True)
    tf.debugging.enable_check_numerics()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.set_visible_devices(gpus[0], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    annotation_path = "/home/data_c/finnweng/coco/annotations/instances_train2017.json"
    image_root_path = "/home/data_c/finnweng/coco/train2017/"
    config = define_config()
    # model_config = define_model_config()

    '''
    make tfrecord
    '''

    # make_tfrecord(annotation_path)



    '''
    make dataset
    '''

    ds_train = tf.data.TFRecordDataset(config.tfr_fname)
    # ds_train = ds_train.map(_parse_function, tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(_parse_function, 10)
    ds_train = ds_train.map(deserialized,10)
    partial_tf_load_image = partial(tf_load_image, config)
    ds_train = ds_train.map(partial_tf_load_image, 10)
    # ds_train = ds_train.map(tf_resize_image, tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(tf_crop_and_resize_image, 10)
    # ds_train = ds_train.map(random_flip, 10)
    # ds_train = ds_train.map(image_only_aug,10)

    '''
    index one by one
    '''
    # partial_data_preprocess = partial(data_preprocess, config)
    # ds_train = ds_train.map(partial_data_preprocess, 10)

    '''
    batch
    '''
    partial_batch_data_preprocess = partial(batch_data_preprocess_v3, config)
    # partial_batch_data_preprocess = partial(batch_data_preprocess, config)
    ds_train = ds_train.map(partial_batch_data_preprocess, tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(config.shuffle_buffer)
    ds_train = ds_train.batch(config.batch_size, drop_remainder=True)

    
    '''
    check result of batch_data_preprocess
    # needs "return best_anchor,max_iou,x_batch, b_batch, y_batch"
    '''
    # best_anchor,max_iou, x_batch, b_batch, y_batch = next(ds_train.as_numpy_iterator())
    # plot_image_with_grid_cell_partition(x_batch, 1, config)
    # plot_grid(y_batch, 1, config)


    '''
    inspect preprocess
    '''

    one_train_data = next(ds_train.as_numpy_iterator())

    # x_batch, b_batch, y_batch = one_train_data


    # y_pred = tf.zeros([config.batch_size, config.GRID_H, config.GRID_W,config.BOX, 4 + 1 + config.num_of_labels], tf.float32)

    # loss = custom_loss(config,  y_batch,y_pred, b_batch)

    # import pdb
    # pdb.set_trace()



    '''
    define loss
    '''

    # wrapped_loss = Wrapping_Loss(config, custom_loss)

    '''
    define model
    '''
    # yolov2_net = YOLOV2_Net(config)
    # yolov3_net = YOLOV3_Net(config)
    swin_model_config = model_config.get_swin_config()

    swin_encoder = Swin_Encoder(config, \
        norm_layer=tf.keras.layers.LayerNormalization, **swin_model_config)

    detector = Decoder_Net(config)
    

    swin_yolov3_net = Swin_YOLOV3_Net(config, swin_encoder, detector)

    # build model, expose this to show how to deal with dict as fit() input
    model_input = tf.keras.Input(shape=one_train_data[0]["x"].shape[1:],name="image",dtype=tf.float32)

    # output = yolov2_net(model_input)
    y_pred_small_bbox, y_pred_middle_bbox, y_pred_large_bbox = swin_yolov3_net(model_input)

    

    custom_model =  Gradient_Accumulating_Custom_Model(inputs = [model_input],outputs = [y_pred_small_bbox, y_pred_middle_bbox, y_pred_large_bbox], config = config, n_gradients=8)
    # custom_model = Custom_Model(inputs = [model_input],outputs = [y_pred_small_bbox, y_pred_middle_bbox, y_pred_large_bbox], config = config)

    steps_per_epoch = 118287//config.batch_size
    # lr_schedule = Warmup_Cos_Decay_Schedule(config.base_lr, warmup_steps = config.warmup_steps, cos_decay_steps = steps_per_epoch*config.epochs)
    # lr_schedule = Warmup_Cos_Decay_Schedule(config.base_lr, warmup_steps = steps_per_epoch*3, cos_decay_steps = steps_per_epoch*config.epochs)
    # lr_schedule = Warmup_Cos_Decay_Schedule(config.base_lr, warmup_steps = 1, cos_decay_steps = steps_per_epoch*config.epochs)
    lr_schedule = Warmup_Cos_Decay_Schedule(config.base_lr, warmup_steps = 1, \
        cos_decay_steps = steps_per_epoch*config.epochs, alpha= config.end_lr)



    
    
    total_hist = {'loss': [], 'giou_loss': [], 'conf_loss': [], 'prob_loss': []}



    '''
    define callback
    '''
    '''
    len(total_hist["loss"]) starts from 0
    epoch starts from 1(?)
    '''
    


    # previous_epoch = str(7).zfill(4)
    # checkpoint_path = "./model/detection_cp-"+previous_epoch+"/detection.ckpt"
    # custom_model.load_weights(checkpoint_path)

    # import pdb
    # pdb.set_trace()


    custom_model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate = config.base_lr), 
        optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule, clipvalue = 0.5), 
        # loss_fn = custom_loss)
        loss_fn = yolov3_custom_loss)

    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=10, update_freq= 10)



    # save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=config.model_path,
    #     save_weights_only= True,
    #     verbose=1)

    checkpoint_path = "./model/great_batch_detection_cp-{epoch:04d}/detection.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    print('len(total_hist["loss"]):',len(total_hist["loss"]))

    save_model_callback = Custom_Save_Model_Callback(
        filepath=checkpoint_path,
        save_weights_only= True,
        verbose=1,
        resume_epoch = len(total_hist["loss"])
    )

    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    callback_list = [tensorboard_callback,save_model_callback, nan_callback]

    '''
    training
    '''

    swin_encoder.load_weights(filepath="./pretrain_weight/swin_encoder.ckpt")

    print(custom_model.summary())

    swin_encoder.trainable = False 
    print("swin_encoder.trainable = False")
    
    hist = custom_model.fit(ds_train,
            epochs=1, 
            steps_per_epoch=1000).history

    swin_encoder.trainable = True
    print("swin_encoder.trainable = True")

    hist = custom_model.fit(ds_train,
            epochs=1, 
            steps_per_epoch=steps_per_epoch,
            # validation_data = ds_val,
            # validation_steps=3,
            callbacks = callback_list).history

