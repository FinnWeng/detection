import numpy as np
import tensorflow as tf
import orjson
import json

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image

import cv2
import math
import os
import pandas as pd
from functools import partial
import time

from kmeans_utils import kmeans, get_best_bbox_setting
from encode_decode_tfrecord import pretrain_tfrecord_generation, _parse_function, deserialized
from preprocess_utils import  tf_load_image, tf_resize_image, tf_crop_and_resize_image, data_preprocess, batch_data_preprocess
from utils import plot_image_with_grid_cell_partition, plot_grid, OutputRescaler, find_high_class_probability_bbox, nonmax_suppression, draw_boxes
from loss_utils import get_cell_grid, custom_loss

from net.detection import Detection_Net, YOLOV2_Net

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import copy




class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config():
    config = AttrDict()
    config.shuffle_buffer = 1000
    # config.shuffle_buffer = 100
    config.batch_size = 32
    config.base_lr = 1e-4
    config.warmup_steps = 1000
    config.epochs = 100
    config.log_dir = "./tf_log/"
    config.model_path = './model/detection.ckpt'
    config.tfr_fname = "./coco_without_img.tfrecord"
    config.box_buffer = 100
    config.image_root_path = "/home/data_c/finnweng/coco/train2017/"

    anchors = tf.convert_to_tensor(
            np.array([0.04076599, 0.061204,
                    0.69486092, 0.74961789,
                    0.13173388, 0.181611,
                    0.48747882 ,0.29158246,
                    0.22299274, 0.49581151]), dtype=tf.float32)

    

    # config.anchors = [[0, 0,anchors[2*i], anchors[2*i+1]] for i in range(int(len(anchors)//2))]
    
    anchors = tf.reshape(anchors, [-1,2])
    config.anchors = anchors

    anchor_zeros = tf.zeros(anchors.shape)
    config.zero_and_anchors = tf.concat([anchor_zeros, anchors], axis = -1)
    
    config.IMAGE_H = 224
    config.IMAGE_W = 224
    config.GRID_H = 7
    config.GRID_W = 7
    config.num_of_labels = 80 # there're missing id in json_data["categories"]
    # config.BOX = int(len(config.anchors)/2)
    config.BOX = anchors.shape[0]

    '''
    0 for index use for traning (+1 for 0 is for padding)
    2 for ture label
    '''
    config.cls_label = [[1, 'person', 1], [2, 'bicycle', 2], [3, 'car', 3], [4, 'motorcycle', 4], [5, 'airplane', 5], [6, 'bus', 6], [7, 'train', 7], [8, 'truck', 8], [9, 'boat', 9], [10, 'traffic light', 10], [11, 'fire hydrant', 11], [12, 'stop sign', 13], [13, 'parking meter', 14], [14, 'bench', 15], [15, 'bird', 16], [16, 'cat', 17], [17, 'dog', 18], [18, 'horse', 19], [19, 'sheep', 20], [20, 'cow', 21], [21, 'elephant', 22], [22, 'bear', 23], [23, 'zebra', 24], [24, 'giraffe', 25], [25, 'backpack', 27], [26, 'umbrella', 28], [27, 'handbag', 31], [28, 'tie', 32], [29, 'suitcase', 33], [30, 'frisbee', 34], [31, 'skis', 35], [32, 'snowboard', 36], [33, 'sports ball', 37], [34, 'kite', 38], [35, 'baseball bat', 39], [36, 'baseball glove', 40], [37, 'skateboard', 41], [38, 'surfboard', 42], [39, 'tennis racket', 43], [40, 'bottle', 44], [41, 'wine glass', 46], [42, 'cup', 47], [43, 'fork', 48], [44, 'knife', 49], [45, 'spoon', 50], [46, 'bowl', 51], [47, 'banana', 52], [48, 'apple', 53], [49, 'sandwich', 54], [50, 'orange', 55], [51, 'broccoli', 56], [52, 'carrot', 57], [53, 'hot dog', 58], [54, 'pizza', 59], [55, 'donut', 60], [56, 'cake', 61], [57, 'chair', 62], [58, 'couch', 63], [59, 'potted plant', 64], [60, 'bed', 65], [61, 'dining table', 67], [62, 'toilet', 70], [63, 'tv', 72], [64, 'laptop', 73], [65, 'mouse', 74], [66, 'remote', 75], [67, 'keyboard', 76], [68, 'cell phone', 77], [69, 'microwave', 78], [70, 'oven', 79], [71, 'toaster', 80], [72, 'sink', 81], [73, 'refrigerator', 82], [74, 'book', 84], [75, 'clock', 85], [76, 'vase', 86], [77, 'scissors', 87], [78, 'teddy bear', 88], [79, 'hair drier', 89], [80, 'toothbrush', 90]]

    config.cell_grid = get_cell_grid(config.GRID_W,config.GRID_H,config.batch_size,config.BOX)

    config.LAMBDA_NO_OBJECT = 1.0
    config.LAMBDA_OBJECT    = 5.0
    config.LAMBDA_COORD     = 1.0
    config.LAMBDA_CLASS     = 1.0

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



class Custom_Model(tf.keras.Model):
    def __init__(self, config, inputs, outputs):
        super(Custom_Model, self).__init__(inputs,outputs)
        self.config = config

    def compile(self, optimizer, loss_fn):
        super(Custom_Model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, y = data
        x_batch = x["x"]
        y_batch, b_batch = y["y"], y["true_boxes"]

        with tf.GradientTape() as tape:
            y_pred = self(x_batch)  # Forward pass
            # Compute the loss valuese
            # (the loss function is configured in `compile()`)
            loss = self.loss_fn(self.config, y_batch,b_batch, y_pred)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        result = {"loss":loss}
        return result



def save_result(anno_by_image_id_list, config, obj_threshold, iou_threshold,result_path):

    obj_threshold =  obj_threshold
    iou_threshold = iou_threshold 

    outputRescaler = OutputRescaler(config.anchors)

    box_list = []
    
    # for anno in anno_by_image_id_list:
    # for idx, anno in enumerate(anno_by_image_id_list[0:10]):
    for idx, anno in enumerate(anno_by_image_id_list):
        this_img_info = anno[0]
        this_id = this_img_info["id"]
        this_img_height = this_img_info["height"]
        this_img_width = this_img_info["width"]
        
        this_img_path = image_root_path + str(this_id).zfill(12)+".jpg"
        # print("this_img_path:",this_img_path)

        img = cv2.imread(this_img_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        


        '''
        preprocess
        '''
        dim = (224,224) # (width, height)
        x_batch = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # cv2.imwrite("x_batch.png",x_batch)


        x_batch = np.expand_dims(x_batch.astype(np.float32)/255, axis = 0)


        '''
        inference
        '''

        this_output = custom_model(x_batch).numpy()

        netout_scaled   = outputRescaler.fit(this_output[0])

     

        boxes = find_high_class_probability_bbox(netout_scaled,obj_threshold)

        if len(boxes) > 0:
            final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = draw_boxes(img,final_boxes,config.cls_label,verbose=False)

            cv2.imwrite("./result/model_output_result_{}.png".format(idx),img)

            # import pdb
            # pdb.set_trace()          
    

        
            for box in final_boxes:
                one_box_xmin = box.xmin.tolist()
                one_box_ymin = box.ymin.tolist()
                one_box_xmax = box.xmax.tolist()
                one_box_ymax = box.ymax.tolist()

                one_box_w = one_box_xmax - one_box_xmin
                one_box_h = one_box_ymax - one_box_ymin

                one_box_label = box.get_label()
                one_box_score = box.get_score()
                # print("this_id:",type(this_id))
                # print("one_box_label:",one_box_label)
                # print("score:", type(one_box_score))
                # import pdb
                # pdb.set_trace()

                category_id = one_box_label.tolist() # this is idx, not category_id
                category_id = config.cls_label[category_id][2]

                bbox = [one_box_xmin*this_img_width, one_box_ymin*this_img_height, one_box_w*this_img_width, one_box_h*this_img_height]
                # print("bbox:",bbox, one_box_xmin, this_img_width)
                one_box = {"image_id":this_id,"category_id":category_id,"bbox":bbox,"score":one_box_score.tolist()} # the label needs +1 to become coco label
                
                box_list.append(copy.deepcopy(one_box))

                # print("one_box;", one_box)


        # print("final_boxes:",final_boxes)
        if idx%100 ==0:
            print("idx:", idx)

    


    # import pdb
    # pdb.set_trace()



    with open(result_path, 'w') as outfile:
        json.dump(box_list, outfile)
        






if __name__ == "__main__":

    # tf.data.experimental.enable_debug_mode()
    # tf.config.experimental_run_functions_eagerly(True)

    annotation_path = "/home/data_c/finnweng/coco/annotations/instances_val2017.json"
    image_root_path = "/home/data_c/finnweng/coco/val2017/"
    result_path = "./result/result.json"

    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here

    config = define_config()
    model_config = define_model_config()


    json_data = get_json_data(annotation_path)

    # for id, ann in enumerate(json_data):
    #     bb = ann['bbox']
    #     print(ann["id"])



    '''
    result should be list of dict, like this:
    [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},{"image_id":73,"category_id":11,"bbox":[61,22.75,504,609.67],"score":0.318}]
    '''



    # detect_net = Detection_Net(config)
    yolov2_net = YOLOV2_Net(config)

    # build model, expose this to show how to deal with dict as fit() input
    model_input = tf.keras.Input(shape=(224,224,3),name="image",dtype=tf.float32)

    # output = detect_net(model_input)
    output = yolov2_net(model_input)

    custom_model = Custom_Model(inputs = [model_input],outputs = [output], config = config)



    custom_model.load_weights(config.model_path)



    anno_by_image_id_list = get_anno_by_image_id_list(json_data)


    obj_threshold = 0.001
    iou_threshold = 0.5

    box_list = []


    #initialize COCO ground truth api
    annFile = annotation_path
    cocoGt=COCO(annFile)

    '''
    save result
    '''
    save_result(anno_by_image_id_list, config, obj_threshold, iou_threshold,result_path)


    '''
    coco eval
    '''
    resFile = result_path
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]
    

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
