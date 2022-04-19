import tensorflow as tf


'''

some tf.data image preprocess

'''


def tf_load_image(config, id_tensor, height_tensor, width_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor):
    # id_tensor, height_tensor, width_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor = data_input

    img = tf.io.read_file(id_tensor)
    img = tf.io.decode_jpeg(img, channels=3)

    return img, height_tensor, width_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor


def tf_resize_image(img, height_tensor, width_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor):
    

    img = tf.image.resize(img, [224,224]) # will be float32
    xmax_tensor = xmax_tensor/tf.cast(width_tensor,tf.float32)*224
    xmin_tensor = xmin_tensor/tf.cast(width_tensor,tf.float32)*224
    ymax_tensor = ymax_tensor/tf.cast(height_tensor,tf.float32)*224
    ymin_tensor = ymin_tensor/tf.cast(height_tensor,tf.float32)*224

    width_tensor = tf.cast(width_tensor,tf.int32)
    height_tensor = tf.cast(height_tensor,tf.int32)
    
    return  img, height_tensor, width_tensor, xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor


def tf_crop_and_resize_image(img, height_tensor, width_tensor, class_tensor,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor):

    width_tensor = tf.cast(width_tensor,tf.int32)
    height_tensor = tf.cast(height_tensor,tf.int32)

    minimum_x = tf.reduce_min(xmin_tensor)
    maximum_x = tf.reduce_max(xmax_tensor)
    minimum_y = tf.reduce_min(ymin_tensor)
    maximum_y = tf.reduce_max(ymax_tensor)


    # import pdb
    # pdb.set_trace()
    if tf.cast(tf.math.floor(minimum_x),tf.int32) == 0:
        start_x = 0
    else:
        start_x = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.floor(minimum_x),tf.int32), dtype=tf.int32)
    
    if tf.cast(tf.math.ceil(maximum_x),tf.int32) == width_tensor:
        end_x = width_tensor
    else:
        end_x = tf.random.uniform(shape=[], minval=tf.cast(tf.math.ceil(maximum_x),tf.int32), maxval=width_tensor, dtype=tf.int32)

    if tf.cast(tf.math.floor(minimum_y),tf.int32) == 0:
        start_y = 0
    else:
        start_y = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.floor(minimum_y),tf.int32), dtype=tf.int32)
    
    if tf.cast(tf.math.ceil(maximum_y),tf.int32) == height_tensor:
        end_y = height_tensor
    else:
        end_y = tf.random.uniform(shape=[], minval=tf.cast(tf.math.ceil(maximum_y),tf.int32), maxval=height_tensor, dtype=tf.int32)
    

    img = img[start_y:end_y, start_x:end_x,:]

    new_width = end_x - start_x
    new_height = end_y - start_y

    img = tf.image.resize(img, [224,224])
    xmax_tensor = (xmax_tensor - tf.cast(start_x,tf.float32))/tf.cast(new_width,tf.float32)*224
    xmin_tensor = (xmin_tensor - tf.cast(start_x,tf.float32))/tf.cast(new_width,tf.float32)*224
    ymax_tensor = (ymax_tensor - tf.cast(start_y,tf.float32))/tf.cast(new_height,tf.float32)*224
    ymin_tensor = (ymin_tensor - tf.cast(start_y,tf.float32))/tf.cast(new_height,tf.float32)*224
    
    return img, height_tensor, width_tensor, xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor




'''
tools for original version and batch version
'''


# @tf.function
# def rescale_centerxy(config, xmax, xmin, ymax, ymin):
#     '''
#     obj:     dictionary containing xmin, xmax, ymin, ymax
#     config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
#     '''
#     center_x = .5*(xmin + xmax)
#     center_x = center_x / (float(config.IMAGE_W) / config.GRID_W) # center_x/32
#     center_y = .5*(ymin + ymax)
#     center_y = center_y / (float(config.IMAGE_H) / config.GRID_H)
#     return(center_x,center_y)

# @tf.function
# def rescale_cebterwh(config, xmax, xmin, ymax, ymin):
#     '''
#     obj:     dictionary containing xmin, xmax, ymin, ymax
#     config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
#     '''    
#     # unit: grid cell
#     center_w = (xmax - xmin) / (float(config.IMAGE_W) / config.GRID_W) 
#     # unit: grid cell
#     center_h = (ymax - ymin) / (float(config.IMAGE_H) / config.GRID_H) 
#     return(center_w,center_h)


@tf.function
def rescale_centerxy(config, xmax, xmin, ymax, ymin, scaling_rate):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''
    center_x = .5*(xmin + xmax)
    center_x = center_x / scaling_rate # center_x/8 or center_x/16 or center_x/32
    center_y = .5*(ymin + ymax)
    center_y = center_y / scaling_rate # center_y/8 or center_y/16 or center_y/32
    return(center_x,center_y)

@tf.function
def rescale_cebterwh(config, xmax, xmin, ymax, ymin, scaling_rate):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''    
    # unit: grid cell
    center_w = (xmax - xmin) / scaling_rate # center_w/8 or center_w/16 or center_w/32
    # unit: grid cell
    center_h = (ymax - ymin) / scaling_rate # center_h/8 or center_h/16 or center_h/32
    return(center_w,center_h)







'''

batch process I write

'''


@tf.function
def batch_interval_overlap(interval_a, interval_b):
    a_max_greaterthan_b_min = interval_a[1] > interval_b[0]
    b_max_greaterthan_a_min = interval_b[1] > interval_a[0]
    a_max_greaterthan_b_max = interval_a[1] > interval_b[1]
    a_min_greaterthan_b_min = interval_a[0] > interval_b[0]

    '''
    decide whether they overlap
    '''
    overlap = tf.math.logical_and(a_max_greaterthan_b_min, b_max_greaterthan_a_min)
    # print(overlap)

    '''
    decide how to compute
    '''
    case_1 = tf.cast(tf.math.logical_and( tf.math.logical_not(a_max_greaterthan_b_max), tf.math.logical_not(a_min_greaterthan_b_min)), tf.float32)
    case_2 = tf.cast(tf.math.logical_and( a_max_greaterthan_b_max, a_min_greaterthan_b_min), tf.float32)
    case_3 = tf.cast(tf.math.logical_and( tf.math.logical_not(a_max_greaterthan_b_max), a_min_greaterthan_b_min), tf.float32)
    case_4 = tf.cast(tf.math.logical_and( a_max_greaterthan_b_max, tf.math.logical_not(a_min_greaterthan_b_min)), tf.float32)

    # print("case_1:", case_1) # 100, 5, 1
    # print("case_2:", case_2)
    # print("case_3:", case_3)
    # print("case_4:", case_4)
    
    case_1_area = (interval_a[1] - interval_b[0]) * case_1
    case_2_area = (interval_b[1] - interval_a[0]) * case_2
    case_3_area = (interval_a[1] - interval_a[0]) * case_3
    case_4_area = (interval_b[1] - interval_b[0]) * case_4

    

    overlap_area = case_1_area + case_2_area + case_3_area + case_4_area
    overlap_area = tf.cast(overlap, tf.float32)*overlap_area


    return overlap_area
    

@tf.function
def batch_bbox_iou(box1, box2):

    # intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    # intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  


    intersect_w = batch_interval_overlap([box1[:,:,0:1], box1[:,:,2:3]], [box2[:,:,0:1], box2[:,:,2:3]])
    intersect_h = batch_interval_overlap([box1[:,:,1:2], box1[:,:,3:4]], [box2[:,:,1:2], box2[:,:,3:4]])  

    intersect = intersect_w * intersect_h

    w1, h1 = box1[:,:,2:3]-box1[:,:,0:1], box1[:,:,3:4]-box1[:,:,1:2]
    w2, h2 = box2[:,:,2:3]-box2[:,:,0:1], box2[:,:,3:4]-box2[:,:,1:2]

    union = w1*h1 + w2*h2 - intersect + 1e-10

    return intersect / union


@tf.function
def batch_best_anchor_box_finder(config, ious):
    # find the anchor that best predicts this box


    # print("iou.shape:",iou.shape) # 100, 5, 1

    best_anchor = tf.math.argmax(ious, axis = 1, output_type=tf.int32)
    max_iou = tf.reduce_max(ious, axis = 1)

    best_anchor = tf.reshape(best_anchor, [config.box_buffer])
    max_iou = tf.reshape(max_iou, [config.box_buffer])

    return best_anchor, max_iou

    # return(best_anchor,max_iou)



@tf.function
def tf_bbox_iou(boxes1, boxes2):


    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1) # x min, ymin, x max, y max
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.math.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.math.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.math.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / (union_area + 1e-10)


@tf.function
def batch_data_preprocess_v3(config, img, height_tensor, width_tensor, xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor):
    # x_batch = np.zeros((config.batch_size, config.IMAGE_H, config.IMAGE_W, 3))  # input images
    # b_batch = np.zeros((config.batch_size, 1     , 1     , 1    ,  config.box_buffer, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
    # y_batch = np.zeros((config.batch_size, config.GRID_H,  config.GRID_W, config.BOX, 4+1+config.num_of_labels)) # desired network output

    
    # b_batch = tf.zeros((1,  1, 1, config.box_buffer)) # desired network output

    # print("xmax_tensor:",xmax_tensor.shape) #  (100,)
    # print("class_tensor:",class_tensor.shape, class_tensor) #  (100,)
    

    num_of_bbox = tf.reduce_sum(tf.cast(class_tensor != 0, tf.int32))

    '''
    deal with small bbox, then middle bbox, then large bbox.
            (large image)     (midlle image)     (small image)
    the small bbox will have more anchors than middle and large bbox one

    '''
    y_batch_list = []
    b_batch_list = []
    center_list = []
    best_anchor_list = []
    max_iou_list = []
    ious_list = []

    


    center_x, center_y = rescale_centerxy(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, 1) #  # (100,)  # (100,)
    center_w, center_h = rescale_cebterwh(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, 1) # (100,)  # (100,)
    # box = [center_x, center_y, center_w, center_h]
    boxes = tf.stack([center_x, center_y, center_w, center_h], axis = 1) # 100,4 

    # print("boxes:",boxes)


    for box_size_idx in range(3):

        '''
        find best anchor suit for bbox
        '''
        this_resize_scale = 2**(3+box_size_idx)

        y_batch = tf.zeros(((config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), config.BOX, 4+1+config.num_of_labels)) # desired network output        

        center_x, center_y = rescale_centerxy(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) #  # (100,)  # (100,)
        center_w, center_h = rescale_cebterwh(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) # (100,)  # (100,)

        bbox_xywh_scaled = tf.stack([center_x, center_y, center_w, center_h], axis = 1) # 100, 4
        bbox_xywh_scaled = tf.stack([bbox_xywh_scaled]*config.BOX, axis = 1) # 100, 3, 4
        # print("bbox_xywh_scaled:",bbox_xywh_scaled.shape)



        anchors = config.anchors[box_size_idx] # 3,2
        anchors = tf.stack([anchors]*config.box_buffer, 0) # 100,3,2  # anchors vs scaled xywh

        grid_x = tf.cast(tf.math.floor(center_x),tf.int32) # (100,)
        grid_y = tf.cast(tf.math.floor(center_y),tf.int32) # (100,)

     
        anchors_xy = tf.cast(tf.stack([grid_x, grid_y],axis = 1),tf.float32) + 0.5 # 100, 2
        anchors_xy = tf.stack([anchors_xy]*config.BOX,axis = 1) # 100, 3, 2
        anchors_xywh = tf.concat([anchors_xy, anchors],axis = 2) # 100,3,4 

        center_list.append([center_w, center_h])

        '''
        update_or_not: # (num_of_box, config.BOX)
        updates: # (num_of_box, 85), include corresponding grid: # (num_of_box,4)
        [updates]*config.BOX # (num_of_box, config.BOX, 85)

        tensor:  (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), config.BOX, 85))
        outer_shape = tensor.shape[:index_depth] = (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), (config.BOX)
        inner_shape = 85

        index_depth = 3 # 3 for  (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), (config.BOX)
        batch_shape = num_of_box
        index_shape: [num_of_box, 3] 

        update: batch_shape + inner_shape = (num_of_box, 85)


        tensor:  (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), config.BOX, 85))
        outer_shape =  (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale)
        inner_shape = config.BOX, 85

        update_value: # (num_of_box, 85), include corresponding grid: # (num_of_box,4)
        update_or_not: # (num_of_box, config.BOX)
        update_value = update_value*3 # ([num_of_box, 3, 85])
        update = update_value* update_or_not # (num_of_box, 3, 85)

        index_depth = 2 # 2 for  (config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale)
        batch_shape = (num_of_box)
        index_shape = (num_of_box, 2)
        '''

        # ious = batch_bbox_iou(bbox_xywh_scaled,anchors_xywh)

        ious = tf_bbox_iou(bbox_xywh_scaled,anchors_xywh)
        # print("ious:", ious.shape) # (100, config.BOX, 1)
        ious = tf.reshape(ious, [config.box_buffer,config.BOX]) # (100, config.BOX)

        # over this threshold, take all anchors
        ious_over_threshould = ious > config.take_upper_threshold # (100, config.BOX)

        best_anchor,max_iou = batch_best_anchor_box_finder(config, ious) 
        best_anchor_one_hot = tf.one_hot(best_anchor, depth = config.BOX) # (100, config.BOX)

        update_or_not = tf.cast(ious_over_threshould, tf.float32) + best_anchor_one_hot # (100, config.BOX)
        update_or_not = tf.clip_by_value(update_or_not, clip_value_min=0, clip_value_max=1) # (100, config.BOX)

        update_or_not = update_or_not[:num_of_bbox]
        update_or_not = tf.expand_dims(update_or_not, axis = -1) # num_of_bbox, config.BOX, 1
        # print("update_or_not:",update_or_not.shape)

        onehot_cls_label = tf.one_hot(class_tensor, depth = (config.num_of_labels))
        update_index_indicator = tf.ones([config.box_buffer,1])
        # print("update_index_indicator.shape:",update_index_indicator.shape) # (100, 90)
        update_value = tf.concat([boxes, update_index_indicator, onehot_cls_label], axis = 1) # (100,95)
        update_value = update_value[:num_of_bbox] # (n, 95)
        update_value = tf.stack([update_value]*config.BOX, axis = 1) # ([num_of_box, config.BOX, 85])

        updates = update_value* update_or_not 
        # print("updates:",updates.shape)


        update_index = tf.stack([grid_y, grid_x],axis = 1) # the 0,0,0, ... will be those padding. these should be dropped eventaully 
        update_index = update_index[:num_of_bbox] # (n, 2)
        # print("update_index:",update_index.shape)


        # max_iou = max_iou[:num_of_bbox]


        '''

        y_batch:
        center_x, center_y, center_w, center_h, this is box or not(update_index_indicator).  The center_x and cener_y are not scaled.
        '''


        y_batch = tf.tensor_scatter_nd_update(y_batch, update_index, updates)

        b_batch = tf.reshape(boxes, [1,1,1,config.box_buffer, 4])

        y_batch_list.append(y_batch)
        b_batch_list.append(b_batch)







    '''
    img normalizing
    '''
    x_batch = img/255.

    # return best_anchor,max_iou,x_batch, b_batch, y_batch
    # return [x_batch, b_batch], y_batch

    return {"x":x_batch}, {"small_y":y_batch_list[0],"small_true_boxes": b_batch_list[0], 
                            "middle_y":y_batch_list[1], "middle_true_boxes":b_batch_list[1], 
                            "large_y":y_batch_list[2], "large_true_boxes":b_batch_list[2],
                            "height_tensor":height_tensor, "width_tensor":width_tensor}

    # return {"x":x_batch}, {"small_y":y_batch_list[0],"small_true_boxes": b_batch_list[0], 
    #                     "middle_y":y_batch_list[1], "middle_true_boxes":b_batch_list[1], 
    #                     "large_y":y_batch_list[2], "large_true_boxes":b_batch_list[2]}



