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
    
    return  img,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor


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
    
    return img,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor




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

    union = w1*h1 + w2*h2 - intersect

    return intersect / union


@tf.function
def batch_best_anchor_box_finder(config, center_w, center_h, box_size_idx):
    # find the anchor that best predicts this box
    best_anchor = -1
    max_iou     = -1.0
    # each Anchor box is specialized to have a certain shape.
    # e.g., flat large rectangle, or small square

    # center_h, center_w: shape:(100,)
    center_w = tf.expand_dims(center_w, axis = 1) # (100,1)
    center_h = tf.expand_dims(center_h, axis = 1) # (100,1)

    anchors = config.zero_and_anchors[box_size_idx]  # 3,3,4 => 3,4

    shifted_box = tf.concat([tf.zeros([config.box_buffer,1]),tf.zeros([config.box_buffer,1]),center_w, center_h],axis = 1) # 100, 4
    # ##  For given object, find the best anchor box!
    # for i in range(len(anchors)): ## run through each anchor box
    #     anchor = anchors[i]
    #     iou    = bbox_iou(shifted_box, anchor)
    #     if max_iou < iou:
    #         best_anchor = i
    #         max_iou     = iou

    expanded_shifted_box = tf.stack([shifted_box]*config.BOX, axis = 1) # 100, new axis n = 3, 4
    batched_anchors =  tf.stack([anchors]*config.box_buffer, axis = 0) # 100, 3, 4

    

    iou = batch_bbox_iou(expanded_shifted_box, batched_anchors)

    # print("iou.shape:",iou.shape) # 100, 5, 1

    best_anchor = tf.math.argmax(iou, axis = 1, output_type=tf.int32)
    max_iou = tf.reduce_max(iou, axis = 1)

    best_anchor = tf.reshape(best_anchor, [config.box_buffer])
    max_iou = tf.reshape(max_iou, [config.box_buffer])

    return best_anchor, max_iou

    # return(best_anchor,max_iou)



def batch_data_preprocess(config, img,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor):
    # x_batch = np.zeros((config.batch_size, config.IMAGE_H, config.IMAGE_W, 3))  # input images
    # b_batch = np.zeros((config.batch_size, 1     , 1     , 1    ,  config.box_buffer, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
    # y_batch = np.zeros((config.batch_size, config.GRID_H,  config.GRID_W, config.BOX, 4+1+config.num_of_labels)) # desired network output

    
    # b_batch = tf.zeros((1,  1, 1, config.box_buffer)) # desired network output

    num_of_bbox = tf.reduce_sum(tf.cast(class_tensor != 0, tf.int32))

    '''
    deal with small bbox, then middle bbox, then large bbox.
            (large image)     (midlle image)     (small image)
    the small bbox will have more anchors than middle and large bbox one

    '''
    y_batch_list = []
    b_batch_list = []

    for box_size_idx in range(3):
        this_resize_scale = 2**(3+box_size_idx)

        y_batch = tf.zeros(((config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), config.BOX, 4+1+config.num_of_labels)) # desired network output


        center_x, center_y = rescale_centerxy(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) #  # (100,)  # (100,)

        grid_x = tf.cast(tf.math.floor(center_x),tf.int32) # (100,)
        grid_y = tf.cast(tf.math.floor(center_y),tf.int32) # (100,)

        center_w, center_h = rescale_cebterwh(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) # (100,)  # (100,)

        '''
        which anchor should we use
        '''

        grid_x = tf.cast(tf.math.floor(center_x),tf.int32) # (100,)
        grid_y = tf.cast(tf.math.floor(center_y),tf.int32) # (100,)

        '''
        find best anchor suit for bbox
        '''
        best_anchor,max_iou = batch_best_anchor_box_finder(config, center_w, center_h, box_size_idx) # (100,) (100,)

        # box = [center_x, center_y, center_w, center_h]
        boxes = tf.stack([center_x, center_y, center_w, center_h], axis = 1) # 100,4 
        # print("boxes.shape:",boxes.shape)
        
        '''
        making data and put it into y_batch
        '''
        update_index = tf.stack([grid_y, grid_x,best_anchor],axis = 1) # the 0,0,0, ... will be those padding. these should be dropped eventaully 
        update_index = update_index[:num_of_bbox] # (n, 3)

        onehot_cls_label = tf.one_hot(class_tensor, depth = (config.num_of_labels+1)) # + 1 for label 0 I difined: padding
        onehot_cls_label = onehot_cls_label[:,1:]
        # print("onehot_cls_label.shape:", onehot_cls_label.shape) # (100, 90)

        update_index_indicator = tf.ones([config.box_buffer,1])
        # print("update_index_indicator.shape:",update_index_indicator.shape) # (100, 90)
        updates = tf.concat([boxes, update_index_indicator, onehot_cls_label], axis = 1) # (100,95)
        updates = updates[:num_of_bbox] # (n, 95)



        # y_batch[grid_y, grid_x, best_anchor, 0:4] = box
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
                            "large_y":y_batch_list[2], "large_true_boxes":b_batch_list[2]}





def batch_data_preprocess_v3(config, img,xmax_tensor,xmin_tensor,ymax_tensor,ymin_tensor, class_tensor):
    # x_batch = np.zeros((config.batch_size, config.IMAGE_H, config.IMAGE_W, 3))  # input images
    # b_batch = np.zeros((config.batch_size, 1     , 1     , 1    ,  config.box_buffer, 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
    # y_batch = np.zeros((config.batch_size, config.GRID_H,  config.GRID_W, config.BOX, 4+1+config.num_of_labels)) # desired network output

    
    # b_batch = tf.zeros((1,  1, 1, config.box_buffer)) # desired network output

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


    for box_size_idx in range(3):

        '''
        find best anchor suit for bbox
        '''
        this_resize_scale = 2**(3+box_size_idx)
        center_w, center_h = rescale_cebterwh(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) # (100,)  # (100,)
        center_list.append([center_w, center_h])
        best_anchor,max_iou = batch_best_anchor_box_finder(config, center_w, center_h, box_size_idx) # (100,) (100,)
        best_anchor_list.append(best_anchor)

        max_iou = max_iou[:num_of_bbox]
        max_iou_list.append(max_iou)

    max_iou_array = tf.stack(max_iou_list, axis = 0)
    # print("max_iou_array:",max_iou_array.shape)
    great_max_iou_array = tf.cast(max_iou_array>0.3, tf.float32) # (3,n_boxes)
    # print("great_max_iou_array:",great_max_iou_array.shape)

    max_iou_idx = tf.argmax(max_iou_array,axis = 0)
    # print("max_iou_idx:",max_iou_idx.shape)
    is_argmax_iou_array = tf.one_hot(max_iou_idx, depth = 3)
    # print("is_argmax_iou_array:",is_argmax_iou_array)
    is_argmax_iou_array= tf.transpose(is_argmax_iou_array, [1,0]) # 3, n_boxes
    # print("is_argmax_iou_array:",is_argmax_iou_array.shape)
    
    update_max_iou_array = great_max_iou_array + is_argmax_iou_array #  (3, n_boxes)

    update_max_iou_array = tf.clip_by_value(update_max_iou_array, clip_value_min=0, clip_value_max=1)



    for box_size_idx in range(3):
        center_w, center_h = center_list[box_size_idx][0], center_list[box_size_idx][1]
        best_anchor = best_anchor_list[box_size_idx]

        this_update_max_iou_array = update_max_iou_array[box_size_idx] # (n_boxes)
        this_update_max_iou_array = tf.expand_dims(this_update_max_iou_array, axis = 1)

        this_resize_scale = 2**(3+box_size_idx)

        y_batch = tf.zeros(((config.IMAGE_H // this_resize_scale), (config.IMAGE_W // this_resize_scale), config.BOX, 4+1+config.num_of_labels)) # desired network output


        center_x, center_y = rescale_centerxy(config, xmax_tensor, xmin_tensor, ymax_tensor, ymin_tensor, this_resize_scale) #  # (100,)  # (100,)

        grid_x = tf.cast(tf.math.floor(center_x),tf.int32) # (100,)
        grid_y = tf.cast(tf.math.floor(center_y),tf.int32) # (100,)

        

        '''
        which anchor should we use
        '''

        grid_x = tf.cast(tf.math.floor(center_x),tf.int32) # (100,)
        grid_y = tf.cast(tf.math.floor(center_y),tf.int32) # (100,)



        # box = [center_x, center_y, center_w, center_h]
        boxes = tf.stack([center_x, center_y, center_w, center_h], axis = 1) # 100,4 
        # print("boxes.shape:",boxes.shape)
        
        '''
        making data and put it into y_batch
        '''
        update_index = tf.stack([grid_y, grid_x,best_anchor],axis = 1) # the 0,0,0, ... will be those padding. these should be dropped eventaully 
        update_index = update_index[:num_of_bbox] # (n, 3)

        onehot_cls_label = tf.one_hot(class_tensor, depth = (config.num_of_labels+1)) # + 1 for label 0 I difined: padding
        onehot_cls_label = onehot_cls_label[:,1:]
        # print("onehot_cls_label.shape:", onehot_cls_label.shape) # (100, 90)

        update_index_indicator = tf.ones([config.box_buffer,1])
        # print("update_index_indicator.shape:",update_index_indicator.shape) # (100, 90)
        updates = tf.concat([boxes, update_index_indicator, onehot_cls_label], axis = 1) # (100,95)
        updates = updates[:num_of_bbox] # (n, 95)

        # print("updates:",updates[...,0])
        # print("this_update_max_iou_array:",this_update_max_iou_array[...,0])

        updates = updates*this_update_max_iou_array
        # print("updates(after):",updates[...,0])



        # y_batch[grid_y, grid_x, best_anchor, 0:4] = box
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
                            "large_y":y_batch_list[2], "large_true_boxes":b_batch_list[2]}




    