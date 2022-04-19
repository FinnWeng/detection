import tensorflow as tf





'''
extra function for pure yolov3
'''

def decode(config, conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + config.num_of_labels))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * config.strides[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * config.anchors[i]) * config.strides[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]+ 1e-10
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou




def yolov3_custom_loss(config, y_true, true_boxes, y_pred):
    '''
    y_pred[box_size_idx], small: (32, 52, 52, 3, 85)
    y_pred[box_size_idx], middle: (32, 26, 26, 3, 85)
    y_pred[box_size_idx], large: (32, 13, 13, 3, 85)

    '''
    # print("y_true[0]:",y_true[0].shape) # (32, 28, 28, 3, 2)
    # print("y_true:",y_true[0][0,:,:,0,0])
    loss_list = []
    for box_size_idx in range(3):
        conv = y_pred[box_size_idx]
        bboxes = true_boxes[box_size_idx]

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = config.strides[box_size_idx] * output_size
        input_size = tf.cast(input_size, tf.float32)

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred = decode(config, conv, box_size_idx)
        label = y_true[box_size_idx]
        # print("label:",label.shape)

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
        # print("pred_xywh[:, :, :, :, tf.newaxis, :]:",pred_xywh[:, :, :, :, tf.newaxis, :].shape) # (32, 52, 52, 3, 1, 4)
        # print("bboxes:",bboxes.shape) # (32, 1, 1, 1, 100, 4)
        # print("bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]:",bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :].shape)

        # iou = bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
        iou = bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes)
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # print("iou", iou.shape) #  (32, 7, 7, 3, 100)
        # print("max_iou:",max_iou.shape) # max_iou: (32, 7, 7, 3, 1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < config.IOU_LOSS_THRESH, tf.float32 )

        # conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # conf_loss = conf_focal * (
        #         respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        #         +
        #         respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        # )

        conf_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)+ \
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob) 
        #Above: for those padding of box class, they will be classified as "0", which is person. But since the respond bbox is 0. it will not effect the model


        '''
        add coord loss

        YAD2K:
        yolo_head will add index and multiply anchor

        adjusted_box[0:2]: shift from anchor origin.
        adjusted_box[2:4]: np.log(box[2] / anchors[best_anchor][0]).

        pred_boxes = K.concatenate(
            (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)
                
        My:        
        y_batch:
        center_x, center_y, center_w, center_h, this is box or not(update_index_indicator)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * config.strides[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * config.anchors[i]) * config.strides[i]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        For make coord loss:
        y_batch => (y_batch[0:2] - grid)
          ,also => tf.log(y_batch[2:4]/config.anchors[i]/config.strides[i])

        conv_raw_dxdy = conv[:, :, :, :, 0:2]
        conv_raw_dwdh = conv[:, :, :, :, 2:4]
        conv_raw_dxdy = tf.sigmoid(conv_raw_dxdy)
        '''
        conv_raw_dxdy = conv[:, :, :, :, 0:2]
        conv_raw_dxdy = tf.sigmoid(conv_raw_dxdy)
        conv_raw_dwdh = conv[:, :, :, :, 2:4] 
        conv_xywh = tf.concat([conv_raw_dxdy,conv_raw_dwdh], axis = -1)

        coord_box_xy = label_xywh[:, :, :, :, 0:2]
        coord_box_wh = label_xywh[:, :, :, :, 2:4]

        # print("coord_box_xy:",coord_box_xy.shape) # (32, 28, 28, 3, 2)
        # print("xy_grid:",xy_grid.shape) # (32, 28, 28, 3, 2)

        # print("coord_box_xy:",coord_box_xy[0,:,:,0,0])
        # print("xy_grid:", xy_grid[0,:,:,0,0])

        coord_box_xy = (coord_box_xy)/config.strides[box_size_idx] - xy_grid # something will be negative. it is ok since there're won't be boxes

        # print("config.anchors[box_size_idx]:",config.anchors[box_size_idx].shape) #  (3, 2)
        # print("coord_box_wh:",coord_box_wh.shape) # coord_box_wh: (32, 7, 7, 3, 2)

        print("min coord_box_wh", tf.reduce_min(coord_box_wh))
        print("config.anchors[box_size_idx]:",config.anchors[box_size_idx])
        print("config.strides[box_size_idx]:",config.strides[box_size_idx])
        coord_box_wh = tf.math.log(coord_box_wh/config.anchors[box_size_idx]/config.strides[box_size_idx] + 1e-10)

        coord_box = tf.concat([coord_box_xy, coord_box_wh],axis = -1)
        print("coord_box:",coord_box.shape)
        print("coord_box:",coord_box[0,0,:,:,:])
        print("conv_xywh:",conv_xywh.shape)

        coord_box_diff = coord_box - conv_xywh

        print("coord_box_diff:",coord_box_diff.shape)
        print("respond_bbox:",respond_bbox.shape)
        print("respond_bbox:",respond_bbox[0, 0, :, :, :])

        coord_loss = respond_bbox* (coord_box_diff**2)




        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
        coord_loss = tf.reduce_mean(tf.reduce_sum(coord_loss, axis=[1,2,3,4]))

        loss_list.append([giou_loss , conf_loss , prob_loss, coord_loss])
    
    # loss = tf.reduce_sum(loss_list)

    # giou_loss + conf_loss + prob_loss
    giou_loss = loss_list[0][0] + loss_list[1][0] + loss_list[2][0]
    conf_loss = loss_list[0][1] + loss_list[1][1] + loss_list[2][1]
    prob_loss = loss_list[0][2] + loss_list[1][2] + loss_list[2][2]
    coord_loss = loss_list[0][3] + loss_list[1][3] + loss_list[2][3]
    
    loss = giou_loss + conf_loss + prob_loss + coord_loss
    return loss, giou_loss, conf_loss,  prob_loss, coord_loss



