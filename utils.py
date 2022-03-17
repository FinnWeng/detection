import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import cv2


def iou(box, clusters):
    '''
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2) 
    '''
    x = np.minimum(clusters[:, 0], box[0]) 
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def plot_image_with_grid_cell_partition(x_batch, irow, config):
    img = x_batch[irow]
    img = np.around(img).astype(np.uint8)
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    
    '''
    H
    '''
    GRID_  = config.GRID_H ## 13
    IMAGE_ = config.IMAGE_H ## 416

    pltax   = plt.axhline
    plttick = plt.yticks
        
    for count in range(GRID_):
        l = IMAGE_*count/GRID_
        pltax(l,color="yellow",alpha=0.3) 
    plttick([(i + 0.5)*IMAGE_/GRID_ for i in range(GRID_)],
            ["iGRID{}={}".format("H",i) for i in range(GRID_)])

    '''
    W
    '''
    GRID_  = config.GRID_W ## 13
    IMAGE_ = config.IMAGE_W ## 416

    pltax   = plt.axvline
    plttick = plt.xticks
        
    for count in range(GRID_):
        l = IMAGE_*count/GRID_
        pltax(l,color="yellow",alpha=0.3) 
    plttick([(i + 0.5)*IMAGE_/GRID_ for i in range(GRID_)],
            ["iGRID{}={}".format("W",i) for i in range(GRID_)])

    

def plot_grid(y_batch, irow, config):
    import seaborn as sns
    color_palette = list(sns.xkcd_rgb.values())
    iobj = 0
    for igrid_h in range(config.GRID_H):
        for igrid_w in range(config.GRID_W):
            for ianchor in range(config.BOX):
                vec = y_batch[irow,igrid_h,igrid_w,ianchor,:]
                C = vec[4] ## ground truth confidence
                if C == 1:
                    # class_nm = np.array(LABELS)[np.where(vec[5:])]
                    class_name = config.cls_label[np.where(vec[5:])[0][0]][1]
                    print("class_name", class_name)
                    x, y, w, h = vec[:4]
                    multx = config.IMAGE_W/config.GRID_W
                    multy = config.IMAGE_H/config.GRID_H
                    c = color_palette[iobj]
                    iobj += 1
                    xmin = x - 0.5*w
                    ymin = y - 0.5*h
                    xmax = x + 0.5*w
                    ymax = y + 0.5*h
                    # center
                    plt.text(x*multx,y*multy,
                             "X",color=c,fontsize=23)
                    plt.plot(np.array([xmin,xmin])*multx,
                             np.array([ymin,ymax])*multy,color=c,linewidth=10)
                    plt.plot(np.array([xmin,xmax])*multx,
                             np.array([ymin,ymin])*multy,color=c,linewidth=10)
                    plt.plot(np.array([xmax,xmax])*multx,
                             np.array([ymax,ymin])*multy,color=c,linewidth=10)  
                    plt.plot(np.array([xmin,xmax])*multx,
                             np.array([ymax,ymax])*multy,color=c,linewidth=10)
    
    plt.savefig("./check_grid.jpg")


def check_grid(x_batch,config):
    plot_image_with_grid_cell_partition(x_batch, 1, config)
    plot_grid(y_batch, 1, config)


'''
evaluation
'''

class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        '''
        ANCHORS: a np.array of even number length e.g.
        
        _ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                    2,4, ##  width=2, height=4,  tall large anchor box
                    1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union
    
    def find(self,center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0,center_w, center_h)
        ##  For given object, find the best anchor box!
        for i in range(len(self.anchors)): ## run through each anchor box
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return(best_anchor,max_iou)    

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        ## the code below are used during inference
        # probability
        self.confidence      = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])

class OutputRescaler(object):
    def __init__(self,ANCHORS):
        # self.ANCHORS = ANCHORS
        self.ANCHORS = ANCHORS.numpy()

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)
    def get_shifting_matrix(self,netout):
        
        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[...,0]
        
        # ANCHORSw = self.ANCHORS[::2]
        # ANCHORSh = self.ANCHORS[1::2]

        ANCHORSw = self.ANCHORS[:,0].tolist()
        ANCHORSh = self.ANCHORS[:,1].tolist()



       
        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:,igrid_w,:] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h,:,:] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):    
            mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no) 
        for ianchor in range(BOX):    
            mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]



        return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

    def fit(self, netout):    
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)
        
        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]
        
        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)


        # bounding box parameters
        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_GRID_W)/GRID_W # x      unit: range between 0 and 1
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_GRID_H)/GRID_H # y      unit: range between 0 and 1
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_ANCHOR_W)/GRID_W      # width  unit: range between 0 and 1
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_ANCHOR_H)/GRID_H      # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1 
        netout[..., 4]   = self._sigmoid(netout[..., 4])
        expand_conf      = np.expand_dims(netout[...,4],-1) # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold 

    
        return(netout)


def find_high_class_probability_bbox(netout_scaled, obj_threshold):
    '''
    == Input == 
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)
    
             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==
    
    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C 
    
             
    '''
    GRID_H, GRID_W, BOX = netout_scaled.shape[:3]
    
    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scaled[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scaled[row,col,b,:4]
                    confidence = netout_scaled[row,col,b,4]
                    # print("np.min(x):", np.min(x))
                    # print("w", w)
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    # box = BoundBox(x, y, x+w, y+h, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return(boxes)


# def origin_nonmax_suppression(boxes,iou_threshold,obj_threshold):
def nonmax_suppression(boxes,iou_threshold,obj_threshold):
    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder    = BestAnchorBoxFinder([])
    
    CLASS    = len(boxes[0].classes)
    index_boxes = []   
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        #sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:  
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    # print("bbox_iou:",bbox_iou)
                    # print("boxes[i].get_score():",boxes[i].get_score() )
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)
                        
    newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]
    # print("len(box):",len(boxes))
    # print("newboxes:",len(newboxes))     





    return newboxes



# def nonmax_suppression(boxes,iou_threshold,obj_threshold):
#     '''
#     boxes : list containing "good" BoundBox of a frame
#             [BoundBox(),BoundBox(),...]
#     '''
#     bestAnchorBoxFinder    = BestAnchorBoxFinder([])
    
#     CLASS    = len(boxes[0].classes)
#     index_boxes = []   
#     newboxes = []
#     # suppress non-maximal boxes
#     for c in range(CLASS):
#         this_class_boxes = []

#         # extract class probabilities of the c^th class from multiple bbox
#         class_probability_from_bbxs = [box.classes[c] for box in boxes]

#         #sorted_indices[i] contains the i^th largest class probabilities
#         sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

#         for i in range(len(sorted_indices)):
#             index_i = sorted_indices[i]

#             if boxes[index_i].classes[c] <= obj_threshold:
#                 break
#             else:
#                 this_box = copy.deepcopy(boxes[index_i])
#                 classes = np.zeros(boxes[index_i].classes.shape)
#                 classes[c] = boxes[index_i].classes[c]
#                 this_box.set_class(classes)
#                 this_class_boxes.append(copy.deepcopy(this_box))

#             # if class probability is zero then ignore
#             if boxes[index_i].classes[c] == 0:  
#                 continue
#             else:
#                 # index_boxes.append(index_i)
#                 for j in range(i+1, len(sorted_indices)):
#                     if boxes[j].classes[c] <= obj_threshold:
#                         break


#                     index_j = sorted_indices[j]
                    
#                     # check if the selected i^th bounding box has high IOU with any of the remaining bbox
#                     # if so, the remaining bbox' class probabilities are set to 0.
#                     bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
#                     print("bbox_iou:",bbox_iou)
#                     print("boxes[i].get_score():",boxes[i].get_score() )
#                     if bbox_iou >= iou_threshold:
#                         pass
#                     else:
#                         this_box = copy.deepcopy(boxes[index_j])
#                         classes = np.zeros(boxes[index_j].classes.shape)
#                         classes[c] = boxes[index_j].classes[c]
#                         this_box.set_class(classes)
#                         # print("this_box.label:",this_box.label)
#                         this_class_boxes.append(copy.deepcopy(this_box))
            
#         newboxes.extend(this_class_boxes)
                        
#     # newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]
#     # print("len(box):",len(boxes))
#     # print("newboxes:",len(newboxes))     





#     return newboxes




def draw_boxes(image, boxes, labels, obj_baseline=0.05,verbose=False):
    '''
    image : np.array of shape (N height, N width, 3)
    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
    '''
    def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c
    
    image = copy.deepcopy(image)
    image_h, image_w, _ = image.shape
    score_rescaled  = np.array([box.get_score() for box in boxes])
    score_rescaled /= obj_baseline
    
    colors = sns.color_palette("husl", 8)
    # print("######################################3")
    for sr, box,color in zip(score_rescaled,boxes, colors):
        xmin = adjust_minmax(int(box.xmin*image_w),image_w)
        ymin = adjust_minmax(int(box.ymin*image_h),image_h)
        xmax = adjust_minmax(int(box.xmax*image_w),image_w)
        ymax = adjust_minmax(int(box.ymax*image_h),image_h)

 
        
        # text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())
        text = "{:10} {:4.3f}".format(labels[box.label][1], box.get_score())

        if verbose:
            print("{} xmin={:4.0f},ymin={:4.0f},xmax={:4.0f},ymax={:4.0f}".format(text,xmin,ymin,xmax,ymax,text))
        cv2.rectangle(image, 
                      pt1=(xmin,ymin), 
                      pt2=(xmax,ymax), 
                      color=color, 
                      thickness=1)
        cv2.putText(img       = image, 
                    text      = text, 
                    org       = (xmin+ 13, ymin + 13),
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1e-3 * image_h,
                    color     = (1, 0, 1),
                    thickness = 1)
        
    return image