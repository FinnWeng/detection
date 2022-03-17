import numpy as np

from utils import iou

def kmeans(boxes, k, dist=np.median,seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances     = np.empty((rows, k)) ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k): # I made change to lars76's code here to make the code faster
            distances[:,icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break
            
        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters,nearest_clusters,distances


def get_best_bbox_setting(data_list):
    wh = []
    for anno in data_list:
        aw = float(anno["width"])  # width of the original image
        ah = float(anno["height"]) # height of the original image
        for bbox in anno["bbox"]:
            w = (bbox["xmax"] - bbox["xmin"])/aw # make the width range between [0,GRID_W)
            h = (bbox["ymax"] - bbox["ymin"])/ah # make the width range between [0,GRID_H)
            temp = [w,h]
            wh.append(temp)
    wh = np.array(wh)

    '''
    to show impact of different k
    '''
    # kmax = 11
    # dist = np.mean
    # results = {}
    # for k in range(2,kmax):
    #     clusters, nearest_clusters, distances = kmeans(wh,k,seed=2,dist=dist)
    #     WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
    #     result = {"clusters":             clusters,
    #             "nearest_clusters":     nearest_clusters,
    #             "distances":            distances,
    #             "WithinClusterMeanDist": WithinClusterMeanDist}
    #     print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
    #     results[k] = result

    '''
    get k = 5 anchor
    '''

    k = 5 
    dist = np.mean
    clusters, nearest_clusters, distances = kmeans(wh,k,seed=2,dist=dist)
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
    result = {"clusters":             clusters,
            "nearest_clusters":     nearest_clusters,
            "distances":            distances,
            "WithinClusterMeanDist": WithinClusterMeanDist}
    print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
    print(result["clusters"])

    #  5 clusters: mean IoU = 0.4942
    #     [[0.04076599 0.061204  ]
    #     [0.69486092 0.74961789]
    #     [0.13173388 0.181611  ]
    #     [0.48747882 0.29158246]
    #     [0.22299274 0.49581151]]