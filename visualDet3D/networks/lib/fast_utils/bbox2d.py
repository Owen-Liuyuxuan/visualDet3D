import numpy as np
from numba import jit

@jit
def bbox2d_area(bbox2d):
    """
        input:
            [n, 4] [x1, y1, x2, y2]
        return:
            [n, ]
    """
    dx = bbox2d[:, 2] - bbox2d[:, 0]
    dy = bbox2d[:, 3] - bbox2d[:, 1]
    dx[np.where(dx < 0)] = 0
    dy[np.where(dy < 0)] = 0
    area = dx * dy
    return area

@jit
def iou_2d_combination(box2d_0, box2d_1):
    """
        input:
            [n, 4] [x1, y1, x2, y2]
            [k, 4] [x1, y1, x2, y2]
        return:
            [n, k]
    """
    n = box2d_0.shape[0]
    k = box2d_1.shape[0]
    result = np.zeros((n, k))
    bbox_2d_repeated = np.zeros((k, 4))
    for i in range(n):
        bbox_2d_0 = box2d_0[i] #[1, 4]
        for j in range(k):
            bbox_2d_repeated[j] = bbox_2d_0
        result[i] = iou_2d(bbox_2d_repeated, box2d_1)
    return result

@jit
def iou_2d(box2d_0, box2d_1):
    """
        input:
            [n, 4] [x1, y1, x2, y2]
            [n, 4] [x1, y1, x2, y2]
        return:
            [n, ]
    """
    n = box2d_0.shape[0]
    result = np.zeros(n)

    area_0 = bbox2d_area(box2d_0)
    area_1 = bbox2d_area(box2d_1)
    
    for i in range(n):
        x1 = max(box2d_0[i, 0], box2d_1[i, 0])
        x2 = min(box2d_0[i, 2], box2d_1[i, 2])
        y1 = max(box2d_0[i, 1], box2d_1[i, 1])
        y2 = min(box2d_0[i, 3], box2d_1[i, 3])
        dx = x2 - x1
        dy = y2 - y1
        if dx <= 0 or dy <= 0:
            result[i] = 0
        else:
            area = dx * dy
            result[i] = area / (area_0[i] + area_1[i] - area)
    return result

@jit
def xyxy2xywh(box2d):
    """
        input   : [n, 4] [x1, y1, x2, y2]
        return  : [n, 4] [x, y, w, h]

        numpy accelerated
    """
    center_x = 0.5 * (box2d[:, 0] + box2d[:, 2])
    center_y = 0.5 * (box2d[:, 1] + box2d[:, 3])
    width_x  = box2d[:, 2] - box2d[:, 0]
    width_y  = box2d[:, 3] - box2d[:, 1]
    result = np.zeros_like(box2d)
    result[:, 0] = center_x
    result[:, 1] = center_y
    result[:, 2] = width_x
    result[:, 3] = width_y
    return result

@jit
def xywh2xyxy(box2d):
    """
        input   :  [n, 4] [x, y, w, h]
        return  :  [n, 4] [x1, y1, x2, y2]
    """
    halfw = 0.5*box2d[:, 2]
    halfh = 0.5*box2d[:, 3]

    result = np.zeros_like(box2d)
    result[:, 0] = box2d[:, 0] - halfw
    result[:, 1] = box2d[:, 1] - halfh
    result[:, 2] = box2d[:, 0] + halfw
    result[:, 3] = box2d[:, 1] + halfh
    return result

@jit
def compute_center_targets(gts, anchors, epsilon=1e-6):
    """
        input:
            gts:[n, 2] [cx, cy]
            anchors:[n, 4] [cx, cy, w, h]
            epsilon: float, avoid overflow when np.any(anchors[2:4] == 0)
        output:
            target: [n, 2]
    """
    targets = (gts - anchors[:, 0:2]) / (anchors[:, 2:4] + epsilon)
    return targets

@jit 
def compute_scale_ratios(gts, anchors, epsilon=1e-6):
    """
        input:
            gts:[n, 2] [w, h]
            anchors:[n, 4] [cx, cy, w, h]
            epsilon: float, avoid overflow when np.any(anchors[2:4] == 0)
        output:
            target: [n, 2]
    """
    targets = gts / (anchors[2:4] + 1e-6)
    return targets

@jit
def compute_targets(gts, anchors):
    """
        input:
            gts: [n, 4] [x1, y1, x2, y2]
            anchors: [n, 4] [x1, y1, x2, y2]
    """
    gts_xywh = xyxy2xywh(gts)
    anchor_xywh = xyxy2xywh(anchors)

    results = np.zeros_like(gts)
    results[:, 0:2] = compute_center_targets(gts_xywh[:, 0:2], anchor_xywh)
    results[:, 2:4] = compute_center_targets(gts_xywh[:, 2:4], anchor_xywh)
    return results

@jit
def determine_targets(gts, anchors, bg_threshold=0.3, fg_threshold=0.4):
    """
        inputs:
            gts: [n, 4] [x1, y1, x2, y2]
            anchors: [k, 4] [x1, y1, x2, y2]
            bg_threshold: float, 0.3~0.4
            fg_threshold: float, 0.4~0.5
        outputs:
            anchor_gts_index: [k, ] # the index of gts for each anchor
            positive_index: [] 1d-int32-array of anchor's indexes
            negative_index: [] 1d-int32-array of anchor's indexes
    """
    k = anchors.shape[0]
    iou_n_k = iou_2d_combination(anchors, gts) #[k, n]
    positive_index = []
    negative_index = []
    anchor_gts_index = np.zeros((k), dtype=np.int32)#.astype(np.int32)
    for i in range(k):
        max_iou = np.max(iou_n_k[i])
        if max_iou < bg_threshold:
            negative_index.append(i)
        if max_iou > fg_threshold:
            positive_index.append(i)
            anchor_gts_index[i] = np.argmax(iou_n_k[i])
    return anchor_gts_index, np.array(positive_index, dtype=np.int32), np.array(negative_index, dtype=np.int32)


if __name__ == "__main__":
    box1 = np.random.rand(15, 4)
    box2 = np.random.rand(15, 4)
    xywh2xyxy(box1)
    xyxy2xywh(box1)
    bbox2d_area(box1)
    anchor_gts_index, positive_index, negative_index = determine_targets(box1, box2, 0.3, 0.4)
    print(anchor_gts_index, positive_index, negative_index)
    compute_targets(box1, box2)

    print("compilation succeeded")
    box2 = np.zeros([32, 4])
    import time
    a = time.time()
    xyxy2xywh(box2)
    print(time.time() - a)
    
    a = time.time()
    box1 = np.random.rand(10, 4)
    box2 = np.random.rand(1024, 4)
    determine_targets(box1, box2, 0.3, 0.4)
    
    print(time.time() - a)