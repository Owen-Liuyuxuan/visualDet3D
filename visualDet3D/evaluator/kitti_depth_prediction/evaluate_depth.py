import numpy as np
from numba import jit
import cv2
import os

@jit(cache=True, nopython=True)
def compute_errors(image_gt, image_pred):
    """ Compute Errors from two floating point image.
    init errors
    1. mae
    2. rmse
    3. inverse mae
    4. inverse rmse
    5. log mae
    6. log rmse
    7. scale invariant log
    8. abs relative
    9. squared relative
    """
    errors = np.zeros(9)
    num_pixels = 0.0
    # log sum for scale invariant metric
    logSum  = 0.0

    w, h = image_gt.shape
    for i in range(w):
        for j in range(h):
            if image_gt[i, j] > 0.01:
                depth_pred = image_pred[i, j]
                depth_gt   = image_gt[i, j]

                d_err = abs(depth_pred - depth_gt)
                d_err_squared = d_err ** 2

                d_err_inv = abs(1.0 / depth_gt - 1.0 / depth_pred)
                d_err_inv_squared = d_err_inv ** 2

                d_err_log = abs(np.log(depth_pred) - np.log(depth_gt))
                d_err_log_squared = d_err_log ** 2

                # MAE
                errors[0] += d_err
                # rmse
                errors[1] += d_err_squared
                # inv_mae
                errors[2] += d_err_inv
                # inv_rmse
                errors[3] += d_err_inv_squared
                # log
                errors[4] += d_err_log
                errors[5] += d_err_log_squared
                # log diff for scale invariancet metric
                logSum += np.log(depth_gt) - np.log(depth_pred)
                # abs relative
                errors[7] += d_err / depth_gt
                # squared relative
                errors[8] += d_err_squared / (depth_gt ** 2 )

                num_pixels += 1
    # normalize mae
    errors[0] = errors[0] / num_pixels
    # normalize and take sqrt for rmse
    errors[1] = errors[1] / num_pixels
    errors[1] = np.sqrt(errors[1])
    # normalize inverse absoulte error
    errors[2] = errors[2] / num_pixels
    # normalize and take sqrt for inverse rmse
    errors[3] = errors[3] / num_pixels
    errors[3] = np.sqrt(errors[3])
    # normalize log mae
    errors[4] = errors[4] / num_pixels
    # first normalize log rmse -> we need this result later
    normalizedSquaredLog = errors[5] / num_pixels
    errors[5] = np.sqrt(normalizedSquaredLog)
    # calculate scale invariant metric
    errors[6] = np.sqrt(normalizedSquaredLog - (logSum**2 / (num_pixels**2)))
    # normalize abs relative
    errors[7] = errors[7] / num_pixels
    # normalize squared relative
    errors[8] = errors[8] / num_pixels
    return errors

def evaluate_depth(label_path,
             result_path,
             scale=256.0):
    gt_list = os.listdir(label_path)
    gt_list.sort()
    gt_list = [os.path.join(label_path, gt) for gt in gt_list if gt.endswith(".png")]

    result_list = os.listdir(result_path)
    result_list.sort()
    result_list = [os.path.join(result_path, result) for result in result_list if result.endswith(".png")]

    if not len(gt_list) == len(result_list):
        print("Notice: the lenght of gt_list {} is not the same as the result_list {}".format(len(gt_list), len(result_list)))
    print("totally found {} images in {} and {}".format(len(gt_list), label_path, result_path))
    error_vectors = []
    for i in range(len(gt_list)):
        image_gt = cv2.imread(gt_list[i], -1) / scale
        image_pred = cv2.imread(result_list[i], -1) / scale
        error_vectors.append(compute_errors(image_gt, image_pred))
    error_vectors = np.array(error_vectors)
    metric_names = [
        "mae", 
        "rmse", 
        "inverse mae", 
        "inverse rmse", 
        "log mae", 
        "log rmse", 
        "scale invariant log", 
        "abs relative", 
        "squared relative"
    ]
    result_texts = []
    for i in range(len(error_vectors[0])):
        text = "mean {} : {}\n".format(metric_names[i], np.mean(error_vectors[:, i]))
        result_texts.append(text)
    return result_texts

if __name__ == "__main__":
    from fire import Fire
    def main(label_path,
             result_path):
        texts = evaluate(label_path, result_path)
        for text in texts:
            print(text, end="")
    Fire(main)
