import torch
from .similarity import emds2sim
import numpy as np

def top_k(emds, ground_truth, metric="euc"):
    """
    emds: tensor, shape=(n, d),  n is the number of samples, d is the feature dimension (128 in this study)
    ground_truth: tensor, shape=(n, n), ground_distance_matrix, ground_truth[i, j] is the distance between i-th and j-th samples
    """

    query_size = ground_truth.shape[0] 
    ground_truth_similarity = -ground_truth # convert distance to similarity      
    # initialize the topk list
    topk_list = [1, 5, 10, 20, 50]
    total_hits = {k: 0 for k in topk_list}
    total_hits["R10@50"] = 0 
    
    grd_dis_sorted = torch.argsort(ground_truth_similarity, descending=True, dim=1) 
    predict_sim_matrix = emds2sim(emds, metric=metric)  # calculate the predicted similarity matrix
    pre_dis_sorted = torch.argsort(predict_sim_matrix, descending=True, dim=1)
    
    # Calculate the number of hits.
    for i in range(query_size):
        grd_local = grd_dis_sorted[i, 1:]  # exclude self
        pre_local = pre_dis_sorted[i, 1:]  
        for k in topk_list:
            pred_top_k = pre_local[:k]
            real_top_k = grd_local[:k]
            intersection = np.intersect1d(pred_top_k, real_top_k)
            total_hits[k] += len(intersection)
        r10_50 = np.intersect1d(pre_local[:50], grd_local[:10])  # calculate the hit number for  R10@50 
        total_hits["R10@50"] += len(r10_50) 

    # Calculate the hit rate.
    for k in topk_list:
        total_hits[k] = total_hits[k]/(k*query_size)
    total_hits["R10@50"] = total_hits["R10@50"]/(10*query_size) 
    # The precision is 6 decimal.
    for k in total_hits.keys():
        total_hits[k] = round(total_hits[k], 6)
    return total_hits


def reformat_top_k(topk_acc, target_metric):
    hr1, hr5, hr10, hr20, hr50, r10_50 = topk_acc[1], topk_acc[5], topk_acc[10], topk_acc[20], topk_acc[50], topk_acc["R10@50"]
    
    target = target_metric.upper()
    score = {"{}-HR@1".format(target):hr1,
                                "{}-HR@5".format(target):hr5,
                                "{}-HR@10".format(target):hr10,
                                "{}-HR@20".format(target):hr20,
                                "{}-HR@50".format(target):hr50, 
                                "{}-R10@50".format(target):r10_50}
    return score
