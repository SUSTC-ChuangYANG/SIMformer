# Tool components for calculating similarity
# 1. pairwise_emd2sim: Given pairs of embedding vectors, calculate their pairwise similarity
# 2. emds2sim: Given embedding vectors, calculate the similarity matrix

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch 
from torch import nn


def chebyshev_distance(v1, v2):
    """
    Compute the Chebyshev distance between two batches of vectors.

    Parameters:
    - A (torch.Tensor): Tensor of shape (batch_size, hidden_dim)
    - B (torch.Tensor): Tensor of shape (batch_size, hidden_dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size,) containing the Chebyshev distances
    """
    # calculate the absolute difference between v1 and v2
    diffs = torch.abs(v1 - v2)
    # following the direction of hidden_dim, calculate the maximum value of the absolute difference
    max_diffs = torch.max(diffs, dim=-1).values
    
    return max_diffs

def chebyshev_distance_matrix(X):
    """
    Compute the Chebyshev distance matrix for a batch of vectors.
    Parameters:
    - X (torch.Tensor): Tensor of shape (batch_size, hidden_dim)
    Returns:
    - torch.Tensor: Distance matrix of shape (batch_size, batch_size),
                    where element [i, j] is the Chebyshev distance between
                    vectors X[i] and X[j].
    """
    # expand the dimension of X to shape (batch_size, 1, hidden_dim) 
    X_expanded = X.unsqueeze(1)
    # expand the dimension of X to shape (1, batch_size, hidden_dim)
    X_tiled = X.unsqueeze(0)
    # calculate the absolute difference between X_expanded and X_tiled along the hidden_dim dimension
    diffs = torch.abs(X_expanded - X_tiled)
    # calculate the Chebyshev distance matrix on hidden_dim dimension
    distance_matrix = torch.max(diffs, dim=2).values
    return distance_matrix
        
def pairwise_emd2sim(v1, v2, metric="euc"):
    """
    Given two batches of vectors, calculate the pairwise similarity between them. Supports cos, euc, chebyshev.
    - v1: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
    - v2: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
   - metric: str, method of similarity calculation, named Representation Similarity Function in our paper 
        - cos: Cosine similarity
        - euc: Euclidean distance
        - chebyshev: Chebyshev distance
    """ 
    cos_sim = nn.CosineSimilarity(dim=-1) 
    if metric == "cos":
        sim_score = cos_sim(v1, v2)
    if metric == "euc":
        predict_dis_matrix = torch.norm((v1 - v2), p=2, dim=-1)
        sim_score = torch.exp(-predict_dis_matrix) # Map to a similarity space of 0-1
    if metric == "chebyshev":
        sim_score = torch.exp(-chebyshev_distance(v1, v2))
    return sim_score    

def emds2sim(emds, metric="euc"):
    """
    Given embedding vectors, calculate the similarity matrix. Supports cos, euc, chebyshev.
    - emds: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
    - metric: str, method of similarity calculation, named Representation Similarity Function in our paper 
        - cos: Cosine similarity
        - euc: Euclidean distance
        - chebyshev: Chebyshev distance
    """
    if metric == "cos":
        sim_matrix = pairwise_cosine_similarity(emds, zero_diagonal=False)
    if metric == "euc":
        predict_dis_matrix = torch.cdist(emds, emds, p=2)
        sim_matrix = torch.exp(-predict_dis_matrix) # Map to a similarity space of 0-1
    if metric == "chebyshev":
        sim_matrix = torch.exp(-chebyshev_distance_matrix(emds))
    
    return sim_matrix
