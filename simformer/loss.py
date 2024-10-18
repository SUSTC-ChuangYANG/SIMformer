from torch.nn import Module
from torch import nn
import torch    
import numpy as np
from utils.similarity import pairwise_emd2sim
    

class SimpleMSELoss(Module):
    def __init__(self, emb_sim_metric="euc"):
        super(SimpleMSELoss, self).__init__()
        self.emb_sim_metric = emb_sim_metric
        self.mse = nn.MSELoss()
        print(f"[Loss Function] Using SimpleMSELoss with {self.emb_sim_metric} as the similarity metric")
        
    def forward(self, traj_vecs1, traj_vecs2, target_dist):
        pred_dist = pairwise_emd2sim(traj_vecs1, traj_vecs2, self.emb_sim_metric)
        return self.mse(pred_dist, target_dist)
    

