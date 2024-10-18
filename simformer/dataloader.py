from torch.utils.data import Dataset, DataLoader
import torch 
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class TransformerDataset(Dataset):
    def __init__(self, trajs, data_range, mode="train"):
        super(TransformerDataset, self).__init__()
        if data_range is None:
            raise ValueError("Data Range is None!")
        self.mean_lon = data_range["mean_lon"] 
        self.mean_lat = data_range["mean_lat"]
        self.std_lon = data_range["std_lon"]
        self.std_lat = data_range["std_lat"] 
        print(f"[Data Preparation] Creating Transformer Dataset for {mode} ...")
        
        self.trajs = self.trajs_normalize(trajs) 
        
        self.IDs = [i for i in range(len(self.trajs))]   # list of data IDs, but also mapping between integer index and ID
        
        print(f"[Data Preparation] Totally {len(self.trajs)} samples prepared in Transformer Dataset.")
        
    def trajs_normalize(self, trajs):
        """
        apply z-score normalization to the trajectory data
        """
        new_trajs = []
        for traj in trajs:
            normlized_traj = [((lon-self.mean_lon)/self.std_lon, (lat-self.mean_lat)/self.std_lat) for (lon, lat) in traj]
            new_trajs.append(normlized_traj)
        print("[Data Preparation] Trajectory Z-Score Normalization Done!")
        return new_trajs    
            
    def __getitem__(self, idx):
        return torch.tensor(self.trajs[idx]), idx  
    
    def get_items(self, indices):
        return [self.__getitem__(i)[0] for i in indices]

    def __len__(self):
        return len(self.IDs)
    
    
class TransformerDataLoader():
    
    def __init__(self, dataset: tuple, data_range=None, batch_size=20, mode="train", sampling_num=20, alpha=16):
        trajs, dis_matrix = dataset 
        self.dis_matrix = dis_matrix
        self.dataset = TransformerDataset(trajs, data_range=data_range, mode=mode)
        self.sampling_num = sampling_num 
        self.batch_size = batch_size
        self.mode = mode
        self.alpha = alpha
        print(f"[Data Preparation] Creating Transformer DataLoader for {mode} ...")
        print(f"[Data Preparation] Transformer DataLoader: {len(trajs)} samples")
        self.dataloader = self.create_dataloader() 
        print(f"[Data Preparation] Transformer DataLoader using alpha={self.alpha} to normalize the distance matrix")
        print(f"[Data Preparation] Transformer DataLoader, shape of distance matrix: {self.dis_matrix.shape}")
        print("-"*100)
        
        
    def get_dataloader(self):  
        return self.dataloader
    
    def get_dis_matrix(self):
        return self.dis_matrix
    
    def random_sampling(self, idx):
        # sampling sampling_num samples randomly for each trajectory
        N = len(self.dis_matrix)
        distances = self.dis_matrix[idx] 
        similarity = torch.exp(-self.alpha*distances) # 1*N
        indices = np.random.choice(range(N), size=self.sampling_num, replace=False) 
        return indices, similarity[indices] 
    
    @staticmethod
    def creat_padding_mask(trajs):
        """Create a mask for a batch of trajectories.
        - False indicates that the position is a padding part that exceeds the original trajectory length
        - while True indicates that the position is the valid part of the trajectory.
        """
        lengths = torch.tensor([len(traj) for traj in trajs])
        max_len = max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return ~mask
        
    #test and eval dataloader 
    def my_collate_fn_test_eval(self, data):
        trajs, indices = zip(*data)
        padding_masks = TransformerDataLoader.creat_padding_mask(trajs)
        padded_trajs = pad_sequence(trajs, batch_first=True, padding_value=0)
        return padded_trajs, padding_masks, torch.tensor(indices)
    
    def my_collate_fn_train(self, data):
        _, indices = zip(*data) 

        batch_anchor_trajs = []  
        batch_target_trajs = []
        batch_distances = []
        for anchor_idx in indices:
            target_indices, sims  = self.random_sampling(anchor_idx)
            batch_distances.append(sims)
            anchor_trajs = self.dataset.get_items([anchor_idx]*(self.sampling_num))
            target_trajs = self.dataset.get_items(target_indices) 
            batch_anchor_trajs.extend(anchor_trajs)
            batch_target_trajs.extend(target_trajs)
        padding_masks_anchor = TransformerDataLoader.creat_padding_mask(batch_anchor_trajs)
        padding_masks_target = TransformerDataLoader.creat_padding_mask(batch_target_trajs)
        batch_anchor_trajs = pad_sequence(batch_anchor_trajs, batch_first=True, padding_value=0)  # (batch_size * sampling_num), max_seq_len, input_dim
        batch_target_trajs = pad_sequence(batch_target_trajs, batch_first=True, padding_value=0)  # (batch_size * sampling_num), max_seq_len, input_dim
        batch_trajs = (batch_anchor_trajs, batch_target_trajs)
        batch_padding_masks = (padding_masks_anchor, padding_masks_target)
        batch_distances = torch.cat(batch_distances, dim=0)
        return batch_trajs, batch_distances, batch_padding_masks
                    
        
    def create_dataloader(self) -> DataLoader:
        """
        given trajectory dataset and batch_size, return the corresponding DataLoader 
        """        
        pairs_num = self.batch_size*self.sampling_num  # calculate all pairs required for training
            
        if self.mode == "train":
            dataloader = DataLoader(dataset= self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=32, 
                                   pin_memory=True, collate_fn=self.my_collate_fn_train)
            print(f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples, {pairs_num} samples per batch")
            return dataloader
        
        if self.mode == "test" or self.mode == "eval": # do not need to construct pairs, just encoding 
            # To keep the batch_size consistent with training, set the batch_size to be batch_size * sampling_num.
            dataloader = DataLoader(dataset= self.dataset, batch_size=pairs_num, shuffle=False, num_workers=32, 
                                   pin_memory=True, collate_fn=self.my_collate_fn_test_eval)
            print(f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples, {pairs_num} samples per batch")
            return dataloader
    
    
 
        
        
        
        
        
        
        
        

    
    



