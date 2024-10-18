import time 
import torch 
from utils import tools
from utils.top_k import top_k, reformat_top_k
import wandb 

class Trainer:
    def __init__(self, model, optimizer, loss, max_epoch, device, target_measure, 
                 emd_metric, logging=False, save_best=False):
        self.model = model 
        self.optimizer = optimizer
        self.loss = loss
        self.max_epoch = max_epoch
        self.device = device 
        self.target_measure = target_measure
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        self.emd_metric = emd_metric
        self.logging = logging # whether logging on the cloud server, using wandb here.
        self.traj_embs = None 
        self.save_best = save_best
        
        assert self.emd_metric is not None, "emd_metric should be specified"
        # Generate a unique identifier for this training.
        self.train_identifier = f"{self.target_measure}_{self.emd_metric}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        
        print(f"[Trainer Initialization] Target Metric: {target_measure.upper()}, Embedding Metric: {emd_metric.upper()}")
        print(f"[Trainer Initialization] Identifier: {self.train_identifier}")
        if self.save_best: print(f"[Trainer Initialization] Saving mode is open. Embeddings, Model Checkpoint will be saved")
        if self.logging: wandb.run.summary["train_identifier"] = self.train_identifier
        
        
    def run(self, train_dataloader, eval_dataloader, test_dataloader):
        best_score = 0
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train(train_dataloader)
            # evaluation 
            eval_score = self.eval(eval_dataloader, stage="Eval")
            if eval_score[50] > best_score:                
                best_score = eval_score[50]
                print(f"[Epoch {self.epoch}] Best Eval Score Updated, Test Model on Test Dataset...")
                # test when a better model is found
                test_score = self.eval(test_dataloader, stage="Test")
                if self.logging:
                    wandb.log(step=self.epoch, data= reformat_top_k(test_score, self.target_measure))        
                    wandb.run.summary["best_score"] = reformat_top_k(test_score, self.target_measure)
        # save model
        if self.save_best:
            tools.save_model(path=f"./checkpoints/SIMformer/{self.train_identifier}", 
                             epoch=self.checkpoint_epoch, model_states=self.model_checkpoint, embs=self.traj_embs)
            
            
        
    
    def train(self, train_dataloader):
        dataloader = train_dataloader.get_dataloader()
        start_time = time.time()
        self.model.train() 
        epoch_loss = 0 
        for batch in dataloader:
            batch_trajs, batch_distances, batch_padding_masks  = batch
            # move data to gpu 
            batch_trajs = tuple(item.to(self.device) for item in batch_trajs)
            
            batch_padding_masks = tuple(item.to(self.device) for item in batch_padding_masks)
            
            
            batch_distances = batch_distances.to(self.device)
            anchor_vec = self.model.forward(batch_trajs[0], batch_padding_masks[0])
            target_vec = self.model.forward(batch_trajs[1], batch_padding_masks[1])
            loss = self.loss(anchor_vec, target_vec, batch_distances) # mse loss 
                
            epoch_loss += loss.item()  # total loss of this epoch  
            
            #################### Gradient descent and backpropagation  ####################
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)  # Prevent gradient explosion.
            self.optimizer.step()
        
        end_time = time.time()  
        print(f"[Epoch {self.epoch}] Epoch Loss:{epoch_loss}, Train Time:{end_time-start_time}")
        
        if self.logging: wandb.log(step=self.epoch, data= {"epoch_loss":epoch_loss,"train_time":end_time-start_time})        
        
        
        
    def eval(self, eval_dataloader, stage="Eval"):
        dataloader = eval_dataloader.get_dataloader()
        dis_matrix = eval_dataloader.get_dis_matrix()
        # 1. Obtain trajectory representation vector
        self.model.eval()
        emds = torch.zeros(len(dataloader.dataset), self.model.hidden_dim).to(self.device) # batch_size * hidden_dim
        with torch.no_grad():
            for batch in dataloader:
                moved_batch = tuple(item.to(self.device) for item in batch)
                trajs, padding_masks, IDs = moved_batch
                traj_vecs = self.model.forward(trajs, padding_masks)
                emds[IDs] = traj_vecs
                
        # 2. calculate top-k acc
        topk_acc = top_k(emds.cpu(), dis_matrix, metric=self.emd_metric)
        
        if stage == "Eval":
            print(f"[Epoch {self.epoch}] {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")
            
        if stage == "Test":
            if self.save_best:  
                self.traj_embs = emds.cpu()
                self.model_checkpoint = self.model.state_dict()
                self.checkpoint_epoch = self.epoch
            print(f"[Epoch {self.epoch}] |-> {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")
        
        return topk_acc
