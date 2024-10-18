import argparse
import sys 
sys.path.append("simformer")
from config import DATASET
from simformer.dataloader import TransformerDataLoader  
from simformer.trainer import Trainer as SIMformerTrainer
from simformer.model import SIMformer
from simformer.loss import SimpleMSELoss
from utils import tools
import torch
import numpy  as np
import wandb 
import pickle as pk

def data_prepare(dataset, target_measure): 
    """
    Return train_data, eval_data, test_data, alpha
    each data is a tuple of (traj_data, dis_matrix) 
    """
    # Load the dataset in a 2:1:7 ratio.
    train_start, eval_start, test_start = 0, 2000, 3000    
    traj_data_path = DATASET[dataset]["traj_data"]
    dis_matrix_path = DATASET[dataset]["dis_matrix"][target_measure]
    traj_data = pk.load(open(traj_data_path, 'rb'))
    dis_matrix = pk.load(open(dis_matrix_path, 'rb'))
    data_range = DATASET[dataset]["data_range"]
    if target_measure == "dtw":
        dis_matrix = dis_matrix/np.max(dis_matrix[:eval_start,:eval_start]) # follow Neutraj, T3S and TMN, normalize the distance matrix by the maximum value of the training set.
    
    train_dis_matrix = torch.tensor(dis_matrix[train_start:eval_start, train_start:eval_start]).float()
    eval_dis_matrix = torch.tensor(dis_matrix[eval_start:test_start, eval_start:test_start]).float()
    test_dis_matrix = torch.tensor(dis_matrix[test_start:, test_start:]).float()
    train_data = (traj_data[train_start:eval_start],train_dis_matrix)
    eval_data = (traj_data[eval_start:test_start], eval_dis_matrix)
    test_data = (traj_data[test_start:], test_dis_matrix)
    
    # TODO. Brand New Topic for next paper: Looking for an automated method to determine the value of alpha.
    if target_measure == "fret":
        alpha = 8
    if target_measure == "haus":
        alpha = 8
    if target_measure == "dtw":
        alpha = 16 
         
    print(f"[Data Preparation] Alpha for {target_measure.upper()}: {alpha}")
    print(f"[Data Preparation] Loading {dataset.upper()}-{target_measure.upper()} Dataset")
    print(f"[Data Preparation] Train: {len(train_data[0])}, Eval: {len(eval_data[0])}, Test: {len(test_data[0])}")
    print("-"*100)
    
    return train_data, eval_data, test_data, alpha, data_range

def go(args):
     # data prepare 
    train_dataset, eval_dataset, test_dataset, alpha, data_range = data_prepare(args.dataset, args.target_measure)
    train_dataloader = TransformerDataLoader(dataset=train_dataset, data_range= data_range, batch_size=args.batch_size, mode="train", alpha=alpha, sampling_num=args.sampling_num)
    eval_dataloader = TransformerDataLoader(dataset=eval_dataset, data_range= data_range, batch_size=args.batch_size, mode="eval", alpha=alpha, sampling_num=args.sampling_num)
    test_dataloader = TransformerDataLoader(dataset=test_dataset, data_range= data_range, batch_size=args.batch_size, mode="test", alpha=alpha, sampling_num=args.sampling_num)
    
    if args.logging: wandb.run.summary["alpha"] = alpha 
    
    # loss 
    loss = SimpleMSELoss(emb_sim_metric=args.emb_sim_metric)

    # model and gpu  
    model = SIMformer(feat_dim=2, hidden_dim=args.hidden_dim, 
                               num_layers=args.num_layers, n_heads=args.n_heads, 
                               dimfeedforward=2*args.hidden_dim,
                               pos_encoding=args.pos_encoding)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # ----------- print the model and device info -----------
    print("-"*40+"DEVICE INFO"+"-"*40)
    print(f"[Device Info] Using {device} for training")
    print("-"*40+"MODEL PRRINTER"+"-"*40)
    print(f"[Model Initialization] Using {device} for training")
    print(f"[Model Initialization] Model: {model}")
    print(f"[Model Initialization] Optimizer: Adam, Learning ratio {args.lr}")
    # Calculate the total number of trainable parameters.
    total_params = tools.para_nums(model)
    if args.logging: wandb.run.summary["total_params"] = total_params
    print("-"*100)
    # -------------------------------------------------------
    
    # Trainer intialization 
    trainer = SIMformerTrainer(model=model, optimizer=optimizer, max_epoch=args.max_epoch, 
                               loss=loss, device=device, target_measure=args.target_measure, 
                               emd_metric=args.emb_sim_metric, logging=args.logging, 
                               save_best=args.save_best)
    
    trainer.run(train_dataloader, eval_dataloader, test_dataloader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # logging 
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--logging", type=int, default=0)
    parser.add_argument("--proj_name", type=str, default="SIMformer")
    parser.add_argument("--dataset", type=str, default="porto", choices=["porto","geolife","ais","tdrive"])
    parser.add_argument("--save_best", type=int, default=1)
    
    # training params 
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=500)
    
    # model params
    parser.add_argument("--sampling_num", type=int, default=20) 
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--pos_encoding", type=str, default="learnable")
    
    # most important params 
    parser.add_argument("--target_measure", type=str, choices=["dtw","haus","fret"], default="dtw")
    parser.add_argument("--emb_sim_metric", type=str, default="euc", help="representation simialrity function", choices=["cos","euc","chebyshev"])
    
    
    args = parser.parse_args()
    # logging set up 
    args.logging = bool(args.logging)
    if not args.logging:
        print("******************** No Wandb logging in this experiment ********************")
    # config for debugging 
    args.save_best = bool(args.save_best)
    if args.logging:
        wandb.init(
            project= args.proj_name,
            config= vars(args), 
            tags = [args.dataset, args.target_measure, args.emb_sim_metric],
            name = args.exp_name
        )
      
    go(args)
    
   
    
    
