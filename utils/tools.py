import os 
import torch 

def save_model(path, epoch, model_states, embs):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    target_path = os.path.join(path,'model_checkpoint_{}.tar'.format(epoch))
    torch.save({
        'epoch':epoch,
        'model_state_dict':model_states,
        "embs": embs},
         target_path
        )
    print(f"[Model Saving] Save Model Checkpoint at Epoch {epoch} in {target_path}")
    
    
def para_nums(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Initialization] Total number of trainable parameters: {total_params}")
    return total_params