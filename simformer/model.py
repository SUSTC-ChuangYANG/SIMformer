import torch 
import torch.nn as nn
from .pos_encoding import get_pos_encoder

def mean_pooling(x, padding_masks):
    """
    input: batch_size, seq_len, hidden_dim 
    output: batch_size, hidden_dim 
    """
    x = x*padding_masks.unsqueeze(-1)
    x = torch.sum(x, dim=1)/torch.sum(padding_masks, dim=1).unsqueeze(-1)  #  mean pooling excluding the padding part. 
    return x

class SIMformer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, dimfeedforward=256, n_heads=16, num_layers=1, 
                 pos_encoding="learnable"):
        """ SIMformer, a 1-layer vanilla siamese transformer, Encoder Only
        Args:
            - feat_dim (int): Input feature dimension, for trajectory data, feat_dim=2, which is longitude and latitude
            - hidden_dim (int): Dimension after linear mapping of longitude and latitude, which is the input dimension of transformer
            - dimfeedforward (int): Dimension of feedforward network in transformer
            - n_heads (int): Number of heads in multi-head attention in transformer
            - num_layers (int): Number of layers in transformer
            - pos_encoding (str): Position encoding method, can be learnable or sin_cos
        """
        super(SIMformer, self).__init__()
        print("[SIMformer] The Input Dim: {}.".format(feat_dim))
        print("[SIMformer] The Encoder Dim: {}.".format(hidden_dim))
        self.hidden_dim = hidden_dim
        self.max_seq_len = 200 # set according to the maximum trajectory length in the dataset, and used for positional encoding
        self.project_inp = nn.Linear(feat_dim, hidden_dim) # point dimensionality enhancement
        self.pos_enc = get_pos_encoder(pos_encoding)(hidden_dim, max_len=self.max_seq_len, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = n_heads, dim_feedforward= dimfeedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # This is to ensure that the output is a positive number, 
        # controlling the range of similarity between 0 and 1, because sometimes cosine similarity is used.
        self.act = nn.ReLU()
        
        print("[SIMformer] The Position Encoding Method: {}.".format(pos_encoding))
        print("[SIMformer] Number of Transformer Layers: {}.".format(num_layers))
        print("[SIMformer] Number of Transformer Heads: {}.".format(n_heads))
        print("[SIMformer] The Feedforward Dimension: {}.".format(dimfeedforward))
    
    def forward(self, x, padding_masks):
        input = self.project_inp(x)
        input = self.pos_enc(input)  # add positional encoding
        output = self.transformer_encoder(input, src_key_padding_mask=~padding_masks)
        output = mean_pooling(output, padding_masks)
        output = self.act(output) 
        return output
        
    
