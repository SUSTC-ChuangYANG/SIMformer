import torch
from torch import nn
import math 

# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0, batch_first=False):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first 
        pe = torch.zeros(max_len, d_model)  # positional encoding, (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2) 
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_len, d_model/2)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1) # (1, max_len, d_model) -> (max_len, 1, d_model)
        
        if self.batch_first:
            pe = pe.permute(1, 0, 2)  # (max_len, 1, d_model) -> (1, max_len, d_model) 
            
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        print(f"[PositionalEncoding] Fixed: d_model={d_model}, max_len={max_len}, batch_first={batch_first}")
        
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, batch_first=False):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        if self.batch_first:
            self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        else:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)
        print(f"[PositionalEncoding] Learnable: d_model={d_model}, max_len={max_len}, batch_first={batch_first}")
        

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))