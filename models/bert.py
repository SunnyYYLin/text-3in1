import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import TransformerConfig

class BERT(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(BERT, self).__init__()
        self.only_one = config.only_one
        self.pos_encoder = PositionalEncoding(
            d_model=config.emb_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers
        )
    
    def forward(self, emb: torch.Tensor, 
                attention_mask: torch.BoolTensor|None=None) -> torch.Tensor:
        attention_mask = ~attention_mask if attention_mask is not None else None
        
        emb = self.pos_encoder(emb) # (batch_size, seq_len, emb_dim)
        output = self.encoder(
                    emb, 
                    src_key_padding_mask=attention_mask
                ) # (batch_size, seq_len, emb_dim)
        
        if self.only_one:
            output = output.max(dim=1).values  # (batch_size, emb_dim)
        
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x