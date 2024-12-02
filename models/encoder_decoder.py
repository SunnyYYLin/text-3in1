import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import TransformerConfig

class EncoderDecoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(EncoderDecoder, self).__init__()
        self.only_one = config.only_one
        self.pos_encoder = PositionalEncoding(
            d_model=config.emb_dim
        )
        self.transformer = nn.Transformer(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            batch_first=True  # Set batch_first to True for consistency
        )
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor=None,
                src_mask: torch.Tensor=None,
                tgt_mask: torch.Tensor=None,
                src_key_padding_mask: torch.Tensor=None,
                tgt_key_padding_mask: torch.Tensor=None) -> torch.Tensor:
        src = self.pos_encoder(src) # (batch_size, src_seq_len, emb_dim)
        tgt = self.pos_encoder(tgt) if tgt is not None else None # (batch_size, tgt_seq_len, emb_dim)
        # Transformer backbone
        output_emb = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        ) # (batch_size, tgt_seq_len, emb_dim)
        
        return output_emb
    
    def generate_square_subsequent_mask(self, sz: int):
        return self.transformer.generate_square_subsequent_mask(sz)
    
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