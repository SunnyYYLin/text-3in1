import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import CNNConfig

class TextCNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super(TextCNN, self).__init__()
        self.convs_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num,
                kernel_size=(size, config.emb_dim)
            ) for num, size in zip(config.num_filters, config.filter_sizes)
        ])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, emb: torch.Tensor, # (batch_size, seq_len, emb_dim)
                attention_mask: torch.BoolTensor|None=None) -> torch.Tensor:
        if attention_mask is not None:
            emb = emb * attention_mask.unsqueeze(-1).float() # (batch_size, seq_len, emb_dim)

        emb = emb.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        x_list = [conv(emb).squeeze(3) for conv in self.convs_layers]  # [(batch_size, num_filters, seq_len-filter_size+1)]
        x_list = [F.adaptive_max_pool1d(x, 1).squeeze(2) for x in x_list]  # [(batch_size, num_filters)]
        x = torch.cat(x_list, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        return x