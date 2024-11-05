import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import CNNConfig

class TextCNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=config.num_filters,
                kernel_size=(filter_size, config.emb_dim)
            )
            for filter_size in config.filter_sizes
        ])

    def forward(self, emb: torch.Tensor, 
                attention_mask: torch.LongTensor|None=None) -> torch.Tensor:
        emb = emb.unsqueeze(1) # (batch_size, 1, seq_len, embedding_dim)
        x_list = [conv(emb).squeeze(3) for conv in self.convs] # [(batch_size, num_filters, seq_len-filter_size+1)]
        x_list = [F.adaptive_max_pool1d(x, 1).squeeze(2) for x in x_list] # [(batch_size, num_filters)]
        x = torch.cat(x_list, dim=1) # (batch_size, num_filters * len(filter_sizes))
        return x