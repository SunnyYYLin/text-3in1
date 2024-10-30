import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.cnn_config import TextCNNConfig
from .text_base_model import TextBaseModel

class SentimentCNN(TextBaseModel):
    def __init__(self, config: TextCNNConfig) -> None:
        super(SentimentCNN, self).__init__(config)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=config.num_filters,
                kernel_size=(filter_size, config.emb_dim)
            )
            for filter_size in config.filter_sizes
        ])
        self.classifier = nn.LazyLinear(config.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self._init_lazy()

    def features(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1) # (batch_size, 1, seq_len, embedding_dim)
        x_list = [conv(x).squeeze(3) for conv in self.convs] # [(batch_size, num_filters, seq_len-filter_size+1)]
        x_list = [F.adaptive_max_pool1d(x, 1).squeeze(2) for x in x_list] # [(batch_size, num_filters)]
        x = torch.cat(x_list, dim=1) # (batch_size, num_filters * len(filter_sizes))
        return x
    
    def forward(self, input_ids: torch.LongTensor, 
                attention_mask: torch.LongTensor|None=None,
                labels: torch.LongTensor|None=None) -> dict[str, torch.Tensor]:
        logits = self.classifier(self.features(input_ids))
        if labels is not None:
            loss = self.loss(logits, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}