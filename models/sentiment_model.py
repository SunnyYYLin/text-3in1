import torch
import torch.nn as nn
from configs import PipelineConfig
from .backbone import get_backbone

PADDING_IDX = 8019

class SentimentModel(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(SentimentModel, self).__init__()
        model_cls = get_backbone(config.model)
        print(config.emb_dim)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size + 1,
            embedding_dim=config.emb_dim,
            padding_idx=8019
        )
        self.backbone = model_cls(config)
        self.classifier = nn.LazyLinear(config.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self._init_lazy()
        
    def _init_lazy(self):
        dummy_input = torch.zeros((1, 32), dtype=torch.long)
        self.forward(dummy_input)
    
    def forward(self, input_ids: torch.LongTensor, 
                attention_mask: torch.BoolTensor|None=None,
                labels: torch.LongTensor|None=None) -> dict[str, torch.Tensor]:
        emb = self.embedding(input_ids)
        features = self.backbone(emb, attention_mask)
        logits = self.classifier(features)
        if labels is not None:
            loss = self.loss(logits, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}
