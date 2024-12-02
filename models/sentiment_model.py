import torch
import torch.nn as nn
from configs import PipelineConfig
from .backbone import get_backbone

class SentimentModel(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(SentimentModel, self).__init__()
        model_cls = get_backbone(config)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size + 2,
            embedding_dim=config.emb_dim,
            padding_idx=config.PAD_ID
        )
        self.backbone = model_cls(config)
        sizes = config.mlp_dims + [config.num_classes]
        mlp_layers = [nn.LazyLinear(out_features=size) for size in sizes]
        activations = [nn.ReLU() for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        dropouts = [nn.Dropout(p=config.dropout) for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        self.classifier = nn.Sequential(*[
            layer for layers in zip(mlp_layers, activations, dropouts) for layer in layers
        ])
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
