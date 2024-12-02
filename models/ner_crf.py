import torch
import torch.nn as nn
from .crf import CRF
from configs import PipelineConfig
from .backbone import get_backbone

class NER_CRF(nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super(NER_CRF, self).__init__()
        model_cls = get_backbone(config)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=-1
        )
        self.backbone = model_cls(config)
        sizes = config.mlp_dims + [config.num_tags + 2]
        mlp_layers = [nn.LazyLinear(out_features=size) for size in sizes]
        activations = [nn.ReLU() for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        dropouts = [nn.Dropout(p=config.dropout) for _ in range(len(mlp_layers) - 1)] + [nn.Identity()]
        self.classifier = nn.Sequential(*[
            layer for layers in zip(mlp_layers, activations, dropouts) for layer in layers
        ])
        self.crf = CRF(config.num_tags, config.device != 'cpu')
        self._init_lazy()
        
    def _init_lazy(self):
        dummy_input = torch.zeros((1, 32), dtype=torch.long)
        self.forward(dummy_input)
    
    def forward(self, input_ids: torch.LongTensor,
                attention_mask: torch.Tensor|None=None,
                labels: torch.LongTensor|None=None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        features = self.backbone(emb, attention_mask)
        logits = self.classifier(features)
        if labels is not None:
            loss = self.crf(logits, attention_mask, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}