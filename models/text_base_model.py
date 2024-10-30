from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.base_config import BaseConfig

class TextBaseModel(nn.Module, ABC):
    def __init__(self, config: BaseConfig) -> None:
        super(TextBaseModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size + 1,
            embedding_dim=config.emb_dim,
            padding_idx=-1
        )
        self.dropout = nn.Dropout(config.dropout)
        
    def _init_lazy(self):
        dummy_input = torch.zeros((1, 32), dtype=torch.long)
        self.forward(dummy_input)
    
    @abstractmethod
    def features(self, input_ids: torch.LongTensor,
                    attention_mask: torch.LongTensor|None=None) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input_ids: torch.LongTensor,
                    attention_mask: torch.LongTensor|None=None,
                    labels: torch.LongTensor|None=None) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def predict(self, x: torch.LongTensor) -> torch.LongTensor:
        with torch.no_grad():
            logits = self.forward(x)['logits']
            tag = torch.argmax(logits, dim=-1)
            return tag
