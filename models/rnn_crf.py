import torch
import torch.nn as nn
from .crf import CRF
from configs.rnn_config import TextRNNConfig
from .ner_rnn import NER_RNN

class RNN_CRF(nn.Module):
    def __init__(self, config: TextRNNConfig) -> None:
        super(RNN_CRF, self).__init__()
        self.rnn = NER_RNN(config)
        gpu = (config.device != 'cpu')
        self.crf = CRF(config.num_classes - 2, gpu)
    
    def forward(self, input_ids: torch.LongTensor,
                attention_mask: torch.Tensor|None=None,
                labels: torch.LongTensor|None=None) -> torch.Tensor:
        logits = self.rnn(input_ids, attention_mask)
        if labels is not None:
            loss = self.crf(logits, attention_mask, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}
    
    def predict(self, input_ids: torch.LongTensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.rnn(input_ids, attention_mask)
        return self.crf.decode(logits, attention_mask)
