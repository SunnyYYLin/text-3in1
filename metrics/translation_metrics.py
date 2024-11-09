import torch
from torchmetrics import Accuracy, F1Score
from configs import BaseConfig

import torch.nn.functional as F

class TranslationMetrics:
    def __init__(self, config: BaseConfig):
        self.accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.tgt_vocab_size).to(config.device)
        self.f1_metric = F1Score(task="multiclass", 
            num_classes=config.tgt_vocab_size).to(config.device)
    
    def __call__(self, pred):
        labels = torch.tensor(pred.label_ids) # (batch_size, seq_len)
        logits = torch.tensor(pred.predictions) # (batch_size, seq_len, vocab_size)
        preds = torch.argmax(logits, dim=-1) # (batch_size, seq_len)

        accuracy = self.accuracy_metric(preds, labels)
        
        return {
            'accuracy': accuracy.item(),
        }