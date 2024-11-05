import torch
from torchmetrics import Accuracy, F1Score
from configs import BaseConfig

class SentimentMetircs:
    def __init__(self, config: BaseConfig):
        self.accuracy_metric = Accuracy(task="binary" 
            if config.num_classes == 2 else "multiclass", 
            num_classes=config.num_classes).to(config.device)
        self.f1_metric = F1Score(task="binary" 
            if config.num_classes == 2 else "multiclass", 
            num_classes=config.num_classes).to(config.device)
    
    def __call__(self, pred):
        labels = torch.tensor(pred.label_ids)
        preds = torch.tensor(pred.predictions).argmax(dim=-1)

        accuracy = self.accuracy_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)
        return {
            'accuracy': accuracy.item(),
            'f1': f1.item(),
        }