import torch
from torchmetrics import Accuracy, F1Score
from configs import BaseConfig
import sacrebleu
import torch.nn.functional as F
from datasets.translation import PAD_ID

class TranslationMetrics:
    def __init__(self, config: BaseConfig):
        self.accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=PAD_ID).to(config.device)
        self.f1_metric = F1Score(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=PAD_ID).to(config.device)
    
    def compute_bleu(self, preds, refs):
        preds = [" ".join(map(str, pred)) for pred in preds]  # 将预测序列转为字符串
        refs = [[" ".join(map(str, ref))] for ref in refs]  # 引入多参考格式
        bleu = sacrebleu.corpus_bleu(preds, refs)
        return bleu.score

    def __call__(self, pred):
        labels = torch.tensor(pred.label_ids)  # (batch_size, seq_len)
        logits = torch.tensor(pred.predictions)  # (batch_size, seq_len, vocab_size)
        preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        mask = (labels != PAD_ID)
        filtered_preds = preds[mask]
        filtered_labels = labels[mask]

        # 词级别指标
        accuracy = self.accuracy_metric(filtered_preds, filtered_labels)
        f1 = self.f1_metric(filtered_preds, filtered_labels)

        # 句子级别 BLEU
        bleu_score = self.compute_bleu(preds.cpu().tolist(), labels.cpu().tolist())

        return {
            'accuracy': accuracy.item(),
            'f1_score': f1.item(),
            'bleu_score': bleu_score,
        }
        
class WordLevelMetrics:
    def __init__(self, config: BaseConfig):
        self.accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=PAD_ID).to(config.device)
        self.f1_metric = F1Score(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=PAD_ID).to(config.device)

    def compute(self, preds, labels, mask):
        filtered_preds = preds[mask]
        filtered_labels = labels[mask]
        accuracy = self.accuracy_metric(filtered_preds, filtered_labels)
        f1 = self.f1_metric(filtered_preds, filtered_labels)
        return accuracy, f1

class SentenceLevelMetrics:
    @staticmethod
    def compute_bleu(preds, refs):
        preds = [" ".join(map(str, pred)) for pred in preds]
        refs = [[" ".join(map(str, ref))] for ref in refs]
        bleu = sacrebleu.corpus_bleu(preds, refs)
        return bleu.score
