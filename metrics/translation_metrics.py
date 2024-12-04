import torch
from torchmetrics import Accuracy, F1Score
from configs import BaseConfig
import sacrebleu
import torch.nn.functional as F
from data_utils.translation import TranslationDataset

class TranslationMetrics:
    def __init__(self, config: BaseConfig):
        self.TGT_PAD_ID = config.TGT_PAD_ID
        self.accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=config.TGT_PAD_ID).to(config.device)
        self.f1_metric = F1Score(task="multiclass", 
            num_classes=config.tgt_vocab_size, ignore_index=config.TGT_PAD_ID).to(config.device)
        self.dataset = TranslationDataset(config, 'test')
    
    def compute_bleu(self, hypos: torch.Tensor, refs: torch.Tensor):
        hypos = self.dataset.decode_tgt(hypos, keep_token=True)
        refs = self.dataset.decode_tgt(refs, keep_token=True)
        bleu = sacrebleu.corpus_bleu(hypos, [refs])
        return bleu.score

    def __call__(self, pred):
        labels = torch.tensor(pred.label_ids)[:, 1:]  # (batch_size, seq_len)
        logits = torch.tensor(pred.predictions)  # (batch_size, seq_len, vocab_size)
        preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        mask = (labels != self.TGT_PAD_ID)
        filtered_preds = preds[mask]
        filtered_labels = labels[mask]

        # 词级别指标
        accuracy = self.accuracy_metric(filtered_preds, filtered_labels)
        f1 = self.f1_metric(filtered_preds, filtered_labels)

        # 句子级别 BLEU
        bleu_score = self.compute_bleu(preds, labels)

        return {
            'accuracy': accuracy.item(),
            'f1_score': f1.item(),
            'bleu_score': bleu_score,
        }
