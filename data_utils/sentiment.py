import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin

UNK = 'UNK'
PAD = '[PAD]'
CLS = '[CLS]'
SEP = '[SEP]'

def collate_fn(batch: list[dict[str, ]]) -> dict[str, torch.Tensor]:
    texts: list[list[int]] = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    labels: list[int] = [item['label'] for item in batch]

    padded_input_ids = pad_sequence(texts, batch_first=True, padding_value=PAD_ID)
    attention_mask = (padded_input_ids != PAD_ID)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": padded_input_ids, "attention_mask": attention_mask, "labels": labels}

class SentimentDataCollator(DataCollatorMixin):
    def __call__(self, features: list[dict[str, ]]) -> dict[str, torch.Tensor]:
        return collate_fn(features)

class SentimentDataset(Dataset):
    def __init__(self, config, part: str) -> None:
        path = config.data_path
        vocab_path = os.path.join(path, 'vocab.json')
        data_path = os.path.join(path, f'{part}.jsonl')
        # 读取词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab: dict[str, int] = json.load(f)
        # 特殊token加入词汇表
        global PAD_ID
        global CLS_ID
        global SEP_ID
        PAD_ID = len(self.vocab)
        CLS_ID = PAD_ID + 1
        SEP_ID = CLS_ID + 1
        config.PAD_ID = PAD_ID
        config.CLS_ID = CLS_ID
        config.SEP_ID = SEP_ID
        self.vocab[PAD] = PAD_ID
        self.vocab[CLS] = CLS_ID
        self.vocab[SEP] = SEP_ID
        
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.token2id = self.vocab
        
        self.data = self.load_data(data_path)
        config.num_classes = len(set(item['label'] for item in self.data))
        config.vocab_size = len(self.vocab)
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Vocab size: {config.vocab_size}, Num classes: {config.num_classes}")
    
    def load_data(self, data_path) -> list[dict[str, list[int]|int]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data: list[dict[str, str]] = [json.loads(line) for line in f.readlines()]
        
        # 将文本转换为 ID，并保存标签
        data: list[dict] = []
        for item in raw_data:
            text = item['text']
            text_ids = [self.vocab.get(t, self.vocab[UNK]) for t in text]
            label = int(item['label'])
            data.append({"input_ids": text_ids, "label": label})
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def decode_input(self, input_ids: torch.LongTensor) -> str:
        return ''.join([self.id2token[id.item()] for id in input_ids if id != PAD_ID])
    
    def decode_label(self, label: torch.LongTensor) -> str:
        return 'positive' if label.item() == 1 else 'negative'