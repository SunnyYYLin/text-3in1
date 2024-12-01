import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin

UNK = 'UNK'
PAD_ID = 8019

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
    """
    SentimentDataset is a custom dataset class for sentiment analysis tasks. It loads and shuffles data from a JSONL file,
    Each sample {'input_ids':list[int], 'label':int} in the dataset contains tokenized text converted to IDs and a label.
    
    Attributes:
        vocab (dict[str, int]): A dictionary mapping words to their corresponding IDs.
        data (list[dict]): A list of dictionaries containing tokenized text and labels.
    Methods:
        __init__(path: str, part: str) -> None:
            Initializes the SentimentDataset with the given path and part.
        load_data(data_path: str) -> list[dict]:
            Loads and shuffles the data from the specified JSONL file, converts text to IDs, and stores labels.
        __len__() -> int:
            Returns the number of samples in the dataset.
        __getitem__(index: int) -> dict:
            Returns the sample at the specified index.
    """
    
    def __init__(self, path: str, part: str) -> None:
        vocab_path = os.path.join(path, 'vocab.json')
        data_path = os.path.join(path, f'{part}.jsonl')
        # 读取词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab: dict[str, int] = json.load(f)
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.token2id = self.vocab
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path) -> list[dict]:
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
        return ''.join([self.id2token[idx.item()] for idx in input_ids if idx != PAD_ID])
    
    def decode_label(self, label: torch.LongTensor) -> str:
        return 'positive' if label.item() == 1 else 'negative'