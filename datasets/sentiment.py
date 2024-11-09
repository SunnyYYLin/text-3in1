import json
import os
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

UNK = 'UNK'
PADDING_IDX = 8019

def collate_fn(batch: list[dict[str, ]]) -> dict[str, torch.Tensor]:
    texts: list[list[int]] = [item['input_ids'] for item in batch]
    labels: list[int] = [item['label'] for item in batch]

    batch_size = len(texts)
    max_length = max(len(text) for text in texts)

    input_ids = torch.full((batch_size, max_length), PADDING_IDX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, text in enumerate(texts):
        input_ids[i, :len(text)] = torch.tensor(text, dtype=torch.long)
        attention_mask[i, :len(text)] = True

    # 转换标签为张量
    labels = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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
    
