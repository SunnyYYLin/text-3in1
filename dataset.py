import random
import json
import torch
from torch.utils.data import Dataset

def collate_fn(data):
    pad_idx = 4999
    texts = [d[0] for d in data]
    label = [d[1] for d in data]
    batch_size = len(texts)
    max_length = max([len(t) for t in texts])
    text_ids = torch.ones((batch_size, max_length)).long().fill_(pad_idx)
    label_ids = torch.tensor(label).long()
    for idx, text in enumerate(texts):
        text_ids[idx, :len(text)] = torch.tensor(text)
    return text_ids, label_ids

class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path) -> None:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
            random.shuffle(raw_data)
            
        data = []
        for item in raw_data:
            text = item['text']
            text_id = [self.vocab[t] if t in self.vocab.keys() else self.vocab['UNK'] for t in text]
            label = int(item['label'])
            data.append([text_id, label])
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]