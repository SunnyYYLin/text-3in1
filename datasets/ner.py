import json
import os
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

PADDING_IDX = 4832
OUTSIDE_TAG = 0
    
class NERDataCollator(DataCollatorMixin):
    """封装了自定义的 collate_fn, 用于 Hugging Face 的 Trainer。"""
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)

        # 创建填充后的张量
        word_ids = torch.full((batch_size, max_seq_len), PADDING_IDX, dtype=torch.long)
        tag_ids = torch.full((batch_size, max_seq_len), OUTSIDE_TAG, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        # 填充每个序列
        for i, (word, tag) in enumerate(zip(input_ids, labels)):
            word_ids[i, :len(word)] = word
            tag_ids[i, :len(tag)] = tag
            attention_mask[i, :len(word)] = True

        return {
            "input_ids": word_ids,
            "attention_mask": attention_mask,
            "labels": tag_ids
        }

class NERDataset(Dataset):
    def __init__(self, path: str, part: str):
        # 加载数据和词汇表
        data_path = os.path.join(path, f'{part}.txt')
        chr_vocab_path = os.path.join(path, 'chr_vocab.json')
        tag_vocab_path = os.path.join(path, 'tag_vocab.json')
        self.chr_vocab = json.load(open(chr_vocab_path, 'r', encoding='utf-8'))
        self.tag_vocab = json.load(open(tag_vocab_path, 'r', encoding='utf-8'))
        self.data = self.load_data(data_path)
        print(f"Loaded {len(self.data)} samples from {data_path}.")
        
    def load_data(self, data_path: str):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            sentence, tags = [], []
            for line in f:
                if line.strip() == '':
                    # 完整句子结束，存储到数据集中
                    if sentence:
                        data.append({"sentence": sentence, "tags": tags})
                        sentence, tags = [], []
                else:
                    word, tag = line.strip().split('\t')
                    sentence.append(word)
                    tags.append(tag)
            # 处理最后一个句子（如果存在）
            if sentence:
                data.append({"sentence": sentence, "tags": tags})
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        """根据索引获取句子及其标签，并将它们转换为 ID。"""
        sample = self.data[index]
        char_ids = [self.chr_vocab[char] for char in sample["sentence"]]
        tag_ids = [self.tag_vocab[tag] for tag in sample["tags"]]
        return {
            "input_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(tag_ids, dtype=torch.long)
        }
