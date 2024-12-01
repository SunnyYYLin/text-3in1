import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin

PAD_ID = 4832
O_ID = 0
    
class NERDataCollator(DataCollatorMixin):
    """封装了自定义的 collate_fn, 用于 Hugging Face 的 Trainer。"""
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
        attention_mask = (padded_input_ids != PAD_ID)
        labels = pad_sequence(labels, batch_first=True, padding_value=O_ID)      

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class NERDataset(Dataset):
    def __init__(self, path: str, part: str):
        # 加载数据和词汇表
        data_path = os.path.join(path, f'{part}.txt')
        chr_vocab_path = os.path.join(path, 'chr_vocab.json')
        tag_vocab_path = os.path.join(path, 'tag_vocab.json')
        self.chr_vocab = json.load(open(chr_vocab_path, 'r', encoding='utf-8'))
        self.chr_id2token = {v: k for k, v in self.chr_vocab.items()}
        self.chr_token2id = self.chr_vocab
        self.tag_vocab = json.load(open(tag_vocab_path, 'r', encoding='utf-8'))
        self.tag_id2token = {v: k for k, v in self.tag_vocab.items()}
        self.tag_token2id = self.tag_vocab
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

    def decode_input(self, input_ids: torch.LongTensor) -> str:
        return ' '.join([self.chr_id2token[idx] for idx in input_ids.tolist() if idx != PAD_ID])
    
    def decode_label(self, label_ids: torch.LongTensor, attn_mask: torch.BoolTensor) -> str:
        return ' '.join([self.tag_id2token[idx] for idx, mask in zip(label_ids.tolist(), attn_mask.tolist()) if mask])