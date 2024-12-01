import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin

EOS_ID = 0
GO_ID = 1
UNK_ID = 2
PAD_ID = 3
TOKEN_IDENTIFIER = '@@'
    
class TranslationDataCollator(DataCollatorMixin):
    """封装了自定义的 collate_fn, 用于 Hugging Face 的 Trainer。"""
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        src_ids = [torch.tensor(f["src_ids"]+[EOS_ID], dtype=torch.long) for f in features]
        tgt_ids = [torch.tensor([GO_ID]+f["tgt_ids"]+[EOS_ID], dtype=torch.long) for f in features]
        assert len(src_ids) == len(tgt_ids), "Mismatched source and target batch sizes."
        
        padded_src_ids = pad_sequence(src_ids, batch_first=True, padding_value=PAD_ID)
        padded_tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=PAD_ID)
        src_padding_mask = (padded_src_ids != PAD_ID)
        tgt_padding_mask = (padded_tgt_ids != PAD_ID)
        
        return {
            "src_ids": padded_src_ids, # (batch_size, src_max_len+1)
            "src_padding_mask": src_padding_mask, # (batch_size, src_max_len+1)
            "tgt_ids": padded_tgt_ids, # (batch_size, tgt_max_len+2)
            "tgt_padding_mask": tgt_padding_mask # (batch_size, tgt_max_len+2)
        }

class TranslationDataset(Dataset):
    def __init__(self, path: str, part: str):
        zh_path = os.path.join(path, f'{part}.zh')
        en_path = os.path.join(path, f'{part}.en')
        zh_vocab_path = os.path.join(path, f'train.zh.json')
        en_vocab_path = os.path.join(path, f'train.en.json')
        with open(zh_vocab_path, 'r', encoding='utf-8') as f:
            self.zh_vocab: dict[str, int] = json.load(f)
        with open(en_vocab_path, 'r', encoding='utf-8') as f:
            self.en_vocab: dict[str, int] = json.load(f)
        self.zh_id2token = {v: k for k, v in self.zh_vocab.items()}
        self.zh_token2id = self.zh_vocab
        self.en_id2token = {v: k for k, v in self.en_vocab.items()}
        self.en_token2id = self.en_vocab
        self.tgt_texts = self.load_data(zh_path)
        self.src_texts = self.load_data(en_path)
        assert len(self.src_texts) == len(self.tgt_texts), "Mismatched source and target lengths."
        
    def load_data(self, data_path: str) -> list[str]:
        with open(data_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        data = [sentence.strip().split() for sentence in sentences]
        return data
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_ids = [self.en_vocab.get(token, UNK_ID) for token in src_text]
        tgt_ids = [self.zh_vocab.get(token, UNK_ID) for token in tgt_text]
        
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
        
    def decode_src(self, ids: torch.LongTensor, attn_mask: torch.BoolTensor) -> str:
        text = " ".join(self.en_id2token[id.item()] for id, mask in zip(ids, attn_mask) if mask)
        return text.replace(TOKEN_IDENTIFIER+' ', '')
    
    def decode_tgt(self, ids: torch.LongTensor, attn_mask: torch.BoolTensor) -> str:
        text = "".join(self.zh_id2token[id.item()] for id, mask in zip(ids, attn_mask) if mask)
        return text.replace(TOKEN_IDENTIFIER, '')

if __name__ == "__main__":
    dataset = TranslationDataset(path="data/translation", part="train")
    print("Dataset size:", len(dataset))
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}:")
        print("Original Source Text:", " ".join(dataset.src_texts[i]))
        print("Original Target Text:", " ".join(dataset.tgt_texts[i]))
        print("Input IDs:", sample["src_ids"])
        print("Labels:", sample["tgt_ids"])
    
    collator = TranslationDataCollator()
    batch = [dataset[i] for i in range(3)]
    collated_batch = collator(batch)
    print("Collated batch:")
    print("Input IDs:", collated_batch["src_ids"])
    print("Attention Mask:", collated_batch["src_mask"])
    print("Labels:", collated_batch["tgt_ids"])
    print("Target Attention Mask:", collated_batch["tgt_mask"])