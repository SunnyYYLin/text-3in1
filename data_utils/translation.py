import json
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin

SRC_EOS = '<EOS>'
SRC_GO = '<GO>'
SRC_UNK = '<UNK>'
SRC_PAD = '<PAD>'
TGT_EOS = '<EOS>'
TGT_GO = '<GO>'
TGT_UNK = '<UNK>'
TGT_PAD = '<PAD>'
TOKEN_IDENTIFIER = '@@'
    
class TranslationDataCollator(DataCollatorMixin):
    """封装了自定义的 collate_fn, 用于 Hugging Face 的 Trainer。"""
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        src_ids = [torch.tensor(f["src_ids"]+[SRC_EOS_ID], dtype=torch.long) for f in features]
        tgt_ids = [torch.tensor([TGT_GO_ID]+f["tgt_ids"]+[TGT_EOS_ID], dtype=torch.long) for f in features]
        assert len(src_ids) == len(tgt_ids), "Mismatched source and target batch sizes."
        
        padded_src_ids = pad_sequence(src_ids, batch_first=True, padding_value=SRC_PAD_ID)
        padded_tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=TGT_PAD_ID)
        src_padding_mask = (padded_src_ids != SRC_PAD_ID)
        tgt_padding_mask = (padded_tgt_ids != TGT_PAD_ID)
        
        return {
            "src_ids": padded_src_ids, # (batch_size, src_max_len+1)
            "src_padding_mask": src_padding_mask, # (batch_size, src_max_len+1)
            "tgt_ids": padded_tgt_ids, # (batch_size, tgt_max_len+2)
            "tgt_padding_mask": tgt_padding_mask # (batch_size, tgt_max_len+2)
        }

class TranslationDataset(Dataset):
    def __init__(self, config, part: str):
        path = config.data_path
        src_path = path / f'{part}.{config.src_lang}'
        tgt_path = path / f'{part}.{config.tgt_lang}'
        src_vocab_path = path / f'train.{config.src_lang}.json'
        tgt_vocab_path = path / f'train.{config.tgt_lang}.json'
        with open(src_vocab_path, 'r', encoding='utf-8') as f:
            self.src_vocab: dict[str, int] = json.load(f)
        with open(tgt_vocab_path, 'r', encoding='utf-8') as f:
            self.tgt_vocab: dict[str, int] = json.load(f)
            
        global SRC_EOS_ID
        global SRC_GO_ID
        global SRC_UNK_ID
        global SRC_PAD_ID
        global TGT_EOS_ID
        global TGT_GO_ID
        global TGT_UNK_ID
        global TGT_PAD_ID
        SRC_EOS_ID = self.src_vocab[SRC_EOS]
        SRC_GO_ID = self.src_vocab[SRC_GO]
        SRC_UNK_ID = self.src_vocab[SRC_UNK]
        SRC_PAD_ID = self.src_vocab[SRC_PAD]
        TGT_EOS_ID = self.tgt_vocab[TGT_EOS]
        TGT_GO_ID = self.tgt_vocab[TGT_GO]
        TGT_UNK_ID = self.tgt_vocab[TGT_UNK]
        TGT_PAD_ID = self.tgt_vocab[TGT_PAD]
        config.SRC_EOS_ID = SRC_EOS_ID
        config.SRC_GO_ID = SRC_GO_ID
        config.SRC_UNK_ID = SRC_UNK_ID
        config.SRC_PAD_ID = SRC_PAD_ID
        config.TGT_EOS_ID = TGT_EOS_ID
        config.TGT_GO_ID = TGT_GO_ID
        config.TGT_UNK_ID = TGT_UNK_ID
        config.TGT_PAD_ID = TGT_PAD_ID
        config.src_vocab_size = len(self.src_vocab)
        config.tgt_vocab_size = len(self.tgt_vocab)
            
        self.src_id2token = {v: k for k, v in self.src_vocab.items()}
        self.src_token2id = self.src_vocab
        self.tgt_id2token = {v: k for k, v in self.tgt_vocab.items()}
        self.tgt_token2id = self.tgt_vocab
        self.tgt_texts = self.load_data(tgt_path)
        self.src_texts = self.load_data(src_path)
        config.src_vocab_size = len(self.src_vocab)
        config.tgt_vocab_size = len(self.tgt_vocab)
        print(f"Loaded {len(self.src_texts)} source samples from {src_path}.")
        print(f"Loaded {len(self.tgt_texts)} target samples from {tgt_path}.")
        print(f"Source vocab size: {config.src_vocab_size}, Target vocab size: {config.tgt_vocab_size}")
        assert len(self.src_texts) == len(self.tgt_texts), "Mismatched source and target lengths."
        
    def load_data(self, data_path: str) -> list[str]:
        with open(data_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        data = [sentence.strip().split() for sentence in sentences]
        return data
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_ids = [self.src_vocab.get(token, SRC_UNK_ID) for token in src_text]
        tgt_ids = [self.tgt_vocab.get(token, TGT_UNK_ID) for token in tgt_text]
        
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
        
    def decode_src(self, ids: torch.LongTensor, keep_token=False, keep_GO=True):
        match ids.ndim:
            case 1:
                text = " ".join(self.src_id2token[id.item()] for id in ids 
                                if id != SRC_PAD_ID and id != -100)
                if not keep_GO:
                    text = text.replace(SRC_GO, '')
                end = text.find(SRC_EOS)
                text = text[:end]
                if keep_token:
                    return text
                else:
                    return text.replace(TOKEN_IDENTIFIER, '')
            case 2:
                return [self.decode_src(id, keep_token=keep_token) for id in ids]
            case _:
                raise ValueError("Input tensor must have 1 or 2 dimensions.")
    
    def decode_tgt(self, ids: torch.LongTensor, keep_token=False):
        match ids.ndim:
            case 1:
                text = " ".join(self.tgt_id2token[id.item()] for id in ids 
                                if id != TGT_PAD_ID and id != -100)
                end = text.find(TGT_EOS)
                text = text[:end]
                if keep_token:
                    return text
                return text.replace(TOKEN_IDENTIFIER, '')
            case 2:
                return [self.decode_tgt(id, keep_token=keep_token) for id in ids]
            case _:
                raise ValueError("Input tensor must have 1 or 2 dimensions.")

# if __name__ == "__main__":
#     dataset = TranslationDataset(path="data/translation", part="train")
#     print("Dataset size:", len(dataset))
#     for i in range(3):
#         sample = dataset[i]
#         print(f"Sample {i}:")
#         print("Original Source Text:", " ".join(dataset.src_texts[i]))
#         print("Original Target Text:", " ".join(dataset.tgt_texts[i]))
#         print("Input IDs:", sample["src_ids"])
#         print("Labels:", sample["tgt_ids"])
    
#     collator = TranslationDataCollator()
#     batch = [dataset[i] for i in range(3)]
#     collated_batch = collator(batch)
#     print("Collated batch:")
#     print("Input IDs:", collated_batch["src_ids"])
#     print("Attention Mask:", collated_batch["src_mask"])
#     print("Labels:", collated_batch["tgt_ids"])
#     print("Target Attention Mask:", collated_batch["tgt_mask"])