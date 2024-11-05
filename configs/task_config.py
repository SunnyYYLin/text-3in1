import os
import json
from dataclasses import dataclass

class SentimentTaskConfig:
    num_classes: int = 2
    def __post_init__(self):
        if hasattr(self, 'data_path') and self.vocab_size == -1:
            with open(os.path.join(self.data_path, 'vocab.json'), 
                      'r', encoding='utf-8') as f:
                self.vocab_size = len(json.load(f))
        if hasattr(self, 'only_last'):
            self.only_last = True
    
class NERTaskConfig:
    num_tags: int = -1
    def __post_init__(self):
        if hasattr(self, 'data_path'):
            if self.vocab_size == -1:
                with open(os.path.join(self.data_path, 'chr_vocab.json'), 
                          'r', encoding='utf-8') as f:
                    self.vocab_size = len(json.load(f))
            if self.num_tags == -1:
                with open(os.path.join(self.data_path, 'tag_vocab.json'), 
                          'r', encoding='utf-8') as f:
                    self.num_tags = len(json.load(f))
        if hasattr(self, 'only_last'):
            self.only_last = False