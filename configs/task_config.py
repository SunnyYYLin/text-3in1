import os
import json
from dataclasses import dataclass, field
from typing import TypeAlias
from .model_config import ModelConfig

@dataclass
class SentimentTaskConfig:
    num_classes: int = 2
    vocab_size: int = -1
    label_names: list[str] = field(default_factory=lambda: ['labels'])
    
    def init(self, data_path: str):
        with open(os.path.join(data_path, 'vocab.json'), 
                  'r', encoding='utf-8') as f:
            self.vocab_size = len(json.load(f))
    
    def modify_model(self, model_config: ModelConfig) -> ModelConfig:
        return model_config

@dataclass
class NERTaskConfig:
    num_tags: int = -1
    vocab_size: int = -1
    label_names: list[str] = field(default_factory=lambda: ['labels'])
    
    def init(self, data_path: str):
        with open(os.path.join(data_path, 'chr_vocab.json'), 
                  'r', encoding='utf-8') as f:
            self.vocab_size = len(json.load(f))
        with open(os.path.join(data_path, 'tag_vocab.json'), 
                  'r', encoding='utf-8') as f:
            self.num_tags = len(json.load(f))
            
    def modify_model(self, model_config: ModelConfig) -> ModelConfig:
        if hasattr(model_config, 'only_one'):
            model_config.only_one = False
        return model_config

@dataclass       
class TranslationTaskConfig:
    src_vocab_size: int = -1
    tgt_vocab_size: int = -1
    label_names: list[str] = field(default_factory=lambda: ['tgt_ids'])
    
    def init(self, data_path: str):
        with open(os.path.join(data_path, 'train.en.json'), 
                  'r', encoding='utf-8') as f:
            self.src_vocab_size = len(json.load(f))
        with open(os.path.join(data_path, 'train.zh.json'), 
                  'r', encoding='utf-8') as f:
            self.tgt_vocab_size = len(json.load(f))
    
    def modify_model(self, model_config: ModelConfig) -> ModelConfig:
        if hasattr(model_config, 'only_one'):
            model_config.only_one = False
        return model_config
            
TaskConfig: TypeAlias = SentimentTaskConfig|NERTaskConfig|TranslationTaskConfig
TASK_CONFIG_CLASSES = {
    'sentiment': SentimentTaskConfig,
    'ner': NERTaskConfig,
    'translation': TranslationTaskConfig
}