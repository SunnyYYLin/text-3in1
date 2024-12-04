import os
import json
from dataclasses import dataclass, field
from typing import TypeAlias
from .model_config import ModelConfig

@dataclass
class SentimentTaskConfig:
    label_names: list[str] = field(default_factory=lambda: ['labels'])
    max_len: int = 2048
    metric_for_best_model: str = 'accuracy'
    greater_is_better: bool = True
    
    def modify_model(self, model_config: ModelConfig) -> ModelConfig:
        return model_config

@dataclass
class NERTaskConfig:
    label_names: list[str] = field(default_factory=lambda: ['labels'])
    max_len: int = 256
    metric_for_best_model: str = 'f1'
    greater_is_better: bool = True
            
    def modify_model(self, model_config: ModelConfig) -> ModelConfig:
        if hasattr(model_config, 'only_one'):
            model_config.only_one = False
        return model_config

@dataclass       
class TranslationTaskConfig:
    src_lang: str = 'en'
    tgt_lang: str = 'zh'
    label_names: list[str] = field(default_factory=lambda: ['tgt_ids'])
    max_len: int = 512
    metric_for_best_model: str = 'bleu_score'
    greater_is_better: bool = True
    
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