import os
import json
from .ner import NERDataset, NERDataCollator
from .sentiment import SentimentDataset, SentimentDataCollator
from configs import BaseConfig

def get_datasets(config: BaseConfig):
    if config.task == 'ner':
        with open(os.path.join(config.data_path, 'chr_vocab.json')) as f:
            config.vocab_size = len(json.load(f))
        return NERDataset(config.data_path, 'train'), \
            NERDataset(config.data_path, 'val'), \
            NERDataset(config.data_path, 'test')
    elif config.task == 'sentiment':
        with open(os.path.join(config.data_path, 'vocab.json')) as f:
            config.vocab_size = len(json.load(f))
        return SentimentDataset(config.data_path, 'train'), \
            SentimentDataset(config.data_path, 'val'), \
            SentimentDataset(config.data_path, 'test')
    else:
        raise NotImplementedError(f"Unsupported task: {config.task}")
    
def get_collators(config: BaseConfig):
    if config.task == 'ner':
        return NERDataCollator()
    elif config.task == 'sentiment':
        return SentimentDataCollator()
    else:
        raise NotImplementedError(f"Unsupported task: {config.task}")