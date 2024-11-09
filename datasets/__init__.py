import os
import json
from .ner import NERDataset, NERDataCollator
from .sentiment import SentimentDataset, SentimentDataCollator
from .translation import TranslationDataset, TranslationDataCollator
from configs import BaseConfig

def get_datasets(config: BaseConfig):
    if config.task == 'ner':
        return NERDataset(config.data_path, 'train'), \
            NERDataset(config.data_path, 'val'), \
            NERDataset(config.data_path, 'test')
    elif config.task == 'sentiment':
        return SentimentDataset(config.data_path, 'train'), \
            SentimentDataset(config.data_path, 'val'), \
            SentimentDataset(config.data_path, 'test')
    elif config.task == 'translation':
        return TranslationDataset(config.data_path, 'train'), \
            TranslationDataset(config.data_path, 'val'), \
            TranslationDataset(config.data_path, 'test')
    else:
        raise NotImplementedError(f"Unsupported task: {config.task}")
    
def get_collators(config: BaseConfig):
    match config.task:
        case 'ner':
            return NERDataCollator()
        case 'sentiment':
            return SentimentDataCollator()
        case 'translation':
            return TranslationDataCollator()
        case _:
            raise NotImplementedError(f"Unsupported task: {config.task}")