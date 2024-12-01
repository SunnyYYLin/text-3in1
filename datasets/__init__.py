import os
import json
from .ner import NERDataset, NERDataCollator
from .sentiment import SentimentDataset, SentimentDataCollator
from .translation import TranslationDataset, TranslationDataCollator
from configs import BaseConfig

def get_datasets(config: BaseConfig):
    if config.task == 'ner':
        return NERDataset(config, 'train'), \
            NERDataset(config, 'val'), \
            NERDataset(config, 'test')
    elif config.task == 'sentiment':
        return SentimentDataset(config, 'train'), \
            SentimentDataset(config, 'val'), \
            SentimentDataset(config, 'test')
    elif config.task == 'translation':
        return TranslationDataset(config, 'train'), \
            TranslationDataset(config, 'val'), \
            TranslationDataset(config, 'test')
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