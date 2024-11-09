from .sentiment_metrics import SentimentMetircs
from .ner_metrics import NERMetrics
from .translation_metrics import TranslationMetrics
from configs import PipelineConfig

def get_metrics(config: PipelineConfig):
    if config.task == 'sentiment':
        return SentimentMetircs(config)
    elif config.task == 'ner':
        return NERMetrics(config)
    elif config.task == 'translation':
        return TranslationMetrics(config)
    else:
        return None