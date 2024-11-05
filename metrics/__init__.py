from .sentiment_metrics import SentimentMetircs
from .ner_metrics import NERMetrics
from configs import BaseConfig

def get_metrics(config: BaseConfig):
    if config.task == 'sentiment':
        return SentimentMetircs(config)
    elif config.task == 'ner':
        return NERMetrics(config)
    else:
        raise NotImplementedError(f"Task {config.task} not implemented!")