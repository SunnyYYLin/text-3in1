from configs import TextCNNConfig, TextRNNConfig
from .text_base_model import TextBaseModel
from .sentiment_cnn import SentimentCNN
from .sentiment_rnn import SentimentRNN
from .rnn_crf import RNN_CRF

def get_model(config):
    if config.task == 'sentiment':
        if isinstance(config, TextCNNConfig):
            return SentimentCNN(config)
        elif isinstance(config, TextRNNConfig):
            return SentimentRNN(config)
        else:
            raise ValueError(f'Unsupported model type: {config.model_type}')
    elif config.task == 'ner':
        if isinstance(config, TextRNNConfig):
            return RNN_CRF(config)
        else:
            raise ValueError(f'Unsupported model type: {config.model_type}')
    else:
        raise ValueError(f'Unsupported model type: {config.__class__.__name__}')