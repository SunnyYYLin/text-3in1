from .text_cnn import TextCNN
from .text_rnn import TextRNN
from .encoder import Encoder
from .encoder_decoder import EncoderDecoder

def get_backbone(config):
    match config.model.lower():
        case 'rnn':
            model_cls = TextRNN
        case 'cnn':
            model_cls = TextCNN
        case 'transformer':
            model_cls = EncoderDecoder if config.task == 'translation' else Encoder
        case _:
            raise NotImplementedError(f"Unsupported model type: {config.model}")
    return model_cls