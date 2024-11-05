from .text_cnn import TextCNN
from .text_rnn import TextRNN

def get_backbone(model: str):
    match model.lower():
        case 'rnn':
            model_cls = TextRNN
        case 'cnn':
            model_cls = TextCNN
        case _:
            raise NotImplementedError(f"Unsupported model type: {model}")
    return model_cls