import os
from safetensors.torch import load_file
from .sentiment_model import SentimentModel
from .ner_crf import NER_CRF
from .translater import Translater

def get_model(config):
    match config.task:
        case 'sentiment':
            model = SentimentModel(config)
        case 'ner':
            model = NER_CRF(config)
        case 'translation':
            model = Translater(config)
        case _:
            raise NotImplementedError(f"Unsupported task: {config.task}")
    
    if config.mode == 'test' or 'example':
        checkpoint_list = os.listdir(config.save_path)
        assert len(checkpoint_list) > 0, f'No checkpoint found from {config.save_path}'
        best_step = sorted(checkpoint_list, key=lambda x: int(x.split('-')[-1]))[-1]
        checkpoint_path = os.path.join(config.save_path, best_step, 'model.safetensors')
        model.load_state_dict(load_file(checkpoint_path))
        print(f"Load model from {checkpoint_path}")
    
    return model