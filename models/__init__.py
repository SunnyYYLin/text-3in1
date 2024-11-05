import os
from safetensors.torch import load_file
from .sentiment_model import SentimentModel
from .ner_crf import NER_CRF

def get_model(config):
    match config.task:
        case 'sentiment':
            model = SentimentModel(config)
        case 'ner':
            model = NER_CRF(config)
        case _:
            raise NotImplementedError(f"Unsupported task: {config.task}")
    
    if config.mode == 'test':
        checkpoint_list = os.listdir(config.save_dir)
        assert len(checkpoint_list) > 0, 'No checkpoint found'
        best_step = sorted(checkpoint_list, key=lambda x: int(x.split('-')[-1]))[-1]
        checkpoint_path = os.path.join(config.save_dir, best_step, 'model.safetensors')
        model.load_state_dict(load_file(checkpoint_path))
    
    print(model)
    
    return model