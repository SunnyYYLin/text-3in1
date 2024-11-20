import os
from pathlib import Path
from safetensors.torch import load_file
from configs.model_config import get_model_config_from_abbr, CLASSES2STR
from configs.pipeline_config import PipelineConfig
from .sentiment_model import SentimentModel
from .ner_crf import NER_CRF
from .translater import Translater

def get_model(config: PipelineConfig):
    if config.model_dir is not None:
        model_config = get_model_config_from_abbr(config.model_dir.name)
        config.model_config = config.task_config.modify_model(model_config)
        for cls in CLASSES2STR:
            if isinstance(config.model_config, cls):
                config.base_config.model = CLASSES2STR[cls]
                break
    
    match config.task:
        case 'sentiment':
            model = SentimentModel(config)
        case 'ner':
            model = NER_CRF(config)
        case 'translation':
            model = Translater(config)
        case _:
            raise NotImplementedError(f"Unsupported task: {config.task}")
    
    if config.mode == 'test' or config.mode == 'example':
        checkpoint_list = os.listdir(config.model_dir)
        assert len(checkpoint_list) > 0, f'No checkpoint found from {config.model_dir}'
        best_step = sorted(checkpoint_list, key=lambda x: int(x.split('-')[-1]))[-1]
        checkpoint_path = os.path.join(config.model_dir, best_step, 'model.safetensors')
        print(f"Load model from {checkpoint_path}")
        model.load_state_dict(load_file(checkpoint_path))
        return model
    
    return model