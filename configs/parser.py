import argparse
from inspect import signature
from typing import Type
from .base_config import BaseConfig
from .model_config import MODEL_CONFIG_CLASSES
from .task_config import TASK_CONFIG_CLASSES
from .pipeline_config import PipelineConfig

class ConfigParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        
        # base args
        self.add_argument('--mode', default='train', choices=['train', 'test'])
        self.add_argument('--task', default='sentiment', choices=['sentiment', 'ner', 'translation'])
        self.add_argument('--model', default='cnn')
        self.add_argument('--data_dir', default='data')
        self.add_argument('--save_dir', default='checkpoints')
        self.add_argument('--log_dir', default='logs')
        # train
        self.add_argument('--num_epoch', default=8, type=int)
        self.add_argument('--batch_size', default=64, type=int)
        self.add_argument('--loss_interval', default=1, type=int)
        self.add_argument('--acc_interval', default=100, type=int)
        self.add_argument('--lr', default=1e-3, type=float)
        self.add_argument('--dropout', default=0.5, type=float)
        self.add_argument('--verbose', default=False, type=bool)
        self.add_argument('--fp16', action='store_true')
        
        
        # TextModel
        self.add_argument('--emb_dim', default=256, type=int)
        
        # Sentiment
        self.add_argument('--num_classes', default=2, type=int)

        # CNN
        self.add_argument('--filter_sizes', default="[3,4,5]", type=str,
                          help='Filter sizes for CNN. Format: "[size1, size2, ...]"')
        self.add_argument('--num_filters', default="[2,2,2]", type=str,
                          help='Number of filters for each layer. Format: "[num1, num2, ...]"')

        # RNN
        self.add_argument('--hidden_size', type=int, default=256, help='Hidden size for RNN/Transformer FFN')
        self.add_argument('--num_layers', type=int, default=2, help='Number of layers for RNN/Transformer FFN')
        self.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN')
        self.add_argument('--rnn_type', type=str, default='LSTM', help='Type of RNN: LSTM or GRU')
        
        # Transformer
        self.add_argument('--num_heads', type=int, default=8, help='Number of heads for multi-head attention')
        self.add_argument('--ffn_size', type=int, default=256, help='Hidden size for Transformer FFN')
        # num_layers is added at RNN
    
    def parse_config(self):
        args = self.parse_args()
        task_config_cls = TASK_CONFIG_CLASSES[args.task]
        model_config_cls = MODEL_CONFIG_CLASSES[args.model]
        base_config = BaseConfig(**self.filter_kwargs(BaseConfig))
        task_config = task_config_cls(**self.filter_kwargs(task_config_cls))
        model_config = model_config_cls(**self.filter_kwargs(model_config_cls))
        return PipelineConfig(base_config, model_config, task_config)
    
    def filter_kwargs(self, cls: Type) -> dict[str, ]:
        args = self.parse_args()
        kwdefaults = {arg: self.get_default(arg) for arg in vars(args)}
        kwargs = vars(args)
        cls_params = signature(cls).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in cls_params and v != kwdefaults[k]}
        return kwargs