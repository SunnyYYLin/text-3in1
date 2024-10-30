import argparse
from inspect import signature
from typing import Type
from .cnn_config import TextCNNConfig
from .rnn_config import TextRNNConfig

class ConfigParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        
        # base args
        self.add_argument('--task', default='sentiment', choices=['sentiment', 'ner'])
        self.add_argument('--model', default='cnn', choices=['cnn', 'rnn', 'transformer'])
        self.add_argument('--data_path', default='data/sentiment')
        self.add_argument('--save_dir', default=None)
        self.add_argument('--log_dir', default='logs')
        self.add_argument('--num_epoch', default=8, type=int)
        self.add_argument('--batch_size', default=64, type=int)
        self.add_argument('--loss_interval', default=1, type=int)
        self.add_argument('--acc_interval', default=100, type=int)
        self.add_argument('--lr', default=1e-3, type=float)
        self.add_argument('--dropout', default=0.5, type=float)
        self.add_argument('--embedding_dim', default=256, type=int)
        self.add_argument('--verbose', default=False, type=bool)
        self.add_argument('--fp16', action='store_true')
        self.add_argument('--num_classes', default=2, type=int)

        # CNN
        self.add_argument('--filter_sizes', default=[3, 4, 5], nargs='+', type=int)
        self.add_argument('--num_filters', default=1, type=int)

        # RNN
        self.add_argument('--hidden_size', type=int, default=256, help='Hidden size for RNN')
        self.add_argument('--num_layers', type=int, default=2, help='Number of layers for RNN')
        self.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN')
        self.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Type of RNN: LSTM or GRU')
    
    def parse_config(self):
        model = self.parse_args().model
        if model == 'cnn':
            return TextCNNConfig(**self.filter_kwargs(TextCNNConfig))
        elif model == 'rnn':
            return TextRNNConfig(**self.filter_kwargs(TextRNNConfig))
        else:
            raise NotImplementedError(f'{model} is not implemented')
    
    def filter_kwargs(self, cls: Type) -> dict[str, ]:
        args = self.parse_args()
        kwdefaults = {arg: self.get_default(arg) for arg in vars(args)}
        kwargs = vars(args)
        cls_params = signature(cls).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in cls_params and v != kwdefaults[k]}
        return kwargs