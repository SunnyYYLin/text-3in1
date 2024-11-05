import os
import json
from dataclasses import dataclass, field

@dataclass
class TextModelConfig:
    emb_dim: int = 256
    vocab_size: int = -1

@dataclass
class CNNConfig(TextModelConfig):
    filter_sizes: list = field(default_factory=lambda: [4, 5, 6])
    num_filters: int = 1
    output_size: int = 2
    
    def __post_init__(self):
        super().__post_init__()
        self.log_dir = os.path.join(self.log_dir, self.abbr())
        self.save_dir = os.path.join(self.save_dir, self.abbr())
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def abbr(self) -> str:
        filter_config = '-'.join([str(size) for size in self.filter_sizes])
        cnn_config = f"CNN_filters{filter_config}_repeat{self.num_filters}_emb{self.emb_dim}"
        return cnn_config
    
    @staticmethod
    def from_abbr(abbr: str):
        config_list = abbr.split('_')
        filter_config = config_list[0].replace('CNN_filters', '')
        filter_sizes = [int(size) for size in filter_config.split('-')]
        num_filters = int(config_list[1].replace('repeat', ''))
        emb_dim = int(config_list[2].replace('emb', ''))
        config = CNNConfig(filter_sizes=filter_sizes, num_filters=num_filters, emb_dim=emb_dim)
        return config
        
@dataclass
class RNNConfig(TextModelConfig):
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    rnn_type: str = 'LSTM'  # RNN 类型，可以是 'LSTM' 或 'GRU'
    only_last: bool = True  # 是否只使用最后一个时间步的输出
    
    def __post_init__(self):
        super().__post_init__()
        self.log_dir = os.path.join(self.log_dir, self.abbr())
        self.save_dir = os.path.join(self.save_dir, self.abbr())
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def abbr(self) -> str:
        """Return an abbreviation string of the config for saving files."""
        rnn_config = \
            f"{self.rnn_type.upper()}_layers{self.num_layers}_hidden{self.hidden_size}_emb{self.emb_dim}"
        if self.bidirectional:
            rnn_config += "_bi"
        return rnn_config
    
    @staticmethod
    def from_abbr(abbr: str):
        """Return a TextRNNConfig object from an abbreviation string."""
        config_list = abbr.split('_')
        rnn_type = config_list[0]
        num_layers = int(config_list[1].replace('layers', ''))
        hidden_size = int(config_list[2].replace('hidden', ''))
        emb_dim = int(config_list[3].replace('emb', ''))
        bidirectional = True if 'bi' in config_list else False
        config = RNNConfig(rnn_type=rnn_type, num_layers=num_layers,
                               hidden_size=hidden_size, emb_dim=emb_dim,
                               bidirectional=bidirectional)
        return config
    
@dataclass
class TransformerConfig(TextModelConfig):
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048

    def abbr(self) -> str:
        transformer_config = (
            f"layers{self.num_layers}_dmodel{self.d_model}_"
            f"heads{self.num_heads}_dff{self.dff}_drop{self.dropout_rate}"
        )
        return transformer_config

    @staticmethod
    def from_abbr(abbr: str):
        parts = abbr.split('_')
        num_layers = int(parts[0].replace('layers', ''))
        d_model = int(parts[1].replace('dmodel', ''))
        num_heads = int(parts[2].replace('heads', ''))
        dff = int(parts[3].replace('dff', ''))
        return TransformerConfig(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff
        )
