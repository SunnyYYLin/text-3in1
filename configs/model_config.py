import os
from dataclasses import dataclass, field
from typing import TypeAlias
import json

@dataclass
class TextModelConfig:
    emb_dim: int = 256
    dropout: float = 0.5
    mlp_dims: str|list[int] = field(default_factory=lambda: [])
    
    def __post_init__(self):
        # Since the `mlp_dims` could be list[int] or a str of list[int]
        try:
            self.mlp_dims: list[int] = eval(self.mlp_dims) if isinstance(self.mlp_dims, str) \
                else self.mlp_dims
        except:
            raise ValueError("Invalid mlp_dims format. Please use the format: [size1, size2, ...]")

@dataclass
class CNNConfig(TextModelConfig):
    filter_sizes: str|list[int] = field(default_factory=lambda: [3,4,5])
    num_filters: str|list[int] = field(default_factory=lambda: [2,2,2])
    dropout: float = 0.5
    
    def __post_init__(self):
        super().__post_init__()
        # Since the `filter_sizes` and `num_filters` could be list[int] or a str of list[int]
        try:
            self.filter_sizes: list[int] = eval(self.filter_sizes) if isinstance(self.filter_sizes, str) \
                else self.filter_sizes
        except:
            raise ValueError("Invalid filter_sizes format. Please use the format: [size1, size2, ...]")
        
        try:
            self.num_filters: list[int] = eval(self.num_filters) if isinstance(self.num_filters, str) \
                else self.num_filters
        except:
            raise ValueError("Invalid num_filters format. Please use the format: [num1, num2, ...]")
        
        assert len(self.filter_sizes) == len(self.num_filters), \
            "The number of filter sizes should be equal to the number of num_filters."
    
    def abbr(self) -> str:
        cnn_config = f"CNN_filters{self.filter_sizes}_num{self.num_filters}_emb{self.emb_dim}_dropout{self.dropout}"
        return cnn_config
    
    @staticmethod
    def from_abbr(abbr: str):
        config_list = abbr.split('_')
        filter_sizes = eval(config_list[1].removeprefix('filters'))
        num_filters = eval(config_list[2].removeprefix('num'))
        emb_dim = int(config_list[3].removeprefix('emb'))
        dropout = float(config_list[4].removeprefix('dropout'))
        config = CNNConfig(filter_sizes=filter_sizes, num_filters=num_filters, emb_dim=emb_dim, dropout=dropout)
        return config
        
@dataclass
class RNNConfig(TextModelConfig):
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    rnn_type: str = 'lstm'  # RNN 类型，可以是 'LSTM' 或 'GRU'
    only_one: bool = True  # 是否只使用一个输出
    dropout: float = 0.2
    
    def __post_init__(self):
        super().__post_init__()
        self.rnn_type = self.rnn_type.lower()
    
    def abbr(self) -> str:
        """Return an abbreviation string of the config for saving files."""
        rnn_config = \
            f"{self.rnn_type.upper()}_layers{self.num_layers}_hidden{self.hidden_size}_emb{self.emb_dim}_dropout{self.dropout}"
        if self.bidirectional:
            rnn_config += "_bi"
        return rnn_config
    
    @staticmethod
    def from_abbr(abbr: str):
        """Return a TextRNNConfig object from an abbreviation string."""
        config_list = abbr.split('_')
        rnn_type = config_list[0]
        num_layers = int(config_list[1].removeprefix('layers'))
        hidden_size = int(config_list[2].removeprefix('hidden'))
        emb_dim = int(config_list[3].removeprefix('emb'))
        dropout = float(config_list[4].removeprefix('dropout'))
        bidirectional = True if 'bi' in config_list else False
        config = RNNConfig(rnn_type=rnn_type, num_layers=num_layers,
                               hidden_size=hidden_size, emb_dim=emb_dim,
                               bidirectional=bidirectional, dropout=dropout)
        return config
    
@dataclass
class TransformerConfig(TextModelConfig):
    ffn_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    only_one: bool = True  # 是否只使用一个输出
    dropout: float = 0.1
    
    def __post_init__(self):
        super().__post_init__()
        assert self.emb_dim % self.num_heads == 0, \
            "embedding dim should be divisible by num_heads."

    def abbr(self) -> str:
        """Return an abbreviation string of the config for saving files."""
        transformer_config = f"Transformer_layers{self.num_layers}_ffnsize{self.ffn_size}_heads{self.num_heads}_emb{self.emb_dim}_dropout{self.dropout}"
        return transformer_config
    
    @staticmethod
    def from_abbr(abbr: str):
        """Return a TextTransformerConfig object from an abbreviation string."""
        config_list = abbr.split('_')
        num_layers = int(config_list[1].removeprefix('layers'))
        ffn_size = int(config_list[2].removeprefix('ffnsize'))
        num_heads = int(config_list[3].removeprefix('heads'))
        emb_dim = int(config_list[4].removeprefix('emb'))
        dropout = float(config_list[5].removeprefix('dropout'))
        config = TransformerConfig(num_layers=num_layers, ffn_size=ffn_size,
                                       num_heads=num_heads, emb_dim=emb_dim, dropout=dropout)
        return config

ModelConfig: TypeAlias = CNNConfig|RNNConfig|TransformerConfig
MODEL_CONFIG_CLASSES = {
    'cnn': CNNConfig,
    'rnn': RNNConfig,
    'transformer': TransformerConfig
}
CLASSES2STR = {
    CNNConfig: 'cnn',
    RNNConfig: 'rnn',
    TransformerConfig: 'transformer'
}

def get_model_config_from_abbr(abbr: str) -> ModelConfig:
    """Return a model config object from an abbreviation string."""
    model_type = abbr.split('_')[0]
    if model_type == 'CNN':
        return CNNConfig.from_abbr(abbr)
    elif model_type in ['LSTM', 'GRU']:
        return RNNConfig.from_abbr(abbr)
    elif model_type == 'Transformer':
        return TransformerConfig.from_abbr(abbr)
    else:
        raise ValueError(f"Invalid model type: {model_type}")