from typing import TypeAlias
from dataclasses import dataclass
from .base_config import BaseConfig
from .model_config import CNNConfig, RNNConfig, TransformerConfig
from .task_config import SentimentTaskConfig, NERTaskConfig

@dataclass
class SentiCNNConfig(BaseConfig, CNNConfig, SentimentTaskConfig):
    def __post_init__(self):
        BaseConfig.__post_init__(self)
        CNNConfig.__post_init__(self)
        SentimentTaskConfig.__post_init__(self)

@dataclass
class SentiRNNConfig(BaseConfig, RNNConfig, SentimentTaskConfig):
    def __post_init__(self):
        BaseConfig.__post_init__(self)
        RNNConfig.__post_init__(self)
        SentimentTaskConfig.__post_init__(self)

@dataclass
class SentiTransformerConfig(BaseConfig, TransformerConfig, SentimentTaskConfig):
    def __post_init__(self):
        BaseConfig.__post_init__(self)
        TransformerConfig.__post_init__(self)
        SentimentTaskConfig.__post_init__(self)

@dataclass
class NER_RNNConfig(BaseConfig, RNNConfig, NERTaskConfig):
    def __post_init__(self):
        BaseConfig.__post_init__(self)
        RNNConfig.__post_init__(self)
        NERTaskConfig.__post_init__(self)

@dataclass
class NERTransformerConfig(BaseConfig, TransformerConfig, NERTaskConfig):
    def __post_init__(self):
        BaseConfig.__post_init__(self)
        TransformerConfig.__post_init__(self)
        NERConfig.__post_init__(self)
        
SentimentConfig: TypeAlias = SentiCNNConfig|SentiRNNConfig|SentiTransformerConfig
NERConfig: TypeAlias = NER_RNNConfig|NERTransformerConfig
Config: TypeAlias = SentimentConfig|NERConfig