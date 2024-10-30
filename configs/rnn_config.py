import os
from dataclasses import dataclass, field
from .base_config import BaseConfig

@dataclass
class TextRNNConfig(BaseConfig):
    hidden_size: int = 128  # RNN 隐藏层大小
    num_layers: int = 2     # RNN 层数
    bidirectional: bool = True  # 是否使用双向 RNN
    rnn_type: str = 'LSTM'  # RNN 类型，可以是 'LSTM' 或 'GRU'
    
    def __post_init__(self):
        super().__post_init__()
        self.log_dir = os.path.join(self.log_dir, self.abbr())
        self.save_dir = os.path.join(self.save_dir, self.abbr())
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def abbr(self) -> str:
        rnn_config = f"{self.rnn_type}_hidden{self.hidden_size}_layers{self.num_layers}"
        if self.bidirectional:
            rnn_config += "_bi"
        base_config = super().abbr()
        return f"TextRNN_{rnn_config}_{base_config}"