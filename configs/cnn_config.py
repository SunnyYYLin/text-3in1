import os
from dataclasses import dataclass, field
from .base_config import BaseConfig

@dataclass
class TextCNNConfig(BaseConfig):
    filter_sizes: list = field(default_factory=lambda: [4, 5, 6])
    num_filters: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        self.log_dir = os.path.join(self.log_dir, self.abbr())
        self.save_dir = os.path.join(self.save_dir, self.abbr())
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def abbr(self) -> str:
        filter_config = '-'.join([str(size) for size in self.filter_sizes]) \
            + f"_{self.num_filters}"
        base_config = super().abbr()
        return f"TextCNN_{filter_config}_{base_config}"
        