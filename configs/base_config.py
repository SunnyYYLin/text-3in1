from dataclasses import dataclass
import torch
from pathlib import Path

@dataclass
class BaseConfig:
    mode: str = 'train'
    task: str = 'sentiment'
    model: str = 'cnn'
    
    # data paths
    data_dir: Path = Path('data')
    save_dir: Path = Path('checkpoints')
    log_dir: Path = Path('logs')
    model_dir: Path|None = None
    
    # training parameters
    num_epoch: int = 16
    batch_size: int = 64
    loss_interval: int = 10
    acc_interval: int = 100
    lr: float = 1e-3
    fp16: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose: bool = True
    seed: int = 42
    save_best: bool = True
    early_stopping: int = 5
    grad_clip: float = 5.0
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)
        self.log_dir = Path(self.log_dir)
        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir)