from dataclasses import dataclass
import torch

@dataclass
class BaseConfig:
    mode: str = 'train'
    task: str = 'sentiment'
    model: str = 'cnn'
    
    # data paths
    data_dir: str = 'data'
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # training parameters
    num_epoch: int = 16
    batch_size: int = 64
    loss_interval: int = 10
    acc_interval: int = 100
    lr: float = 1e-3
    fp16: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose: bool = True