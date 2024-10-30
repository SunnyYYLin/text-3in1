from dataclasses import dataclass
import os
import torch
from transformers.trainer import TrainingArguments

@dataclass
class BaseConfig:
    task: str = 'sentiment'
    
    # data paths
    data_path: str = ''
    save_dir: str = ''
    log_dir: str = 'logs'
    
    # training parameters
    num_epoch: int = 8
    batch_size: int = 64
    loss_interval: int = 10
    acc_interval: int = 100
    lr: float = 1e-3
    dropout: float = 0.5
    fp16: bool = False
    
    # model parameters
    emb_dim: int = 256
    num_classes: int = 2
    
    # others
    vocab_size = -1
    verbose: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        self.save_dir: str = os.path.join('checkpoints', self.task)
        self.data_path: str = os.path.join('data', self.task)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def abbr(self) -> str:
        return f"{self.emb_dim}_{self.dropout*100:.0f}"
    
    def train_args(self) -> TrainingArguments:
        """将部分配置转换为 TrainingArguments"""
        return TrainingArguments(
            output_dir=self.save_dir,
            num_train_epochs=self.num_epoch,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="steps",
            eval_steps=self.acc_interval,
            learning_rate=self.lr,
            logging_dir=self.log_dir,
            logging_steps=self.loss_interval,
            save_strategy="steps",  # 每个 epoch 保存一次检查点
            save_total_limit=2,  # 只保留最近的两个检查点
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            report_to="tensorboard",  # 使用 TensorBoard 记录日志
            fp16=self.fp16,  # 如果有 GPU，则启用混合精度训练
            disable_tqdm=not self.verbose,  # 是否禁用 TQDM 进度条
            no_cuda=not torch.cuda.is_available(),  # 是否禁用 CUDA
        )