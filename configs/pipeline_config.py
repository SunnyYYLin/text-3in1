import os
from dataclasses import dataclass
from pathlib import Path
from transformers.trainer import TrainingArguments
from .base_config import BaseConfig
from .model_config import ModelConfig
from .task_config import TaskConfig

@dataclass
class PipelineConfig:
    base_config: BaseConfig
    model_config: ModelConfig
    task_config: TaskConfig

    def __post_init__(self):
        # make sure the data, log, and save directories exist
        self.data_path = self.base_config.data_dir/self.base_config.task
        self.log_path = self.base_config.log_dir/self.base_config.task/self.model_config.abbr()
        self.save_path = self.base_config.save_dir/self.base_config.task/self.model_config.abbr()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # set the vocab_size, num_tags, num_classes, only_last or something about model
        self.task_config.init(self.data_path)
        self.model_config = self.task_config.modify_model(self.model_config)

    def __getattr__(self, attr):
        # Check each of the configs for the requested attribute
        if hasattr(self.base_config, attr):
            return getattr(self.base_config, attr)
        elif hasattr(self.model_config, attr):
            return getattr(self.model_config, attr)
        elif hasattr(self.task_config, attr):
            return getattr(self.task_config, attr)
        else:
            raise AttributeError(f"{attr} not found in any of the provided configs")
        
    def __str__(self) -> str:
        config_dict = {}
        config_dict.update(vars(self.base_config))
        config_dict.update(vars(self.model_config))
        config_dict.update(vars(self.task_config))
        config_dict.pop('data_dir')
        config_dict.pop('save_dir')
        config_dict.pop('log_dir')
        config_dict['data_path'] = self.data_path
        config_dict['log_path'] = self.log_path
        config_dict['save_path'] = self.save_path
        return 'PipelineConfig('+', '.join([f"{k}={v}" for k, v in config_dict.items()])+')'
    
    def train_args(self) -> TrainingArguments:
        """将部分配置转换为 TrainingArguments"""
        return TrainingArguments(
            output_dir=self.save_path,
            num_train_epochs=self.num_epoch,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="steps" if self.mode == "train" else "no",
            eval_steps=self.acc_interval,
            learning_rate=self.lr,
            logging_dir=self.log_path,
            logging_steps=self.loss_interval,
            save_strategy="steps" if self.mode == "train" else "no", 
            save_total_limit=2,  # 只保留最近的两个检查点
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            report_to="tensorboard",  # 使用 TensorBoard 记录日志
            fp16=self.fp16,  # 如果有 GPU，则启用混合精度训练
            disable_tqdm=not self.verbose,  # 是否禁用 TQDM 进度条
            no_cuda=(self.device!='cuda'),  # 是否禁用 CUDA
        )