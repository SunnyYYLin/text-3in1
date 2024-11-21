from models import get_model
from datasets import get_datasets, get_collators
from metrics import get_metrics
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from configs import PipelineConfig, BaseConfig
from configs.model_config import CNNConfig, RNNConfig, TransformerConfig
from configs.task_config import SentimentTaskConfig, NERTaskConfig, TranslationTaskConfig
import logging

def test_sentiment_cnn():
    base_config = BaseConfig(
        mode='train',
        task='sentiment',
        model='cnn',
        data_dir='data',
        save_dir='checkpoints',
        log_dir='logs',
        num_epoch=32,
        batch_size=128,
        loss_interval=10,
        acc_interval=100,
        lr=1e-3,
        fp16=True,
        device='cuda',
        verbose=True
    )
    task_config = SentimentTaskConfig()
    model_configs_list = [
        # 轻量配置
        CNNConfig(emb_dim=128, dropout=0.3, filter_sizes=[2, 3, 4], num_filters=[2, 2, 2]),
        CNNConfig(emb_dim=128, dropout=0.3, filter_sizes=[3, 4, 5], num_filters=[2, 2, 2]),
        
        # 标准配置，适合大多数场景
        CNNConfig(emb_dim=256, dropout=0.3, filter_sizes=[2, 3, 4], num_filters=[4, 4, 4]),
        CNNConfig(emb_dim=256, dropout=0.3, filter_sizes=[3, 4, 5], num_filters=[4, 4, 4]),
        
        # 增强配置，增加过滤器数量和更高的 dropout
        CNNConfig(emb_dim=256, dropout=0.5, filter_sizes=[2, 3, 4], num_filters=[8, 8, 8]),
        CNNConfig(emb_dim=256, dropout=0.5, filter_sizes=[3, 4, 5], num_filters=[8, 8, 8]),
        
        # 更强特征提取配置，适合高资源场景
        CNNConfig(emb_dim=256, dropout=0.5, filter_sizes=[2, 3, 4, 5], num_filters=[4, 4, 4, 4]),
        CNNConfig(emb_dim=256, dropout=0.5, filter_sizes=[2, 3, 4, 5], num_filters=[8, 8, 8, 8]),
    ]

    # 将每组配置组合到 PipelineConfig 中
    configs: list[PipelineConfig] = []
    for model_config in model_configs_list:
        configs.append(PipelineConfig(base_config, model_config, task_config))
    
    for config in configs:
        try:
            train_dataset, val_dataset, test_dataset = get_datasets(config)
            collator = get_collators(config)
            train_args = config.train_args()
            metrics = get_metrics(config)
            model = get_model(config)
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=6,
                early_stopping_threshold=0.001
            )
            trainer = Trainer(model=model, 
                            args=train_args, 
                            train_dataset=train_dataset, 
                            eval_dataset=val_dataset, 
                            data_collator=collator,
                            compute_metrics=metrics,
                            callbacks=[early_stopping]
                            )
            trainer.label_names = config.label_names
            trainer.train()
        except Exception as e:
            logging.error(e)
            with open('failed_configs.txt', 'a') as f:
                f.write(f"Error:{e}\n{config}\n")