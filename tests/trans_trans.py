from models import get_model
from datasets import get_datasets, get_collators
from metrics import get_metrics
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from configs import PipelineConfig, BaseConfig
from configs.model_config import CNNConfig, RNNConfig, TransformerConfig
from configs.task_config import SentimentTaskConfig, NERTaskConfig, TranslationTaskConfig
import logging

def test_translation_transformer():
    base_config = BaseConfig(
        mode='train',
        task='translation',
        model='transformer',
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
    translation_configs = TranslationTaskConfig()
    transformer_configs_list = [
        # 基础配置，低计算成本
        TransformerConfig(emb_dim=256, ffn_size=1024, num_heads=4, num_layers=2, dropout=0.1),
        TransformerConfig(emb_dim=256, ffn_size=1024, num_heads=4, num_layers=4, dropout=0.2),
        
        # 增加模型深度，提高表达能力
        TransformerConfig(emb_dim=256, ffn_size=2048, num_heads=4, num_layers=6, dropout=0.1),
        TransformerConfig(emb_dim=256, ffn_size=2048, num_heads=4, num_layers=6, dropout=0.3),
        
        # 使用较大嵌入维度
        TransformerConfig(emb_dim=512, ffn_size=1024, num_heads=8, num_layers=4, dropout=0.1),
        TransformerConfig(emb_dim=512, ffn_size=2048, num_heads=8, num_layers=4, dropout=0.3),
        
        # 更高的模型深度和参数
        TransformerConfig(emb_dim=512, ffn_size=2048, num_heads=8, num_layers=6, dropout=0.1),
        TransformerConfig(emb_dim=512, ffn_size=2048, num_heads=8, num_layers=6, dropout=0.3),
        
        # 极简配置，用于基线测试
        TransformerConfig(emb_dim=256, ffn_size=1024, num_heads=4, num_layers=2, dropout=0.3),
        
        # 尽量大参数配置，用于高资源环境测试
        TransformerConfig(emb_dim=512, ffn_size=2048, num_heads=8, num_layers=6, dropout=0.1),
    ]

    # 将每组配置组合到 PipelineConfig 中
    configs: list[PipelineConfig] = []
    for transformer_config in transformer_configs_list:
        configs.append(PipelineConfig(base_config, transformer_config, translation_configs))
    
    for config in configs:
        try:
            train_dataset, val_dataset, test_dataset = get_datasets(config)
            collator = get_collators(config)
            train_args = config.train_args()
            metrics = get_metrics(config)
            model = get_model(config)
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=6,
                early_stopping_threshold=0.001 # 指标变化阈值
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
            print(e)
            logging.error(f"Failed to train model with config: {config}\n")
            with open('failed_configs.txt', 'a') as f:
                f.write(f"Error:{e}\n{config}\n\n")