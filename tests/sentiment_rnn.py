from models import get_model
from datasets import get_datasets, get_collators
from metrics import get_metrics
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from configs import PipelineConfig, BaseConfig
from configs.model_config import CNNConfig, RNNConfig, TransformerConfig
from configs.task_config import SentimentTaskConfig, NERTaskConfig, TranslationTaskConfig
import logging

def test_sentiment_rnn():
    base_config = BaseConfig(
        mode='train',
        task='sentiment',
        model='rnn',
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
        # 轻量单向配置，适合快速实验
        RNNConfig(emb_dim=64, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=False, rnn_type='lstm'),
        RNNConfig(emb_dim=64, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=False, rnn_type='gru'),
        
        # 轻量双向配置
        RNNConfig(emb_dim=64, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=True, rnn_type='lstm'),
        RNNConfig(emb_dim=64, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=True, rnn_type='gru'),
        
        # 增加嵌入和隐藏层，单向
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=1, dropout=0.2, bidirectional=False, rnn_type='lstm'),
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=1, dropout=0.2, bidirectional=False, rnn_type='gru'),
        
        # 增加嵌入和隐藏层，双向
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=1, dropout=0.2, bidirectional=True, rnn_type='lstm'),
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=1, dropout=0.2, bidirectional=True, rnn_type='gru'),
        
        # 增强配置，增加层数和隐藏层大小
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=False, rnn_type='lstm'),
        RNNConfig(emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=False, rnn_type='gru'),
        
        # 增强配置，双向，较大隐藏层
        RNNConfig(emb_dim=256, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=True, rnn_type='lstm'),
        RNNConfig(emb_dim=256, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=True, rnn_type='gru')
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