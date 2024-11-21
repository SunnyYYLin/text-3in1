import argparse
from inspect import signature
from typing import Type
from .base_config import BaseConfig
from .model_config import MODEL_CONFIG_CLASSES
from .task_config import TASK_CONFIG_CLASSES
from .pipeline_config import PipelineConfig

import argparse
import json

class ConfigParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        
        # 基础参数
        self.add_argument('--mode', default='train', choices=['train', 'test', 'example'], 
                          help="运行模式: train, test")
        self.add_argument('--task', default='sentiment', choices=['sentiment', 'ner', 'translation'], 
                          help="任务类型: sentiment, ner, or translation")
        self.add_argument('--model', default='cnn', choices=['cnn', 'rnn', 'transformer', 'bert'], 
                          help="模型类型: cnn, rnn, transformer, or bert")
        self.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], 
                          help="运行设备: cuda 或 cpu")
        self.add_argument('--data_dir', default='data', 
                          help="数据存储路径")
        self.add_argument('--save_dir', default='checkpoints', 
                          help="模型保存路径")
        self.add_argument('--log_dir', default='logs', 
                          help="日志保存路径")
        self.add_argument('--seed', type=int, default=42, 
                          help="随机种子，用于结果复现")
        self.add_argument('--save_best', action='store_true', 
                          help="保存验证集上表现最好的模型")
        self.add_argument('--early_stopping', type=int, default=5, 
                          help="早停策略：在验证集上如果连续 N 轮无提升则停止训练")
        self.add_argument('--grad_clip', type=float, default=5.0,
                          help="梯度裁剪阈值")
        self.add_argument('--model_dir', default=None, type=str,
                          help="如果是测试，加载模型的路径，如果留空会自动从模型参数解析")

        # 训练参数
        self.add_argument('--num_epoch', default=16, type=int, 
                          help="训练轮数")
        self.add_argument('--batch_size', default=64, type=int, 
                          help="批次大小")
        self.add_argument('--loss_interval', default=10, type=int, 
                          help="损失打印间隔（以 bacth 为单位）")
        self.add_argument('--acc_interval', default=100, type=int, 
                          help="准确率计算间隔（以 batch 为单位）")
        self.add_argument('--lr', default=1e-3, type=float, 
                          help="学习率")
        self.add_argument('--fp16', action='store_true', 
                          help="使用混合精度训练")

        # 任务参数
        ## 情感分析
        
        ## 命名实体识别

        ## 翻译
        self.add_argument('--src_lang', default='en', type=str, 
                          help="翻译任务的源语言")
        self.add_argument('--tgt_lang', default='zh', type=str, 
                          help="翻译任务的目标语言")
        
        # 模型参数
        self.add_argument('--emb_dim', default=256, type=int, 
                          help="词向量维度")
        self.add_argument('--dropout', default=0.5, type=float, 
                          help="Dropout 比例")
        self.add_argument('--mlp_dims', default="[]", type=str,
                          help="MLP 隐藏层大小，格式: '[size1, size2, ...]'")

        ## CNN 特定参数
        self.add_argument('--filter_sizes', default="[3,4,5]", type=str,
                          help="CNN 滤波器大小，格式: '[size1, size2, ...]'")
        self.add_argument('--num_filters', default="[2,2,2]", type=str,
                          help="CNN 每个滤波器的数量，格式: '[num1, num2, ...]'")

        ## RNN 特定参数
        self.add_argument('--hidden_size', type=int, default=256, 
                          help="RNN 隐藏层大小")
        self.add_argument('--num_layers', type=int, default=2, 
                          help="RNN/Transformer 层数")
        self.add_argument('--bidirectional', action='store_true', 
                          help="使用双向 RNN")
        self.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'rnn', 'gru'], 
                          help="RNN 类型: LSTM, GRU 或普通 RNN")
        self.add_argument('--only_one', action='store_true',
                            help="只使用一个输出")

        ## Transformer 特定参数
        self.add_argument('--num_heads', type=int, default=4, 
                          help="Transformer 中多头注意力的头数")
        self.add_argument('--ffn_size', type=int, default=1024, 
                          help="Transformer FFN 层的隐藏层大小")
        # num_layers已经在RNN中定义
        # only_one已经在RNN中定义

    def parse_list(self, string):
        """
        解析字符串列表参数，例如 "[3,4,5]" 转为 [3, 4, 5]
        """
        return json.loads(string)
    
    def parse_config(self):
        args = self.parse_args()
        for key, value in vars(args).items():
            if isinstance(value, str) and value.startswith('['):
                setattr(args, key, self.parse_list(value))
        task_config_cls = TASK_CONFIG_CLASSES[args.task]
        model_config_cls = MODEL_CONFIG_CLASSES[args.model]
        base_config = BaseConfig(**self.filter_kwargs(BaseConfig))
        task_config = task_config_cls(**self.filter_kwargs(task_config_cls))
        model_config = model_config_cls(**self.filter_kwargs(model_config_cls))
        return PipelineConfig(base_config, model_config, task_config)
    
    def filter_kwargs(self, cls: Type) -> dict[str, ]:
        args = self.parse_args()
        kwdefaults = {arg: self.get_default(arg) for arg in vars(args)}
        kwargs = vars(args)
        cls_params = signature(cls).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in cls_params and v != kwdefaults[k]}
        return kwargs