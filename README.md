# Text 3-in-1

Text 3-in-1 是一个多任务文本处理项目，支持情感分析、命名实体识别和机器翻译。该项目使用 PyTorch 和 Hugging Face Transformers 库进行模型训练和评估。

## 项目结构

```plaintext
.
├── .gitignore
├── checkpoints/  # 模型检查点
│   ├── ner/  # 命名实体识别模型检查点
│   ├── sentiment/  # 情感分析模型检查点
│   └── translation/  # 翻译模型检查点
├── configs/  # 配置文件
│   ├── __init__.py
│   ├── base_config.py  # 基础配置
│   ├── model_config.py  # 模型配置
│   ├── parser.py  # 配置解析器
│   ├── pipeline_config.py  # 流水线配置
│   └── task_config.py  # 任务配置
├── data/  # 数据文件夹
│   ├── ner/  # 命名实体识别数据
│   ├── sentiment/  # 情感分析数据
│   └── translation/  # 翻译数据
├── data_utils/  # 数据处理工具与数据集
│   ├── __init__.py
│   ├── ner.py  # 命名实体识别数据处理与数据集
│   ├── sentiment.py  # 情感分析数据处理与数据集
│   └── translation.py  # 翻译数据处理与数据集
├── environment.yml  # 环境配置文件
├── LICENSE  # 许可证文件
├── logs/  # 日志文件夹
│   ├── ner/  # 命名实体识别日志
│   ├── sentiment/  # 情感分析日志
│   └── translation/  # 翻译日志
├── main.py  # 主程序入口
├── metrics/  # 评估指标
│   ├── ner_metrics.py  # 命名实体识别评估指标
│   ├── sentiment_metrics.py  # 情感分析评估指标
│   └── translation_metrics.py  # 翻译评估指标
├── models/  # 模型文件夹
│   ├── __init__.py
│   ├── backbone.py  # 模型骨干选择
│   ├── crf.py  # 条件随机场模型
│   ├── encoder.py  # 编码器模型
│   ├── encoder_decoder.py  # 编码器-解码器模型
│   ├── ner_crf.py  # 命名实体识别CRF模型
│   ├── sentiment_model.py  # 情感分析模型
│   ├── text_cnn.py  # 文本CNN模型
│   ├── text_rnn.py  # 文本RNN模型
│   └── translater.py  # 翻译模型
├── README.md  # 项目说明文件
├── scripts/  # 脚本文件夹
├── temp.txt  # 临时文件
├── test.py  # 测试文件
├── tests/  # 测试文件夹
│   ├── ner_rnn.py  # 命名实体识别RNN测试
│   ├── ner_trans.py  # 命名实体识别Transformer测试
│   ├── sentiment_cnn.py  # 情感分析CNN测试
│   ├── sentiment_rnn.py  # 情感分析RNN测试
│   ├── sentiment_trans.py  # 情感分析Transformer测试
│   └── trans_trans.py  # 翻译Transformer测试
└── utils/  # 工具文件夹
    └── example.py  # 用于生成与保存测试结果
```

## 使用说明
1. 克隆仓库：

```sh
git clone https://github.com/SunnyYYLin/text-3in1.git
cd text-3in1
```

2. 配置环境：

```sh
conda create -n text3in1
conda activate text3in1
pip install torch torchvision torchaudio
pip install torchmetrics transformers[torch] safetensors sacrebleu
```

或者

```sh
conda env create -f environment.yml
conda activate text3in1
```

3. 准备数据：

将课程网站上的数据如下放入`data/`文件夹（注意机器翻译数据是最初的70k对的版本）：

```plaintext
data
├── ner
│   ├── chr_vocab.json
│   ├── tag_vocab.json
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── sentiment
│   ├── test.jsonl
│   ├── test.txt
│   ├── train.jsonl
│   ├── train.txt
│   ├── val.jsonl
│   ├── val.txt
│   └── vocab.json
└── translation
    ├── test.en
    ├── test.zh
    ├── train.en
    ├── train.en.json
    ├── train.zh
    ├── train.zh.json
    ├── val.en
    └── val.zh
```

4. 训练模型（参数的值只是示例）：

```sh
# 训练CNN情感分类
python3 main.py --mode="train" --task="sentiment" --model="cnn" --filter_sizes="[2,3,4]" --num_filters="[4,4,4]" --dropout=0.5 --emb_dim=128
# 训练RNN(LSTM)情感分类
python3 main.py --mode="train" --task="sentiment" --model="rnn" --rnn_type="lstm" --hidden_size=256 --num_layers=2 --bidirectional --dropout=0.5 --emb_dim=128
# 训练Encoder-Only Transformer情感分类
python3 main.py --mode="train" --task="sentiment" --model="transformer" --num_heads=4 --ffn_size=256 --num_layers=2 --dropout=0.5 --emb_dim=128
# 训练RNN(LSTM)命名实体识别
python3 main.py --mode="train" --task="ner" --model="rnn" --rnn_type="lstm" --hidden_size=256 --num_layers=2 --bidirectional --dropout=0.5 --emb_dim=128
# 训练Encoder-Only Transformer命名实体识别
python3 main.py --mode="train" --task="ner" --model="transformer" --num_heads=4 --ffn_size=256 --num_layers=2 --dropout=0.5 --emb_dim=128
# 训练Encoder-Decoder Transformer机器翻译
python3 main.py --mode="train" --task="translation" --model="transformer" --num_heads=4 --ffn_size=256 --num_layers=2 --dropout=0.5 --emb_dim=128
```
得到的模型权重将会保存在`checkpoints/{task}/{model_config}`内。

5. 生成测试结果：

例如使用

```sh
python3 main.py --mode="example" --task="sentiment" --model_dir="checkpoints/sentiment/GRU_layers1_hidden256_emb128_dropout0.2_bi"
```

通过调整`task`和`model_dir`，可以自动加载配置，测试不同的模型；也可以指定模型的配置进行测试：

```sh
python3 main.py --mode="example" --task="sentiment" --model="cnn" --filter_sizes="[2,3,4]" --num_filters="[4,4,4]" --dropout=0.5 --emb_dim=128
```

测试结果将会保存到`results/`

## 主函数参数说明

- `--mode`：运行模式，选择 `train`、`test` 或 `example`，默认为 `train`。
- `--task`：任务类型，选择 `sentiment`、`ner` 或 `translation`，默认为 `sentiment`。
- `--model`：模型类型，选择 `cnn`、`rnn`、`transformer` 或 `bert`，默认为 `cnn`。
- `--device`：运行设备，选择 `cuda` 或 `cpu`，默认为 `cuda`。
- `--data_dir`：数据存储路径，默认为 `data`。
- `--save_dir`：模型保存路径，默认为 `checkpoints`。
- `--log_dir`：日志保存路径，默认为 `logs`。
- `--save_best`：保存验证集上表现最好的模型。
- `--early_stopping`：早停策略，在验证集上如果连续 N 轮无提升则停止训练，默认为 `5`。
- `--grad_clip`：梯度裁剪阈值，默认为 `5.0`。
- `--model_dir`：如果是测试，加载模型的路径，如果留空会自动从模型参数解析。

### 训练参数

- `--num_epoch`：训练轮数，默认为 `16`。
- `--batch_size`：批次大小，默认为 `64`。
- `--loss_interval`：损失打印间隔（以 batch 为单位），默认为 `10`。
- `--acc_interval`：准确率计算间隔（以 batch 为单位），默认为 `100`。
- `--lr`：学习率，默认为 `1e-3`。
- `--fp16`：使用混合精度训练。

### 任务参数

#### 翻译任务

- `--src_lang`：翻译任务的源语言，默认为 `en`。
- `--tgt_lang`：翻译任务的目标语言，默认为 `zh`。

### 模型参数

- `--emb_dim`：词向量维度，默认为 `256`。
- `--dropout`：Dropout 比例，默认为 `0.5`。
- `--mlp_dims`：MLP 隐藏层大小，格式：`'[size1, size2, ...]'`。

#### CNN 特定参数

- `--filter_sizes`：CNN 滤波器大小，格式：`'[size1, size2, ...]'`，默认为 `[3,4,5]`。
- `--num_filters`：CNN 每个滤波器的数量，格式：`'[num1, num2, ...]'`，默认为 `[2,2,2]`。

#### RNN 特定参数

- `--hidden_size`：RNN 隐藏层大小，默认为 `256`。
- `--num_layers`：RNN/Transformer 层数，默认为 `2`。
- `--bidirectional`：使用双向 RNN。
- `--rnn_type`：RNN 类型，选择 `lstm`、`rnn` 或 `gru`，默认为 `lstm`。

#### Transformer 特定参数

- `--num_heads`：Transformer 中多头注意力的头数，默认为 `4`。
- `--ffn_size`：Transformer FFN 层的隐藏层大小，默认为 `1024`。
- `--num_layers`：RNN/Transformer 层数，默认为 `2`。

## 许可证

本项目使用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。
