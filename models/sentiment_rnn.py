import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from configs.rnn_config import TextRNNConfig
from .text_base_model import TextBaseModel

class SentimentRNN(TextBaseModel):
    def __init__(self, config: TextRNNConfig) -> None:
        super(SentimentRNN, self).__init__(config)
        
        match config.rnn_type.lower():
            case 'lstm':
                rnn_cls = nn.LSTM
            case 'gru':
                rnn_cls = nn.GRU
            case _:
                raise NotImplementedError(f"Unsupported RNN type: {config.rnn_type}")
            
        self.rnn = rnn_cls(
            input_size=config.emb_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        self.classifier = nn.LazyLinear(config.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self._init_lazy()
        
    def features(self, x: torch.LongTensor, mask: torch.Tensor=None) -> torch.Tensor:
        x = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        
        # 计算每个序列的有效长度
        if mask is not None:
            lengths = mask.sum(dim=1)  # (batch_size)
        else:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        lengths = lengths.clamp(min=1)  # 确保长度至少为1，避免索引错误
        # 将输入按序列长度降序排序
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        x_sorted = x[sorted_idx]
        # 打包序列，忽略填充部分
        packed_input = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), 
                                            batch_first=True, enforce_sorted=True)
        # 通过RNN处理
        packed_output, _ = self.rnn(packed_input)
        # 解包输出
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 恢复原始排序
        _, original_idx = sorted_idx.sort()
        output = output[original_idx]
        if not self.rnn.bidirectional:
            # 单向RNN时提取最后一个有效时间步的状态
            last_outputs = output[torch.arange(output.size(0)), lengths - 1]
        else:
            # 双向RNN时提取最后一个有效时间步的前向和后向状态
            hidden_size = self.rnn.hidden_size
            forward_output = output[torch.arange(output.size(0)), lengths - 1, :hidden_size]
            backward_output = output[:, 0, hidden_size:]  # backward的输出来自序列第一个有效位置
            last_outputs = torch.cat([forward_output, backward_output], dim=1)  # (batch_size, hidden_size * 2)
        
        return last_outputs
    
    def forward(self, input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor|None=None,
                labels: torch.LongTensor|None=None) -> dict[str, torch.Tensor]:
        logits = self.classifier(self.features(input_ids, attention_mask))
        if labels is not None:
            loss = self.loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}