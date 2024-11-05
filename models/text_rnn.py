import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from configs.model_config import RNNConfig

class TextRNN(nn.Module):
    def __init__(self, config: RNNConfig) -> None:
        super(TextRNN, self).__init__()
        self.only_last = config.only_last
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
        
    def forward(self, emb: torch.Tensor, 
                attention_mask: torch.LongTensor|None=None) -> torch.Tensor:
        # 计算每个序列的有效长度
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)  # (batch_size)
        else:
            lengths = torch.full((emb.size(0),), emb.size(1), 
                                 dtype=torch.long, device=emb.device)
        lengths = lengths.clamp(min=1)  # 确保长度至少为1，避免索引错误
        # 将输入按序列长度降序排序
        lengths_sorted, sorted_idx = lengths.sort(descending=True)
        emb_sorted = emb[sorted_idx]
        # 打包序列，忽略填充部分
        packed_input = pack_padded_sequence(emb_sorted, lengths_sorted.cpu(), 
                                            batch_first=True, enforce_sorted=True)
        # 通过RNN处理
        packed_output, _ = self.rnn(packed_input)
        # 解包输出
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 恢复原始排序
        _, original_idx = sorted_idx.sort()
        output = output[original_idx]
        
        if self.only_last:
            if self.rnn.bidirectional:
                # 双向RNN时提取最后一个有效时间步的前向和后向状态
                forward_output = output[torch.arange(output.size(0)), lengths - 1, :self.rnn.hidden_size]
                backward_output = output[:, 0, self.rnn.hidden_size:]  # backward的输出来自序列第一个有效位置
                features = torch.cat([forward_output, backward_output], dim=1)  # (batch_size, hidden_size * 2)
            else:
                # 单向RNN时提取最后一个有效时间步的状态
                features = output[torch.arange(output.size(0)), lengths - 1]
        else:
            if self.rnn.bidirectional:
                output = output.view(output.size(0), output.size(1), 2, self.rnn.hidden_size)  # (batch_size, seq_len, 2, hidden_size)
                features = torch.cat((output[:, :, 0, :], output[:, :, 1, :]), dim=-1)  # (batch_size, seq_len, hidden_size * 2)
            else:
                features = output
        
        return features