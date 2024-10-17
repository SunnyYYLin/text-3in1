import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config) -> None:
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size + 1, config.embedding_dim, -1)
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, (k, config.embedding_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(config.filter_sizes), config.num_classes)
        
    def forward(self, x):
        x = self.embedding(x) # batch_size, seq_len, embedding_dim
        x = x.unsqueeze(1) # batch_size, 1, seq_len, embedding_dim
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, 1, seq_len), ...]*len(filter_sizes)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x] # [(batch_size, 1), ...]*len(filter_sizes)
        x = torch.cat(x, 1) # batch_size, len(filter_sizes)
        x = self.dropout(x) # batch_size, len(filter_sizes)
        logits = self.fc(x) # batch_size, num_classes
        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        tag = torch.argmax(logits, dim=-1)
        return tag
