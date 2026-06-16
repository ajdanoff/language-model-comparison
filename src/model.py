"""
Определение моделей LSTM и GRU
"""

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    """Mодель для языкового моделирования с LSTM/GRU"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, rnn_type="LSTM"):
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # ОДНОНАПРАВЛЕННЫЙ LSTM/GRU (не двунаправленный!)
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        """Инициализация весов"""
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.weight)
        nn.init.uniform_(self.linear.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """Форвард проход"""
        emb = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        out, _ = self.rnn(emb)   # (batch_size, seq_length, hidden_dim)
        output = self.linear(out)  # (batch_size, seq_length, vocab_size)
        return output


def count_parameters(model):
    """Подсчёт числа обучаемых параметров"""
    return sum(p.numel() for p in model.parameters())