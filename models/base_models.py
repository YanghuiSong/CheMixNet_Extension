"""基线模型实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """基于指纹的MLP基线模型"""
    
    def __init__(self, input_dim=167, hidden_dims=[256, 128, 64], 
                 output_dim=1, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.layers(x)
        output = self.output_layer(features)
        return output

class CNNBaseline(nn.Module):
    """基于SMILES的CNN基线模型"""
    
    def __init__(self, vocab_size, max_len=100, output_dim=1):
        super().__init__()
        
        # 嵌入层（替代one-hot）
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        
        # CNN层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x的形状: (batch, max_len, vocab_size) -> one-hot
        # 检查输入维度
        if len(x.shape) == 3:
            # 如果是三维张量 (batch, max_len, vocab_size)
            # 转换为索引
            x_indices = torch.argmax(x, dim=-1)
        elif len(x.shape) == 2:
            # 如果已经是二维张量 (batch, max_len)
            x_indices = x.long()
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # 嵌入
        embedded = self.embedding(x_indices)  # (batch, max_len, embed_dim)
        
        # 转置以适应Conv1d
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, max_len)
        
        # CNN特征提取
        features = self.conv_layers(embedded)  # (batch, 256, 1)
        features = features.squeeze(-1)  # (batch, 256)
        
        # 全连接层
        output = self.fc_layers(features)
        return output

class RNNBaseline(nn.Module):
    """基于SMILES的RNN基线模型"""
    
    def __init__(self, vocab_size, max_len=100, hidden_dim=128, 
                 num_layers=2, output_dim=1, bidirectional=True):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # 全连接层
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        
    def forward(self, x):
        # 处理不同类型的输入
        if len(x.shape) == 3:
            # 如果是三维张量 (batch, max_len, vocab_size)
            # 转换为索引
            x_indices = torch.argmax(x, dim=-1)
        elif len(x.shape) == 2:
            # 如果已经是二维张量 (batch, max_len)
            x_indices = x.long()
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # 确保索引在合法范围内
        x_indices = torch.clamp(x_indices, 0, self.embedding.num_embeddings - 1)
        
        # 嵌入
        embedded = self.embedding(x_indices)  # (batch, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 合并前向和后向的最后时间步输出
            forward_hidden = lstm_out[:, -1, :self.hidden_dim]
            backward_hidden = lstm_out[:, 0, self.hidden_dim:]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # 全连接层
        output = self.fc(final_hidden)  # (batch, output_dim)
        return output
    
    def get_attention_weights(self, x):
        """获取注意力权重（用于可视化）"""
        with torch.no_grad():
            x_indices = torch.argmax(x, dim=-1)
            embedded = self.embedding(x_indices)
            lstm_out, _ = self.lstm(embedded)
            attention_weights = torch.softmax(
                self.attention(lstm_out).squeeze(-1), dim=1
            )
            return attention_weights.cpu().numpy()