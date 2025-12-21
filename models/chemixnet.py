"""CheMixNet模型实现 - 论文中的三种架构"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CheMixNetCNN(nn.Module):
    """
    CheMixNet CNN*FC架构
    SMILES用CNN + FC，指纹用FC，最后特征拼接
    """
    
    def __init__(self, vocab_size, max_len=100, fp_dim=167, 
                 hidden_dim=256, output_dim=1, dropout_rate=0.3):
        super().__init__()
        
        # SMILES路径 (CNN)
        self.smiles_conv = nn.Sequential(
            nn.Conv1d(vocab_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.smiles_fc = nn.Linear(128, hidden_dim)
        
        # 指纹路径 (FC)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 输出层
        self.output = nn.Linear(hidden_dim // 2, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, smiles, fp=None):
        # 修复：确保输入是正确格式
        if isinstance(smiles, dict):
            # 如果传入的是字典，从中提取数据
            fp = smiles.get('fingerprint', fp)
            smiles = smiles['smiles']
        elif isinstance(smiles, (list, tuple)) and len(smiles) >= 2:
            # 如果传入的是元组或列表
            fp = smiles[1]
            smiles = smiles[0]
        
        # 调试信息
        # print(f"Forward - smiles type: {type(smiles)}, fp type: {type(fp)}")
        
        # 确保数据在正确的设备上
        device = next(self.parameters()).device
        if hasattr(smiles, 'to'):
            smiles = smiles.to(device)
        if fp is not None and hasattr(fp, 'to'):
            fp = fp.to(device)
        
        # SMILES路径
        # 注意：输入的smiles应该是(batch_size, seq_len, vocab_size)形状
        # 但Conv1d需要(batch_size, channels, seq_len)形状
        if smiles.dim() == 3:
            # 如果是one-hot编码 (batch_size, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = smiles.shape
            # 转换为 (batch_size, vocab_size, seq_len)
            smiles_input = smiles.transpose(1, 2)
        else:
            # 如果已经是合适的形状
            smiles_input = smiles
            
        smiles_features = self.smiles_conv(smiles_input).squeeze(-1)
        smiles_features = self.smiles_fc(smiles_features)
        
        # 指纹路径
        fp_features = self.fp_fc(fp)
        
        # 特征拼接
        combined = torch.cat([smiles_features, fp_features], dim=1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 输出
        output = self.output(fused)
        
        return output

class CheMixNetRNN(nn.Module):
    """
    CheMixNet RNN*FC架构
    SMILES用LSTM/GRU + FC
    """
    
    def __init__(self, vocab_size, max_len=100, fp_dim=167,
                 hidden_dim=128, lstm_layers=2, output_dim=1, dropout_rate=0.3):
        super().__init__()
        
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        
        # SMILES路径 (RNN)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True, 
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False
        )
        self.smiles_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 指纹路径 (FC)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层
        self.output = nn.Linear(hidden_dim // 2, output_dim)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, smiles, fp=None):
        # 修复：确保输入是正确格式
        if isinstance(smiles, dict):
            # 如果传入的是字典，从中提取数据
            fp = smiles.get('fingerprint', fp)
            smiles = smiles['smiles']
        elif isinstance(smiles, (list, tuple)) and len(smiles) >= 2:
            # 如果传入的是元组或列表
            fp = smiles[1]
            smiles = smiles[0]
        
        # 确保数据在正确的设备上
        device = self.embedding.weight.device
        if hasattr(smiles, 'to'):
            smiles = smiles.to(device)
        if fp is not None and hasattr(fp, 'to'):
            fp = fp.to(device)
        
        # 将one-hot转换为索引
        if smiles.dim() == 3:  # 如果是one-hot编码 (batch, seq_len, vocab_size)
            smiles_indices = torch.argmax(smiles, dim=-1)
        else:
            smiles_indices = smiles
        
        # SMILES路径
        embedded = self.embedding(smiles_indices)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一层隐藏状态
        smiles_features = self.smiles_fc(hidden[-1])
        
        # 指纹路径
        if fp is not None:
            fp_features = self.fp_fc(fp)
        else:
            # 如果没有提供指纹，创建零张量
            fp_features = torch.zeros(batch_size, self.fp_fc[-2].out_features, 
                                    device=device)
        
        # 特征拼接
        combined = torch.cat([smiles_features, fp_features], dim=1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 输出
        output = self.output(fused)
        
        return output

class CheMixNetCNNRNN(nn.Module):
    """
    CheMixNet CNN+RNN*FC架构
    SMILES同时用CNN和RNN提取特征
    """
    
    def __init__(self, vocab_size, max_len=100, fp_dim=167,
                 cnn_channels=[64, 128], lstm_hidden=128, lstm_layers=2,
                 output_dim=1, dropout_rate=0.3):
        super().__init__()
        
        # CNN分支
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(max_len, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # RNN分支
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # 指纹路径
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(cnn_channels[1] + lstm_hidden * 2 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 输出层
        self.output = nn.Linear(128, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, smiles, fp=None):
        # 修复：确保输入是正确格式
        if isinstance(smiles, dict):
            # 如果传入的是字典，从中提取数据
            fp = smiles.get('fingerprint', fp)
            smiles = smiles['smiles']
        elif isinstance(smiles, (list, tuple)) and len(smiles) >= 2:
            # 如果传入的是元组或列表
            fp = smiles[1]
            smiles = smiles[0]
        
        # 确保数据在正确的设备上
        if smiles.device != self.cnn_branch[0].weight.device:
            smiles = smiles.to(self.cnn_branch[0].weight.device)
        if fp is not None and fp.device != self.fp_fc[0].weight.device:
            fp = fp.to(self.fp_fc[0].weight.device)
        
        # CNN分支
        cnn_features = self.cnn_branch(smiles).squeeze(-1)
        
        # RNN分支
        if smiles.dim() == 3:  # 如果是one-hot编码 (batch, seq_len, vocab_size)
            smiles_indices = torch.argmax(smiles, dim=-1)
        else:
            smiles_indices = smiles
            
        embedded = self.embedding(smiles_indices)  # 从one-hot转回索引
        lstm_out, (hidden, _) = self.lstm(embedded)
        # 使用最后一个时间步的隐藏状态
        rnn_features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # 双向LSTM拼接
        
        # 指纹路径
        fp_features = self.fp_fc(fp)
        
        # 特征融合
        combined = torch.cat([cnn_features, rnn_features, fp_features], dim=1)
        fused = self.fusion(combined)
        
        # 输出
        output = self.output(fused)
        
        return output

# 别名
CheMixNet = CheMixNetCNN  # 默认使用CNN*FC架构