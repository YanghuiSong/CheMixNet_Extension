"""图神经网络模块（扩展功能）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

class GNNLayer(nn.Module):
    """通用的GNN层"""
    
    def __init__(self, in_channels, out_channels, gnn_type='gcn', 
                 heads=1, dropout=0.2):
        super().__init__()
        
        self.gnn_type = gnn_type
        
        if gnn_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            self.conv = GATConv(in_channels, out_channels, 
                               heads=heads, dropout=dropout)
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        self.norm = nn.BatchNorm1d(out_channels * heads if gnn_type == 'gat' else out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class GNNModel(nn.Module):
    """完整的GNN模型"""
    
    def __init__(self, node_feature_dim, hidden_channels=128, 
                 num_layers=3, gnn_type='gcn', output_dim=1, 
                 pooling='mean', dropout=0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GNNLayer(node_feature_dim, hidden_channels, 
                                   gnn_type, dropout=dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_channels, hidden_channels,
                                       gnn_type, dropout=dropout))
        
        # 池化方式
        self.pooling = pooling
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        # 逐层传播
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # 图池化
        if batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                x = global_max_pool(x, batch)
            elif self.pooling == 'mean_max':
                x_mean = global_mean_pool(x, batch)
                x_max = global_max_pool(x, batch)
                x = torch.cat([x_mean, x_max], dim=1)
        else:
            # 如果没有batch信息，使用全局平均
            x = x.mean(dim=0, keepdim=True)
        
        # 全连接层
        output = self.fc(x)
        return output
    
    def get_node_embeddings(self, x, edge_index, edge_attr=None):
        """获取节点嵌入（用于可视化）"""
        embeddings = [x]
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            embeddings.append(x.detach())
        
        return embeddings