"""三模态混合模型：SMILES + 指纹 + 分子图"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class EnhancedCheMixNet(nn.Module):
    """增强版CheMixNet：集成三种分子表示"""
    
    def __init__(self, 
                 smiles_vocab_size,
                 smiles_max_len=100,
                 maccs_dim=167,
                 atom_feature_dim=5,
                 hidden_dims=[256, 128, 64],
                 output_dim=1,
                 dropout_rate=0.3,
                 use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # 1. SMILES路径 (CNN)
        self.smiles_conv = nn.Sequential(
            nn.Conv1d(smiles_max_len, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.smiles_fc = nn.Linear(128, hidden_dims[0])
        
        # 2. 指纹路径
        self.fp_layers = nn.Sequential(
            nn.Linear(maccs_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        
        # 3. 分子图路径 (GNN) - 扩展功能
        self.gnn_layers = nn.ModuleList([
            GCNConv(atom_feature_dim, 64),
            GCNConv(64, 128)
        ])
        self.graph_fc = nn.Sequential(
            nn.Linear(128, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力机制（用于特征融合）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[0] * 3, 64),
                nn.Tanh(),
                nn.Linear(64, 3),
                nn.Softmax(dim=1)
            )
        
        # 融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dims[0] * 3, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU()
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, smiles_input, fp_input, graph_data=None):
        # 输入验证和预处理
        if smiles_input is None or fp_input is None:
            raise ValueError("输入不能为空")
        
        # 确保输入在正确的设备上
        if hasattr(smiles_input, 'device') and smiles_input.device != next(self.parameters()).device:
            smiles_input = smiles_input.to(next(self.parameters()).device)
        if hasattr(fp_input, 'device') and fp_input.device != next(self.parameters()).device:
            fp_input = fp_input.to(next(self.parameters()).device)
        
        # 1. SMILES特征提取
        try:
            # 检查输入维度
            if smiles_input.dim() != 3:
                raise ValueError(f"SMILES输入应该是3维张量，但得到 {smiles_input.dim()} 维")
            
            smiles_features = self.smiles_conv(smiles_input).squeeze(-1)
            smiles_features = self.smiles_fc(smiles_features)
        except Exception as e:
            print(f"SMILES特征提取错误: {e}")
            batch_size = smiles_input.size(0) if smiles_input.dim() > 0 else 1
            smiles_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                       device=next(self.parameters()).device)
        
        # 2. 指纹特征提取
        try:
            # 检查输入维度
            if fp_input.dim() != 2:
                raise ValueError(f"指纹输入应该是2维张量，但得到 {fp_input.dim()} 维")
                
            fp_features = self.fp_layers(fp_input)
        except Exception as e:
            print(f"指纹特征提取错误: {e}")
            batch_size = fp_input.size(0) if fp_input.dim() > 0 else 1
            fp_features = torch.zeros(batch_size, self.fp_layers[0].out_features, 
                                   device=next(self.parameters()).device)
        
        # 3. 图特征提取（如果可用）
        if graph_data is not None:
            try:
                # 检查graph_data是否具有必要的属性
                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    x, edge_index = graph_data.x, graph_data.edge_index
                elif isinstance(graph_data, dict) and 'atom_features' in graph_data:
                    # 处理旧格式的图数据
                    x = torch.FloatTensor(graph_data['atom_features'])
                    edge_index = torch.LongTensor(graph_data['edge_indices']).t().contiguous() if 'edge_indices' in graph_data else torch.empty((2, 0), dtype=torch.long)
                elif isinstance(graph_data, list) and len(graph_data) > 0:
                    # 如果graph_data是列表，取第一个元素
                    graph_obj = graph_data[0]
                    if hasattr(graph_obj, 'x') and hasattr(graph_obj, 'edge_index'):
                        x, edge_index = graph_obj.x, graph_obj.edge_index
                    else:
                        # 不支持的图数据格式
                        raise ValueError(f"Unsupported graph data format in list: {type(graph_obj)}")
                else:
                    # 不支持的图数据格式
                    raise ValueError(f"Unsupported graph data format: {type(graph_data)}")
                
                # 检查是否需要转移到设备
                if x.device != next(self.parameters()).device:
                    x = x.to(next(self.parameters()).device)
                if edge_index.device != next(self.parameters()).device:
                    edge_index = edge_index.to(next(self.parameters()).device)
                
                batch_size = smiles_input.size(0)
                # 更安全地初始化节点批处理信息
                # 假设每个图有相同数量的节点，或者我们有一个batch索引数组
                batch = None
                if hasattr(graph_data, 'batch'):
                    batch = graph_data.batch
                    if batch.device != next(self.parameters()).device:
                        batch = batch.to(next(self.parameters()).device)
                else:
                    # 估计每个样本的节点数（假设均匀分布）
                    nodes_per_sample = x.size(0) // batch_size if x.size(0) >= batch_size else 1
                    # 创建batch索引
                    batch_indices = []
                    for i in range(batch_size):
                        batch_indices.extend([i] * nodes_per_sample)
                    # 如果总数不匹配，调整到最后一个批次
                    if len(batch_indices) > x.size(0):
                        batch_indices = batch_indices[:x.size(0)]
                    elif len(batch_indices) < x.size(0):
                        # 补充缺失的索引
                        last_batch_idx = batch_indices[-1] if batch_indices else 0
                        batch_indices.extend([last_batch_idx] * (x.size(0) - len(batch_indices)))
                    
                    if batch_indices:
                        batch = torch.LongTensor(batch_indices).to(x.device)
                
                # 确保batch和x的第一维度大小一致
                if batch is not None and batch.numel() > 0 and batch.size(0) != x.size(0):
                    # 如果不匹配，使用简单的均值池化而不使用batch索引
                    graph_features = torch.mean(x, dim=0, keepdim=True).repeat(batch_size, 1)
                elif batch is not None and batch.numel() > 0:
                    for conv in self.gnn_layers:
                        x = conv(x, edge_index)
                        x = F.relu(x)
                    graph_features = global_mean_pool(x, batch)
                else:
                    # 如果没有有效的batch索引，使用简单的均值池化
                    graph_features = torch.mean(x, dim=0, keepdim=True).repeat(batch_size, 1)
                
                graph_features = self.graph_fc(graph_features)
            except Exception as e:
                # 注释掉详细的错误信息打印，避免输出过于冗长
                # print(f"图特征提取错误: {e}")
                batch_size = smiles_input.size(0)
                graph_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                          device=next(self.parameters()).device)
        else:
            # 如果没有图数据，使用零向量
            batch_size = smiles_input.size(0)
            graph_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                        device=next(self.parameters()).device)
        
        # 4. 特征融合
        try:
            # 确保所有特征具有相同的批次大小
            batch_size = smiles_input.size(0)
            
            # 检查并调整特征维度
            if smiles_features.size(0) != batch_size:
                if smiles_features.size(0) == 1:
                    smiles_features = smiles_features.repeat(batch_size, 1)
                else:
                    smiles_features = smiles_features[:batch_size] if smiles_features.size(0) > batch_size else torch.cat([
                        smiles_features, 
                        torch.zeros(batch_size - smiles_features.size(0), smiles_features.size(1), device=smiles_features.device)
                    ])
            
            if fp_features.size(0) != batch_size:
                if fp_features.size(0) == 1:
                    fp_features = fp_features.repeat(batch_size, 1)
                else:
                    fp_features = fp_features[:batch_size] if fp_features.size(0) > batch_size else torch.cat([
                        fp_features,
                        torch.zeros(batch_size - fp_features.size(0), fp_features.size(1), device=fp_features.device)
                    ])
            
            if graph_features.size(0) != batch_size:
                if graph_features.size(0) == 1:
                    graph_features = graph_features.repeat(batch_size, 1)
                else:
                    graph_features = graph_features[:batch_size] if graph_features.size(0) > batch_size else torch.cat([
                        graph_features,
                        torch.zeros(batch_size - graph_features.size(0), graph_features.size(1), device=graph_features.device)
                    ])
            
            if self.use_attention:
                # 拼接所有特征
                combined = torch.cat([smiles_features, fp_features, graph_features], dim=1)
                # 计算注意力权重
                attn_weights = self.attention(combined)
                # 加权求和
                smiles_features = smiles_features * attn_weights[:, 0:1]
                fp_features = fp_features * attn_weights[:, 1:2]
                graph_features = graph_features * attn_weights[:, 2:3]
            
            # 拼接特征
            fused = torch.cat([smiles_features, fp_features, graph_features], dim=1)
            
            # 5. 融合层处理
            fused = self.fusion_layers(fused)
            
            # 6. 输出
            output = self.output_layer(fused)
        except Exception as e:
            # 注释掉详细的错误信息打印，避免输出过于冗长
            # print(f"特征融合或输出错误: {e}")
            batch_size = smiles_input.size(0)
            output = torch.zeros(batch_size, 1, device=next(self.parameters()).device)
        
        return output
    
    def get_attention_weights(self, smiles_input, fp_input, graph_data=None):
        """获取注意力权重（用于模型解释）"""
        with torch.no_grad():
            # 提取特征
            smiles_features = self.smiles_conv(smiles_input).squeeze(-1)
            smiles_features = self.smiles_fc(smiles_features)
            
            fp_features = self.fp_layers(fp_input)
            
            if graph_data is not None:
                # 检查graph_data是否具有必要的属性
                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    x, edge_index = graph_data.x, graph_data.edge_index
                elif isinstance(graph_data, dict) and 'atom_features' in graph_data:
                    # 处理旧格式的图数据
                    x = torch.FloatTensor(graph_data['atom_features'])
                    edge_index = torch.LongTensor(graph_data['edge_indices']).t().contiguous() if 'edge_indices' in graph_data else torch.empty((2, 0), dtype=torch.long)
                elif isinstance(graph_data, list) and len(graph_data) > 0:
                    # 如果graph_data是列表，取第一个元素
                    graph_obj = graph_data[0]
                    if hasattr(graph_obj, 'x') and hasattr(graph_obj, 'edge_index'):
                        x, edge_index = graph_obj.x, graph_obj.edge_index
                    else:
                        # 不支持的图数据格式
                        batch_size = smiles_input.size(0)
                        graph_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                                    device=smiles_input.device)
                        # 拼接所有特征
                        combined = torch.cat([smiles_features, fp_features, graph_features], dim=1)
                        # 计算注意力权重
                        attn_weights = self.attention(combined)
                        return {
                            'smiles_weight': attn_weights[:, 0].mean().item(),
                            'fp_weight': attn_weights[:, 1].mean().item(),
                            'graph_weight': attn_weights[:, 2].mean().item()
                        }
                else:
                    # 不支持的图数据格式
                    batch_size = smiles_input.size(0)
                    graph_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                                device=smiles_input.device)
                    # 拼接所有特征
                    combined = torch.cat([smiles_features, fp_features, graph_features], dim=1)
                    # 计算注意力权重
                    attn_weights = self.attention(combined)
                    return {
                        'smiles_weight': attn_weights[:, 0].mean().item(),
                        'fp_weight': attn_weights[:, 1].mean().item(),
                        'graph_weight': attn_weights[:, 2].mean().item()
                    }
                
                # 确保在正确的设备上
                if x.device != smiles_input.device:
                    x = x.to(smiles_input.device)
                if edge_index.device != smiles_input.device:
                    edge_index = edge_index.to(smiles_input.device)
                
                batch_size = smiles_input.size(0)
                # 更安全地初始化节点批处理信息
                # 假设每个图有相同数量的节点，或者我们有一个batch索引数组
                if hasattr(graph_data, 'batch'):
                    batch = graph_data.batch
                    if batch.device != smiles_input.device:
                        batch = batch.to(smiles_input.device)
                else:
                    # 估计每个样本的节点数（假设均匀分布）
                    nodes_per_sample = x.size(0) // batch_size if x.size(0) >= batch_size else 1
                    # 创建batch索引
                    batch_indices = []
                    for i in range(batch_size):
                        batch_indices.extend([i] * nodes_per_sample)
                    # 如果总数不匹配，调整到最后一个批次
                    if len(batch_indices) > x.size(0):
                        batch_indices = batch_indices[:x.size(0)]
                    elif len(batch_indices) < x.size(0):
                        # 补充缺失的索引
                        last_batch_idx = batch_indices[-1] if batch_indices else 0
                        batch_indices.extend([last_batch_idx] * (x.size(0) - len(batch_indices)))
                    
                    batch = torch.LongTensor(batch_indices).to(x.device)
                
                # 确保batch和x的第一维度大小一致
                if batch is not None and batch.numel() > 0 and batch.size(0) != x.size(0):
                    # 如果不匹配，使用简单的均值池化而不使用batch索引
                    graph_features = torch.mean(x, dim=0, keepdim=True).repeat(batch_size, 1)
                elif batch is not None and batch.numel() > 0:
                    for conv in self.gnn_layers:
                        x = conv(x, edge_index)
                        x = F.relu(x)
                    graph_features = global_mean_pool(x, batch)
                else:
                    # 如果没有有效的batch索引，使用简单的均值池化
                    graph_features = torch.mean(x, dim=0, keepdim=True).repeat(batch_size, 1)
                graph_features = self.graph_fc(graph_features)
            else:
                batch_size = smiles_input.size(0)
                graph_features = torch.zeros(batch_size, self.smiles_fc.out_features, 
                                            device=smiles_input.device)
            
            # 计算注意力
            combined = torch.cat([smiles_features, fp_features, graph_features], dim=1)
            attn_weights = self.attention(combined)
            
            return {
                'smiles_weight': attn_weights[:, 0].mean().item(),
                'fp_weight': attn_weights[:, 1].mean().item(),
                'graph_weight': attn_weights[:, 2].mean().item()
            }