"""完整的训练流程管理"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, precision_score, recall_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from .metrics import calculate_regression_metrics, calculate_classification_metrics

class AdvancedTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 强制使用CUDA设备
        force_cuda = config.get('force_cuda', False)
        if force_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("强制使用CUDA设备")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("检测到CUDA设备，自动使用GPU")
        else:
            self.device = torch.device('cpu')
            print("警告：未检测到CUDA设备，使用CPU进行训练")
        self.model.to(self.device)
        
        # 优化器和损失函数
        # 确保weight_decay是浮点数类型
        weight_decay = config.get('weight_decay', 1e-5)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # 获取并设置梯度裁剪阈值
        self.gradient_clip = config.get('gradient_clip', 1.0)
            
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 损失函数
        self.task_type = config.get('task_type', 'regression')
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
            self.metrics = ['mse', 'rmse', 'mae', 'r2']
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self._get_class_weights())
            self.metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        
        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': [],
            'lr': []
        }
        
        # 早停
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # 是否保存模型检查点
        self.save_checkpoints = config.get('save_checkpoints', False)
        
    def _get_class_weights(self):
        """计算类别权重（用于不平衡数据）"""
        # 从数据加载器获取
        return torch.tensor([1.0]).to(self.device)  # 简化版，并确保在正确设备上
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(train_loader, desc='Training')
        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            try:
                # 添加调试信息
                # 注释掉详细的批次信息打印，避免输出过于冗长
                # print(f"Batch {batch_idx}: 类型={type(batch)}")
                # if isinstance(batch, dict):
                #     print(f"  字典键: {list(batch.keys())}")
                #     for k, v in batch.items():
                #         print(f"    {k}: 类型={type(v)}")
                #         if hasattr(v, 'shape'):
                #             print(f"         形状={v.shape}")
                # elif isinstance(batch, (list, tuple)):
                #     print(f"  列表/元组长度: {len(batch)}")
                #     for i, item in enumerate(batch):
                #         print(f"    [{i}]: 类型={type(item)}")
                #         if hasattr(item, 'shape'):
                #             print(f"          形状={item.shape}")
                # elif isinstance(batch, str):
                #     print(f"  字符串内容前50字符: {batch[:50]}...")
                # else:
                #     print(f"  其他类型内容: {batch}")
                
                # 获取数据并移动到设备
                # 处理不同类型的输入数据
                output = None  # 确保output变量始终有定义
                if isinstance(batch, dict):
                    if 'smiles' in batch and 'fingerprint' in batch:
                        # 混合模型输入（SMILES + 指纹）
                        smiles = batch['smiles']
                        fp = batch['fingerprint']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(fp, torch.Tensor):
                            print(f"警告: 指纹不是张量类型: {type(fp)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        fp = fp.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                            
                        # 图数据（如果可用）
                        graph_data = batch.get('graph_data', None)
                        if graph_data is not None:
                            # 检查graph_data是否具有必要的属性
                            if hasattr(graph_data, 'to'):
                                graph_data = graph_data.to(self.device)
                            elif isinstance(graph_data, list) and len(graph_data) > 0:
                                # 如果graph_data是列表，取第一个元素
                                graph_data = graph_data[0]
                                if hasattr(graph_data, 'to'):
                                    graph_data = graph_data.to(self.device)
                            try:
                                output = self.model(smiles, fp, graph_data)
                            except Exception as e:
                                print(f"模型前向传播错误: {e}")
                                output = self.model(smiles, fp)  # 回退到不使用图数据
                        else:
                            output = self.model(smiles, fp)
                    elif 'features' in batch:
                        # 仅指纹输入
                        features = batch['features']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(features, torch.Tensor):
                            print(f"警告: 特征不是张量类型: {type(features)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        features = features.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        output = self.model(features)
                    else:
                        # 仅SMILES输入
                        smiles = batch['smiles']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        output = self.model(smiles)
                elif isinstance(batch, (list, tuple)):
                    # 处理元组或其他格式
                    if len(batch) >= 2:
                        smiles = batch[0]
                        target = batch[1]
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        
                        if len(batch) >= 3:
                            # 有指纹数据
                            fp = batch[2]
                            if isinstance(fp, torch.Tensor):
                                fp = fp.to(self.device).detach().requires_grad_(True)
                                output = self.model(smiles, fp)
                            else:
                                output = self.model(smiles)
                        else:
                            output = self.model(smiles)
                    else:
                        print(f"Unsupported batch format: {type(batch)}, length: {len(batch)}")
                        continue
                
                # 确保output变量已经被正确赋值
                if output is None:
                    print(f"警告: output变量未被正确赋值，跳过该批次")
                    continue
                
                # 计算损失
                loss = self.criterion(output.view(-1), target.float().view(-1))
                total_loss += loss.item()
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                
                # 收集预测和目标
                all_preds.extend(output.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                # 注释掉详细的错误堆栈信息打印，避免输出过于冗长
                # import traceback
                # traceback.print_exc()
                continue
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # 计算指标
        try:
            if self.task_type == 'regression':
                metrics = calculate_regression_metrics(all_targets, all_preds)
            else:
                metrics = calculate_classification_metrics(all_targets, all_preds)
        except Exception as e:
            print(f"计算训练指标时发生错误: {e}")
            metrics = {}
        
        return avg_loss, metrics
                
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(val_loader, desc='Validation')
        batch_idx = 0
        for batch in pbar:
            batch_idx += 1
            try:
                # 添加调试信息
                # 注释掉详细的批次信息打印，避免输出过于冗长
                # print(f"Batch {batch_idx}: 类型={type(batch)}")
                # if isinstance(batch, dict):
                #     print(f"  字典键: {list(batch.keys())}")
                #     for k, v in batch.items():
                #         print(f"    {k}: 类型={type(v)}")
                #         if hasattr(v, 'shape'):
                #             print(f"         形状={v.shape}")
                # elif isinstance(batch, (list, tuple)):
                #     print(f"  列表/元组长度: {len(batch)}")
                #     for i, item in enumerate(batch):
                #         print(f"    [{i}]: 类型={type(item)}")
                #         if hasattr(item, 'shape'):
                #             print(f"          形状={item.shape}")
                # elif isinstance(batch, str):
                #     print(f"  字符串内容前50字符: {batch[:50]}...")
                # else:
                #     print(f"  其他类型内容: {batch}")
                
                # 获取数据并移动到设备
                # 处理不同类型的输入数据
                output = None  # 确保output变量始终有定义
                if isinstance(batch, dict):
                    if 'smiles' in batch and 'fingerprint' in batch:
                        # 混合模型输入（SMILES + 指纹）
                        smiles = batch['smiles']
                        fp = batch['fingerprint']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(fp, torch.Tensor):
                            print(f"警告: 指纹不是张量类型: {type(fp)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        fp = fp.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                            
                        # 图数据（如果可用）
                        graph_data = batch.get('graph_data', None)
                        if graph_data is not None:
                            # 检查graph_data是否具有必要的属性
                            if hasattr(graph_data, 'to'):
                                graph_data = graph_data.to(self.device)
                            elif isinstance(graph_data, list) and len(graph_data) > 0:
                                # 如果graph_data是列表，取第一个元素
                                graph_data = graph_data[0]
                                if hasattr(graph_data, 'to'):
                                    graph_data = graph_data.to(self.device)
                            try:
                                output = self.model(smiles, fp, graph_data)
                            except Exception as e:
                                print(f"模型前向传播错误: {e}")
                                output = self.model(smiles, fp)  # 回退到不使用图数据
                        else:
                            output = self.model(smiles, fp)
                    elif 'features' in batch:
                        # 仅指纹输入
                        features = batch['features']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(features, torch.Tensor):
                            print(f"警告: 特征不是张量类型: {type(features)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        features = features.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        output = self.model(features)
                    else:
                        # 仅SMILES输入
                        smiles = batch['smiles']
                        target = batch['target']
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        output = self.model(smiles)
                elif isinstance(batch, (list, tuple)):
                    # 处理元组或其他格式
                    if len(batch) >= 2:
                        smiles = batch[0]
                        target = batch[1]
                        
                        # 确保数据是张量并且在正确的设备上
                        if not isinstance(smiles, torch.Tensor):
                            print(f"警告: SMILES不是张量类型: {type(smiles)}")
                            continue
                        if not isinstance(target, torch.Tensor):
                            print(f"警告: 目标不是张量类型: {type(target)}")
                            continue
                            
                        # 移动到正确的设备并确保需要梯度
                        smiles = smiles.to(self.device).detach().requires_grad_(True)
                        target = target.to(self.device)
                        
                        if len(batch) >= 3:
                            # 有指纹数据
                            fp = batch[2]
                            if isinstance(fp, torch.Tensor):
                                fp = fp.to(self.device).detach().requires_grad_(True)
                                output = self.model(smiles, fp)
                            else:
                                output = self.model(smiles)
                        else:
                            output = self.model(smiles)
                    else:
                        print(f"Unsupported batch format: {type(batch)}, length: {len(batch)}")
                        continue
                
                # 确保output变量已经被正确赋值
                if output is None:
                    print(f"警告: output变量未被正确赋值，跳过该批次")
                    continue
                
                # 计算损失
                loss = self.criterion(output.view(-1), target.float().view(-1))
                total_loss += loss.item()
                
                # 收集预测和目标
                all_preds.extend(output.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"验证过程中发生错误: {e}")
                # 注释掉详细的错误堆栈信息打印，避免输出过于冗长
                # import traceback
                # traceback.print_exc()
                continue
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # 计算指标
        try:
            if self.task_type == 'regression':
                metrics = calculate_regression_metrics(all_targets, all_preds)
            else:
                metrics = calculate_classification_metrics(all_targets, all_preds)
        except Exception as e:
            print(f"计算验证指标时发生错误: {e}")
            metrics = {}
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, epochs):
        """训练模型"""
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # 更新历史记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # 打印训练和验证结果
            print(f"Train Loss: {train_loss:.4f}, Train Metrics: {train_metrics}")
            print(f"Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            
            # 早停逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                if self.save_checkpoints:
                    torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("早停：验证损失未改善超过指定的耐心值")
                    break
    
    def validate(self, test_loader):
        """在测试集上评估模型"""
        with torch.no_grad():
            test_loss, test_metrics = self.validate_epoch(test_loader)
        return test_loss, test_metrics
