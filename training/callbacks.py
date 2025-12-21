"""训练回调函数"""
import torch
import numpy as np
import os
from pathlib import Path

class EarlyStopping:
    """早停回调"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍的轮数
            min_delta: 最小改善
            mode: 'min' 或 'max'（最小化损失或最大化指标）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
            self.compare = lambda x, y: x + min_delta < y
        else:  # mode == 'max'
            self.best_score = float('-inf')
            self.compare = lambda x, y: x > y + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """重置状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

class ModelCheckpoint:
    """模型检查点回调"""
    
    def __init__(self, save_dir='./checkpoints', filename='best_model.pt',
                 monitor='val_loss', mode='min', save_best_only=True):
        """
        Args:
            save_dir: 保存目录
            filename: 文件名
            monitor: 监控的指标
            mode: 'min' 或 'max'
            save_best_only: 是否只保存最好的
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
    
    def __call__(self, model, epoch, metrics, optimizer=None, scheduler=None):
        current_score = metrics.get(self.monitor, None)
        
        if current_score is None:
            print(f"警告: 监控指标 {self.monitor} 不存在")
            return
        
        # 检查是否改善
        if self.mode == 'min':
            is_better = current_score < self.best_score
        else:  # mode == 'max'
            is_better = current_score > self.best_score
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'monitor': self.monitor,
            'current_score': current_score,
            'best_score': self.best_score
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 总是保存最新的
        latest_path = self.save_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳，保存
        if is_better or not self.save_best_only:
            if is_better:
                self.best_score = current_score
                self.best_epoch = epoch
            
            save_path = self.save_dir / self.filename
            torch.save(checkpoint, save_path)
            
            if is_better:
                print(f"模型改善: {self.monitor} = {current_score:.6f} "
                     f"(之前最佳: {self.best_score:.6f})")
                print(f"模型保存到: {save_path}")
        
        return is_better
    
    def load_best_model(self, model, optimizer=None, scheduler=None):
        """加载最佳模型"""
        load_path = self.save_dir / self.filename
        if not load_path.exists():
            print(f"警告: 检查点文件不存在: {load_path}")
            return None
        
        checkpoint = torch.load(load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"加载最佳模型 (epoch {checkpoint['epoch']}, "
             f"{self.monitor} = {checkpoint['current_score']:.6f})")
        
        return checkpoint

class LearningRateScheduler:
    """学习率调度器回调"""
    
    def __init__(self, optimizer, scheduler_type='plateau', 
                 patience=5, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, 
                factor=factor, min_lr=min_lr, verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=patience, gamma=factor
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=patience, eta_min=min_lr
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def __call__(self, metric=None):
        if self.scheduler_type == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_current_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def reset(self):
        """重置调度器"""
        # 对于某些调度器需要特殊处理
        if hasattr(self.scheduler, 'base_lrs'):
            self.scheduler.base_lrs = [base_lr for base_lr in self.scheduler.base_lrs]