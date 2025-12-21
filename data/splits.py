"""数据集划分工具"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Subset

def split_dataset(df, test_size=0.2, val_size=0.1, random_seed=42, 
                  stratify_col=None):
    """
    划分数据集为训练集、验证集和测试集
    
    Args:
        df: 数据DataFrame
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_seed: 随机种子
        stratify_col: 用于分层抽样的列（用于分类任务）
    
    Returns:
        train_df, val_df, test_df
    """
    # 首先划分出测试集
    if stratify_col and stratify_col in df.columns:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed,
            stratify=df[stratify_col]
        )
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed
        )
    
    # 从训练验证集中划分验证集
    val_relative_size = val_size / (1 - test_size)
    if stratify_col and stratify_col in train_val_df.columns:
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_relative_size, 
            random_state=random_seed, stratify=train_val_df[stratify_col]
        )
    else:
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_relative_size, 
            random_state=random_seed
        )
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} 个样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 个样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 个样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def kfold_split(df, n_splits=5, random_seed=42, stratify_col=None):
    """
    K折交叉验证划分
    
    Args:
        df: 数据DataFrame
        n_splits: 折数
        random_seed: 随机种子
        stratify_col: 用于分层抽样的列
    
    Yields:
        每次迭代返回 (train_indices, val_indices)
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    if stratify_col and stratify_col in df.columns:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                           random_state=random_seed)
        splits = kf.split(df, df[stratify_col])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = kf.split(df)
    
    for train_indices, val_indices in splits:
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        yield train_df, val_df

def create_dataloaders(train_dataset, val_dataset, test_dataset=None, 
                      batch_size=32, num_workers=0):
    """创建PyTorch DataLoader"""
    from torch.utils.data import DataLoader
    from .dataset import collate_fn
    
    print(f"Creating dataloaders with custom collate_fn")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    loaders = {'train': train_loader, 'val': val_loader}
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
        )
        loaders['test'] = test_loader
    
    return loaders
