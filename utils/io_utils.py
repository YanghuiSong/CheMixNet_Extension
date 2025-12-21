"""IO工具"""
import pickle
import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import gzip
import shutil

def save_pickle(obj, filepath, compress=False):
    """保存对象到pickle文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        with gzip.open(filepath.with_suffix('.pkl.gz'), 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath, compressed=False):
    """从pickle文件加载对象"""
    filepath = Path(filepath)
    
    if compressed or filepath.suffix == '.gz':
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def save_json(obj, filepath, indent=2):
    """保存对象到JSON文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json(filepath):
    """从JSON文件加载对象"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(obj, filepath):
    """保存对象到YAML文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)

def load_yaml(filepath):
    """从YAML文件加载对象"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(state, filepath, is_best=False):
    """保存模型检查点"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        shutil.copyfile(filepath, best_path)

def load_checkpoint(filepath, model=None, optimizer=None, device='cpu'):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def save_array(array, filepath, format='npy'):
    """保存数组"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npy':
        np.save(filepath, array)
    elif format == 'npz':
        np.savez_compressed(filepath, array=array)
    elif format == 'csv':
        pd.DataFrame(array).to_csv(filepath, index=False)
    else:
        raise ValueError(f"不支持的格式: {format}")

def load_array(filepath, format=None):
    """加载数组"""
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:]  # 去掉点号
    
    if format == 'npy':
        return np.load(filepath)
    elif format == 'npz':
        return np.load(filepath)['array']
    elif format == 'csv':
        return pd.read_csv(filepath).values
    else:
        raise ValueError(f"不支持的格式: {format}")

def save_results(results, filepath, format='csv'):
    """保存结果"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        if isinstance(results, dict):
            pd.DataFrame([results]).to_csv(filepath, index=False)
        elif isinstance(results, list):
            pd.DataFrame(results).to_csv(filepath, index=False)
        elif isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
    elif format == 'json':
        save_json(results, filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")

def load_results(filepath, format=None):
    """加载结果"""
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:]  # 去掉点号
    
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'json':
        return load_json(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")