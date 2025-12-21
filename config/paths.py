"""数据路径配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_PATHS = {
    'bace': BASE_DIR / "data/bace.csv",
    'BBBP': BASE_DIR / "data/BBBP.csv", 
    'esol': BASE_DIR / "data/esol.csv",  # 修正文件名
    'HIV': BASE_DIR / "data/HIV.csv",
    'lipophilicity': BASE_DIR / "data/lipophilicity.csv"  # 修正文件名
}

# 数据集元信息
DATASET_INFO = {
    'bace': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'Class'},
    'BBBP': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'p_np'},
    'esol': {'task': 'regression', 'metric': 'rmse', 'target_col': 'measured log solubility in mols per litre'},
    'HIV': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'HIV_active'},
    'lipophilicity': {'task': 'regression', 'metric': 'rmse', 'target_col': 'exp'}
}