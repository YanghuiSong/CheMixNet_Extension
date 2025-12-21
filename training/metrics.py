"""模型评估指标计算"""
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def calculate_regression_metrics(y_true, y_pred):
    """计算回归任务的评估指标"""
    # 确保输入是numpy数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 基本指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE指标 (Mean Absolute Percentage Error)
    # 避免除以零的情况
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')  # 如果所有真实值都是0，则MAPE未定义
    
    # R² Score
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Pearson相关系数
    try:
        pearson_r, _ = pearsonr(y_true, y_pred)
    except:
        pearson_r = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'pearson_r': pearson_r
    }

def calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算分类任务的评估指标"""
    # 确保输入是numpy数组
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # 转换为二进制预测
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # AUC Score
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.0
    
    # 准确率
    accuracy = np.mean(y_true == y_pred)
    
    # 精确率、召回率、F1 Score
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def format_metrics(metrics_dict, task_type='regression'):
    """格式化指标字典为字符串"""
    formatted_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            if abs(value) > 1e-3 or value == 0:
                formatted_metrics[key] = f"{value:.4f}"
            else:
                formatted_metrics[key] = f"{value:.2e}"
        else:
            formatted_metrics[key] = str(value)
    return formatted_metrics