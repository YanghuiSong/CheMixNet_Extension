#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线模型运行脚本
包括MLP、CNN、RNN三种基线模型
在所有数据集上运行实验
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer  # 导入MolecularFeaturizer类
from models.base_models import MLPBaseline, CNNBaseline, RNNBaseline
from training.trainer import AdvancedTrainer
from training.metrics import calculate_regression_metrics, calculate_classification_metrics
from config.paths import DATA_PATHS
from evaluation.visualizer import ResultVisualizer


class FingerprintDataset(Dataset):
    """指纹数据集类"""
    
    def __init__(self, fingerprints, targets):
        """
        初始化数据集
        
        Args:
            fingerprints: 指纹特征数组
            targets: 目标值数组
        """
        self.fingerprints = torch.FloatTensor(fingerprints)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        # 修复：返回与训练器期望匹配的键名
        return {
            'features': self.fingerprints[idx],  # 使用'features'而不是'fingerprint'
            'target': self.targets[idx]
        }


def run_mlp_baseline(dataset_name, config):
    """运行MLP基线模型"""
    print(f"运行MLP基线模型 - {dataset_name}")
    print("=" * 60)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, input_dim = _get_data_loaders(dataset_name, config)
    
    # 初始化模型
    hidden_dims = config['mlp_hidden_dims']
    output_dim = 1
    dropout = config.get('dropout', 0.1)
    
    # 修复：确保使用正确的输入维度（MACCS指纹应该是166维）
    model = MLPBaseline(input_dim, hidden_dims, output_dim, dropout)
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'task_type': config['task_type'],
        'patience': config['patience'],
        'force_cuda': True,  # 强制使用CUDA
        'save_checkpoints': False  # 不再保存检查点
    }
    
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=config['epochs'])
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_mlp")
    fig = visualizer.plot_training_history(
        trainer.history,
        title=f'MLP Baseline Training - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    return test_metrics


def run_cnn_baseline(dataset_name, config):
    """运行CNN基线模型"""
    print(f"运行CNN基线模型 - {dataset_name}")
    print("=" * 60)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, char_dict = _get_smiles_data_loaders(dataset_name, config)
    
    # 初始化模型
    vocab_size = len(char_dict)
    # 修复：根据CNNBaseline的实际构造函数参数调整
    model = CNNBaseline(
        vocab_size=vocab_size, 
        max_len=config['max_smiles_len']
    )
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'task_type': config['task_type'],
        'patience': config['patience'],
        'force_cuda': True,
        'save_checkpoints': False
    }
    
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=config['epochs'])
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_cnn")
    fig = visualizer.plot_training_history(
        trainer.history,
        title=f'CNN Baseline Training - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    return test_metrics


def run_rnn_baseline(dataset_name, config):
    """运行RNN基线模型"""
    print(f"运行RNN基线模型 - {dataset_name}")
    print("=" * 60)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, char_dict = _get_smiles_data_loaders(dataset_name, config)
    
    # 初始化模型
    vocab_size = len(char_dict)
    # 修复：根据RNNBaseline的实际构造函数参数调整
    model = RNNBaseline(
        vocab_size=vocab_size,
        max_len=config['max_smiles_len'],
        hidden_dim=config['rnn_hidden_dim'],
        num_layers=config['rnn_num_layers']
    )
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'task_type': config['task_type'],
        'patience': config['patience'],
        'force_cuda': True,
        'save_checkpoints': False
    }
    
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=config['epochs'])
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_rnn")
    fig = visualizer.plot_training_history(
        trainer.history,
        title=f'RNN Baseline Training - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    return test_metrics


def _get_data_loaders(dataset_name, config):
    """获取指纹数据加载器"""
    # 数据预处理
    # 修复：传递数据集名称而不是整个配置字典
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    df = preprocessor.add_molecular_features(df)
    
    # 特征提取
    # 修复：正确提取MACCS指纹特征
    features = np.array(df['maccs_fingerprint'].tolist())
    targets = df[config['target_column']].values
    
    # 数据集划分
    train_features, val_features, test_features, train_targets, val_targets, test_targets = \
        preprocessor.split_features(features, targets, task_type=config['task_type'])
    
    # 创建数据集
    train_dataset = FingerprintDataset(train_features, train_targets)
    val_dataset = FingerprintDataset(val_features, val_targets)
    test_dataset = FingerprintDataset(test_features, test_targets)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    input_dim = features.shape[1]
    return train_loader, val_loader, test_loader, input_dim


def _get_smiles_data_loaders(dataset_name, config):
    """获取SMILES数据加载器"""
    # 数据预处理
    # 修复：传递数据集名称而不是整个配置字典
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 字符字典构建
    featurizer = MolecularFeaturizer(max_len=config['max_smiles_len'])
    featurizer.build_char_vocab(df['smiles'].tolist())
    char_dict = featurizer.char_dict
    
    # 特征提取
    features = featurizer.encode_smiles_batch(df['smiles'].tolist())
    targets = df[config['target_column']].values
    
    # 数据集划分
    train_features, val_features, test_features, train_targets, val_targets, test_targets = \
        preprocessor.split_features(features, targets, task_type=config['task_type'])
    
    # 创建数据集
    train_dataset = SMILESDataset(train_features, train_targets)
    val_dataset = SMILESDataset(val_features, val_targets)
    test_dataset = SMILESDataset(test_features, test_targets)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader, char_dict


class SMILESDataset(Dataset):
    """SMILES数据集类"""
    
    def __init__(self, smiles, targets):
        """
        初始化数据集
        
        Args:
            smiles: SMILES编码数组
            targets: 目标值数组
        """
        self.smiles = torch.LongTensor(smiles)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'target': self.targets[idx]
        }


def evaluate_model(model, data_loader, task_type='regression'):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            if 'features' in batch:
                inputs = batch['features']
                targets = batch['target']
            elif 'smiles' in batch:
                inputs = batch['smiles']
                targets = batch['target']
            else:
                continue
                
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # 根据任务类型计算指标
    if task_type == 'regression':
        metrics = calculate_regression_metrics(all_targets, all_predictions)
    else:  # classification
        metrics = calculate_classification_metrics(all_targets, all_predictions)
    
    return metrics


def compare_baseline_models(dataset_name, config):
    """比较所有基线模型"""
    print(f"比较基线模型 - {dataset_name}")
    print("=" * 60)
    
    # 运行所有基线模型
    results = {}
    
    # MLP基线
    try:
        mlp_results = run_mlp_baseline(dataset_name, config)
        results['MLP'] = mlp_results
    except Exception as e:
        print(f"MLP基线运行失败: {e}")
        results['MLP'] = {}
    
    # CNN基线
    try:
        cnn_results = run_cnn_baseline(dataset_name, config)
        results['CNN'] = cnn_results
    except Exception as e:
        print(f"CNN基线运行失败: {e}")
        results['CNN'] = {}
    
    # RNN基线
    try:
        rnn_results = run_rnn_baseline(dataset_name, config)
        results['RNN'] = rnn_results
    except Exception as e:
        print(f"RNN基线运行失败: {e}")
        results['RNN'] = {}
    
    # 结果汇总和可视化
    comparison_data = []
    metric_keys = set()
    
    for model_name, metrics in results.items():
        # 只有当模型有结果时才添加到比较数据中
        # 修复：即使是空字典也要添加，以确保所有模型都在比较中显示
        row = {'model': model_name}
        for k, v in metrics.items():
            row[k] = v
            metric_keys.add(k)
        comparison_data.append(row)
    
    # 只有当有数据时才进行可视化
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # 确定主要评估指标
        primary_metric = 'auc' if config['task_type'] == 'classification' else 'r2'
        if primary_metric not in metric_keys:
            primary_metric = list(metric_keys)[0] if metric_keys else 'loss'
        
        # 绘制模型比较图
        visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_comparison")
        fig = visualizer.plot_model_comparison(
            comparison_df,
            metric=primary_metric,
            title=f'Model Comparison - {dataset_name}',
            save_name='model_comparison.png',
            experiment_type='baseline'
        )
        
        # 保存结果到CSV
        comparison_df.to_csv(f"./results/{dataset_name}_baseline_comparison.csv", index=False)
        
        print("\n模型比较结果:")
        print(comparison_df)
    else:
        print("所有基线模型都运行失败，没有结果可以比较。")
    
    # 返回格式化的结果，用于跨数据集比较
    # 修复：始终返回比较数据，而不仅仅是成功的模型
    return comparison_data


def run_all_datasets_baseline_comparison():
    """在所有数据集上运行基线模型并生成综合比较"""
    # 示例配置
    config = {
        'target_column': 'target',
        'task_type': 'regression',  # 或 'classification'
        'batch_size': 32,
        'epochs': 100,  # 增加训练轮数
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,  # 增加耐心值
        'dropout': 0.1,
        'max_smiles_len': 100,
        
        # MLP配置
        'mlp_hidden_dims': [512, 256, 128],
        
        # CNN配置
        'cnn_embed_dim': 128,
        'cnn_kernel_sizes': [3, 5, 7],
        'cnn_num_filters': [64, 128, 128],
        'cnn_hidden_dims': [256, 128],
        
        # RNN配置
        'rnn_embed_dim': 128,
        'rnn_hidden_dim': 128,
        'rnn_num_layers': 2
    }
    
    # 数据集配置
    dataset_config = {
        'bace': {'task_type': 'classification', 'target_column': 'Class'},
        'BBBP': {'task_type': 'classification', 'target_column': 'p_np'},
        'esol': {'task_type': 'regression', 'target_column': 'measured log solubility in mols per litre'},
        'HIV': {'task_type': 'classification', 'target_column': 'HIV_active'},
        'lipophilicity': {'task_type': 'regression', 'target_column': 'exp'}
    }
    
    # 在所有五个数据集上运行
    datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
    
    # 收集所有结果
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*80}")
        
        # 更新配置以包含特定数据集的信息
        dataset_specific_config = config.copy()
        dataset_specific_config.update(dataset_config[dataset])
        
        # 运行基线模型比较
        results = compare_baseline_models(dataset, dataset_specific_config)
        all_results[dataset] = results
    
    # 生成跨数据集比较图表
    print(f"\n{'='*80}")
    print("生成跨数据集比较图表")
    print(f"{'='*80}")
    
    # 准备跨数据集比较数据
    cross_dataset_data = {}
    for dataset in datasets:
        # 修复：compare_baseline_models返回的是列表而不是字典
        cross_dataset_data[dataset] = all_results[dataset]
    
    # 为每个数据集生成综合比较图表
    for dataset in datasets:
        if cross_dataset_data[dataset]:  # 只有当有数据时才生成图表
            comparison_df = pd.DataFrame(cross_dataset_data[dataset])
            
            # 确定主要评估指标
            task_type = dataset_config[dataset]['task_type']
            primary_metric = 'auc' if task_type == 'classification' else 'r2'
            
            # 如果primary_metric不在列中，选择第一个可用的指标
            if primary_metric not in comparison_df.columns:
                available_metrics = [col for col in comparison_df.columns if col not in ['model']]
                primary_metric = available_metrics[0] if available_metrics else 'loss'
            
            # 绘制模型比较图
            visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset}_baseline_summary")
            fig = visualizer.plot_model_comparison(
                comparison_df,
                metric=primary_metric,
                title=f'Baseline Models Comparison - {dataset.upper()}',
                save_name='baseline_model_comparison.png'
            )
            
            # 绘制指标相关性分析（如果有多个指标）
            if len(comparison_df.columns) > 2:
                # 移除 'model' 列进行相关性分析
                numeric_cols = [col for col in comparison_df.columns if col != 'model']
                if len(numeric_cols) > 1:
                    corr_data = comparison_df[numeric_cols]
                    correlation_matrix = corr_data.corr()
                    
                    # 绘制热力图
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                                square=True, linewidths=0.5)
                    plt.title(f'Metric Correlation Analysis - {dataset.upper()}')
                    
                    # 保存图像
                    save_path = visualizer._get_unique_save_path('metric_correlation_analysis.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"指标相关性分析图已保存至: {save_path}")

    # 生成跨数据集比较图表
    formatted_cross_dataset_data = {}
    for dataset in datasets:
        formatted_cross_dataset_data[dataset] = cross_dataset_data[dataset]
    
    # 绘制跨数据集比较图
    visualizer = ResultVisualizer(save_dir="./results/figures/cross_dataset_comparison")
    fig_cross = visualizer.plot_cross_dataset_comparison(
        formatted_cross_dataset_data,
        metric='auc',  # 默认使用AUC，对于回归任务会自动适配
        title='Cross-Dataset Baseline Model Performance',
        save_name='cross_dataset_performance.png'
    )
    
    print("\n所有数据集的基线模型实验已完成！")
    print("结果已保存到 ./results/ 目录中")


if __name__ == "__main__":
    # 运行所有数据集的基线模型比较
    run_all_datasets_baseline_comparison()
