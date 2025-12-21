#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一实验运行脚本
运行所有实验类型：基线、CheMixNet、多模态、消融实验
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.run_baselines import run_all_datasets_baseline_comparison
from experiments.run_chemixnet import compare_chemixnet_models
from experiments.run_multimodal import run_comprehensive_multimodal_experiments
from experiments.run_ablation import compare_ablation_studies
from evaluation.molecule_visualization import main as visualization_main

# 默认配置
DEFAULT_CONFIG = {
    'data': {
        'test_size': 0.2,
        'val_size': 0.1,
        'random_seed': 42,
        'batch_size': 32,
        'max_smiles_len': 100,
        'num_workers': 0  # 添加缺失的num_workers参数
    },
    'training': {
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        'gradient_clip': 1.0
    },
    'model': {
        'chemixnet': {
            'cnn_channels': [64, 128, 256],
            'fc_dims': [512, 256, 128],
            'dropout_rate': 0.3,
            'hidden_dim': 128,
            'lstm_layers': 2,
            'lstm_hidden': 128
        },
        'multimodal': {
            'gnn_hidden_dim': 128,
            'attention_heads': 8,
            'fusion_dim': 256,
            'dropout_rate': 0.3
        }
    }
}

# 数据集配置
DATASET_INFO = {
    'bace': {'task': 'classification', 'target_col': 'Class'},
    'BBBP': {'task': 'classification', 'target_col': 'p_np'},
    'esol': {'task': 'regression', 'target_col': 'measured log solubility in mols per litre'},
    'HIV': {'task': 'classification', 'target_col': 'HIV_active'},
    'lipophilicity': {'task': 'regression', 'target_col': 'exp'}
}

def run_baseline_experiments(config):
    """运行基线实验"""
    print("开始运行基线实验...")
    print("=" * 60)
    
    try:
        results = run_all_datasets_baseline_comparison()
        print("基线实验完成!")
        return results
    except Exception as e:
        print(f"基线实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_chemixnet_experiments(config):
    """运行CheMixNet实验"""
    print("开始运行CheMixNet实验...")
    print("=" * 60)
    
    try:
        # 为每个数据集运行CheMixNet实验
        datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
        results = []
        
        for dataset in datasets:
            if dataset in DATASET_INFO:
                dataset_config = config.copy()
                dataset_config['dataset_info'] = {dataset: DATASET_INFO[dataset]}
                
                try:
                    result = compare_chemixnet_models(dataset, dataset_config)
                    results.append(result)
                    print(f"{dataset} CheMixNet实验完成!")
                except Exception as e:
                    print(f"{dataset} CheMixNet实验失败: {e}")
                    
        print("CheMixNet实验完成!")
        return results
    except Exception as e:
        print(f"CheMixNet实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_multimodal_experiments(config):
    """运行多模态实验"""
    print("开始运行多模态实验...")
    print("=" * 60)
    
    try:
        # 更新配置
        config['dataset_info'] = DATASET_INFO
        results = run_comprehensive_multimodal_experiments(config)
        print("多模态实验完成!")
        return results
    except Exception as e:
        print(f"多模态实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_ablation_experiments(config):
    """运行消融实验"""
    print("开始运行消融实验...")
    print("=" * 60)
    
    try:
        # 为每个数据集运行消融实验
        datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
        all_results = {}
        
        for dataset in datasets:
            if dataset in DATASET_INFO:
                dataset_config = config.copy()
                dataset_config['dataset_info'] = {dataset: DATASET_INFO[dataset]}
                
                try:
                    results = compare_ablation_studies(dataset, dataset_config)
                    all_results[dataset] = results
                    print(f"{dataset} 消融实验完成!")
                except Exception as e:
                    print(f"{dataset} 消融实验失败: {e}")
                    
        print("消融实验完成!")
        return all_results
    except Exception as e:
        print(f"消融实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行所有实验')
    parser.add_argument('--experiment', choices=['baseline', 'chemixnet', 'multimodal', 'ablation', 'all'], 
                       default='all', help='选择要运行的实验类型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/figures', exist_ok=True)
    
    print("开始运行实验...")
    print("=" * 80)
    
    # 根据选择运行相应实验
    if args.experiment == 'baseline' or args.experiment == 'all':
        run_baseline_experiments(config)
        
    if args.experiment == 'chemixnet' or args.experiment == 'all':
        run_chemixnet_experiments(config)
        
    if args.experiment == 'multimodal' or args.experiment == 'all':
        run_multimodal_experiments(config)
        
    if args.experiment == 'ablation' or args.experiment == 'all':
        run_ablation_experiments(config)
    
    print("\n所有选定实验已完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()