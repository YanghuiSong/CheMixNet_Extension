#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态增强实验
测试集成图神经网络、注意力机制等高级特性的增强版CheMixNet
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset, create_dataloaders
from models.multimodal import EnhancedCheMixNet
from training.trainer import AdvancedTrainer
from training.metrics import calculate_regression_metrics, calculate_classification_metrics
from config.paths import DATA_PATHS
from evaluation.visualizer import ResultVisualizer
from evaluation.interpretation import ModelInterpreter

# 数据集信息
DATASET_INFO = {
    'bace': {'task': 'classification', 'target_col': 'Class'},
    'BBBP': {'task': 'classification', 'target_col': 'p_np'},
    'esol': {'task': 'regression', 'target_col': 'measured log solubility in mols per litre'},
    'HIV': {'task': 'classification', 'target_col': 'HIV_active'},
    'lipophilicity': {'task': 'regression', 'target_col': 'exp'}
}


def run_enhanced_chemixnet_experiment(dataset_name, use_graph=False, use_attention=True, config=None):
    """运行增强版CheMixNet实验"""
    graph_str = "with_graph" if use_graph else "no_graph"
    attention_str = "with_attention" if use_attention else "no_attention"
    print(f"运行增强版CheMixNet ({graph_str}, {attention_str}) - {dataset_name}")
    print("=" * 60)
    
    # 确保配置存在
    if config is None:
        config = {}
    
    # 设置默认配置
    default_config = {
        'data': {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_seed': 42,
            'batch_size': 32,
            'max_smiles_len': 100,
            'num_workers': 0  # 添加缺失的num_workers参数
        },
        'model': {
            'base': {
                'hidden_dim': 128,
                'lstm_layers': 2,
                'dropout_rate': 0.3,
                'fusion_dim': 256
            },
            'enhanced': {
                'atom_feature_dim': 5,
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'use_attention': True
            }
        },
        'training': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 50,
            'patience': 10,
            'gradient_clip': 1.0
        }
    }
    
    # 合并配置
    merged_config = default_config.copy()
    for key in config:
        if isinstance(config[key], dict) and key in merged_config:
            merged_config[key].update(config[key])
        else:
            merged_config[key] = config[key]

    # 数据预处理
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 划分数据集
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=merged_config['data']['test_size'],
        val_size=merged_config['data']['val_size'],
        random_seed=merged_config['data']['random_seed'],
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征工程
    featurizer = MolecularFeaturizer(max_len=merged_config['data']['max_smiles_len'])
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    train_dataset = MolecularDataset(
        train_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col'],
        use_graph=use_graph
    )
    val_dataset = MolecularDataset(
        val_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col'],
        use_graph=use_graph
    )
    test_dataset = MolecularDataset(
        test_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col'],
        use_graph=use_graph
    )
    
    # 创建数据加载器
    from data.splits import create_dataloaders
    
    # 使用create_dataloaders函数创建数据加载器
    loaders = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=merged_config['data']['batch_size'],
        num_workers=merged_config['data']['num_workers']
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # 获取词汇表大小
    vocab_size = len(featurizer.char_dict)
    fp_dim = 167  # MACCS指纹维度
    
    # 创建模型
    model = EnhancedCheMixNet(
        smiles_vocab_size=vocab_size,
        smiles_max_len=merged_config['data']['max_smiles_len'],
        maccs_dim=fp_dim,
        atom_feature_dim=merged_config['model']['enhanced']['atom_feature_dim'],
        hidden_dims=merged_config['model']['enhanced']['hidden_dims'],
        output_dim=1,
        dropout_rate=merged_config['model']['enhanced']['dropout_rate'],
        use_attention=use_attention
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': merged_config['training']['learning_rate'],
        'weight_decay': merged_config['training']['weight_decay'],
        'task_type': task_info['task'],
        'patience': merged_config['training']['patience'],
        'epochs': merged_config['training']['epochs']
    }
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=merged_config['training']['epochs'])
    
    # 在测试集上评估
    try:
        test_loss, test_metrics = trainer.validate(test_loader)
        print(f"测试集结果: {test_metrics}")
    except Exception as e:
        print(f"测试集评估失败: {e}")
        test_loss = float('inf')
        test_metrics = {}
    
    # 绘制训练历史
    model_name_suffix = f"{graph_str}_{attention_str}"
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_multimodal_{model_name_suffix}")
    try:
        visualizer.plot_training_history(
            trainer.history,
            title=f'Enhanced Multimodal Model ({model_name_suffix}) - {dataset_name}',
            save_name='training_history.png'
        )
    except Exception as e:
        print(f"绘制训练历史失败: {e}")
    
    # 模型解释（新增功能）
    try:
        interpreter = ModelInterpreter(model, featurizer, device=trainer.device)
        if len(test_dataset) > 0:
            sample_data = test_dataset[0]
            if 'smiles' in sample_data and 'fingerprint' in sample_data:
                try:
                    # 使用适配后的特征重要性计算方法
                    importance_scores, feature_names = interpreter.compute_feature_importance(
                        sample_data['smiles'].unsqueeze(0).to(trainer.device), 
                        sample_data['fingerprint'].unsqueeze(0).to(trainer.device),
                        sample_data.get('graph_data')
                    )
                    if importance_scores is not None and feature_names is not None:
                        try:
                            visualizer.plot_feature_importance(
                                importance_scores, 
                                feature_names,
                                title=f'Multimodal Feature Importance - {dataset_name}',
                                save_name='feature_importance.png'
                            )
                        except Exception as e:
                            print(f"绘制特征重要性失败: {e}")
                    else:
                        print("特征重要性计算失败或返回了无效结果")
                
                except Exception as e:
                    print(f"特征重要性计算失败: {e}")
                
                # 如果使用注意力机制，获取注意力权重
                if use_attention and hasattr(model, 'get_attention_weights'):
                    try:
                        attn_weights = model.get_attention_weights(
                            sample_data['smiles'].unsqueeze(0).to(trainer.device),
                            sample_data['fingerprint'].unsqueeze(0).to(trainer.device),
                            sample_data.get('graph_data')
                        )
                        
                        # 可视化注意力权重
                        try:
                            visualizer.plot_attention_weights(
                                attn_weights,
                                title=f'Attention Weights - {dataset_name}',
                                save_name='attention_weights.png'
                            )
                        except Exception as e:
                            print(f"绘制注意力权重失败: {e}")
                    except Exception as e:
                        print(f"获取注意力权重失败: {e}")
    except Exception as e:
        print(f"模型解释阶段出现问题: {e}")
    
    # 保存结果
    results = {
        'dataset': dataset_name,
        'use_graph': use_graph,
        'use_attention': use_attention,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'model_name_suffix': model_name_suffix
    }
    
    return results


def run_comprehensive_multimodal_experiments(config):
    """运行全面的多模态实验"""
    print("运行全面的多模态实验")
    print("=" * 80)
    
    # 定义实验配置
    experiments = [
        {'use_graph': False, 'use_attention': False},
        {'use_graph': False, 'use_attention': True},
        {'use_graph': True, 'use_attention': False},
        {'use_graph': True, 'use_attention': True}
    ]
    
    # 定义数据集
    if 'datasets' in config:
        datasets = config['datasets']
    else:
        # 默认数据集
        datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
    
    # 如果配置中有dataset_info，使用它来确定要运行的数据集
    if 'dataset_info' in config:
        datasets = list(config['dataset_info'].keys())
    
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        for exp_config in experiments:
            try:
                result = run_enhanced_chemixnet_experiment(
                    dataset_name, 
                    use_graph=exp_config['use_graph'],
                    use_attention=exp_config['use_attention'],
                    config=config
                )
                dataset_results[result['model_name_suffix']] = result
            except Exception as e:
                print(f"实验配置 {exp_config} 运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
                # 添加空结果以保持一致性
                empty_result = {
                    'dataset': dataset_name,
                    'use_graph': exp_config['use_graph'],
                    'use_attention': exp_config['use_attention'],
                    'test_loss': float('inf'),
                    'test_metrics': {},
                    'model_name_suffix': f"{'with_graph' if exp_config['use_graph'] else 'no_graph'}_{'with_attention' if exp_config['use_attention'] else 'no_attention'}"
                }
                dataset_results[empty_result['model_name_suffix']] = empty_result
        
        all_results[dataset_name] = dataset_results
        # 生成单个数据集的比较报告
        generate_single_dataset_report(dataset_name, dataset_results, config)
    
    # 生成总体报告
    generate_overall_report(all_results, config)
    
    return all_results


def generate_single_dataset_report(dataset_name, results, config):
    """生成单个数据集的实验报告"""
    print(f"\n生成 {dataset_name} 数据集报告...")
    
    # 确保results是列表格式
    if isinstance(results, dict):
        results_list = list(results.values())
    else:
        results_list = results
    
    # 创建结果DataFrame
    results_data = []
    for r in results_list:
        row = {
            'model': r['model_name_suffix'],
            'use_graph': r['use_graph'],
            'use_attention': r['use_attention'],
            'test_loss': r['test_loss']
        }
        # 展开metrics字典
        if isinstance(r['test_metrics'], dict):
            row.update(r['test_metrics'])
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # 保存详细结果
    save_path = f'./results/{dataset_name}_multimodal_experiment_results.csv'
    results_df.to_csv(save_path, index=False)
    print(f"详细结果保存到: {save_path}")
    
    # 生成可视化比较
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_multimodal_comparison")
    
    # 确定主要评估指标
    task_info = config['dataset_info'][dataset_name]
    primary_metric = 'auc' if task_info['task'] == 'classification' else 'r2'
    
    # 检查数据中是否有这个指标
    available_metrics = [col for col in results_df.columns if col not in ['model', 'use_graph', 'use_attention', 'test_loss']]
    if primary_metric not in available_metrics and available_metrics:
        primary_metric = available_metrics[0]
    
    if primary_metric in results_df.columns:
        # 绘制模型比较图
        fig = visualizer.plot_model_comparison(
            results_df,
            metric=primary_metric,
            title=f'Multimodal Models Comparison - {dataset_name}',
            save_name='model_comparison.png'
        )
        
        # 绘制热力图
        visualizer.plot_comparison_heatmap(
            results_df,
            title=f'Multimodal Performance Heatmap - {dataset_name}',
            save_name='performance_heatmap.png'
        )


def generate_overall_report(all_results, config):
    """生成整体实验报告"""
    print("\n生成整体实验报告...")
    
    # 创建结果汇总DataFrame
    summary_data = []
    for dataset, results in all_results.items():
        for exp_name, metrics in results.items():
            if isinstance(metrics, dict) and 'test_loss' in metrics:
                row = {
                    'dataset': dataset,
                    'model': exp_name,
                    'use_graph': 'with_graph' in exp_name,
                    'use_attention': 'with_attention' in exp_name,
                    'test_loss': metrics['test_loss']
                }
                # 添加其他指标
                for key, value in metrics.items():
                    if key != 'test_loss':
                        row[key] = value
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存CSV
    csv_path = './results/multimodal_experiment_results.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"整体实验结果保存到: {csv_path}")
    
    # 可视化结果
    try:
        visualizer = ResultVisualizer(save_dir='./results/figures/multimodal_comparison')
        
        # 绘制跨数据集比较图
        if len(summary_df) > 0:
            try:
                visualizer.plot_cross_dataset_comparison(
                    summary_df,
                    metric='test_loss',
                    title='Cross-Dataset Performance Comparison',
                    save_name='cross_dataset_comparison.png'
                )
            except Exception as e:
                print(f"绘制跨数据集比较图失败: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 加载配置
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置中的数据集信息
    config['dataset_info'] = DATASET_INFO
    
    # 创建结果目录
    os.makedirs('./results/figures', exist_ok=True)
    
    # 运行实验
    all_results = run_comprehensive_multimodal_experiments(config)
    
    print("\n多模态实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()