#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合实验脚本
运行所有基线模型和CheMixNet模型的系统性对比实验
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from config.paths import DATA_PATHS
from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset, create_dataloaders
from models.base_models import MLPBaseline, CNNBaseline, RNNBaseline
from models.chemixnet import CheMixNetCNN, CheMixNetRNN
from training.trainer import AdvancedTrainer
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


def run_baseline_experiment(dataset_name, model_name, config):
    """运行基线模型实验"""
    print(f"运行基线模型 ({model_name.upper()}) - {dataset_name}")
    print("=" * 60)
    
    # 数据预处理
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 划分数据集
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_seed=config['data']['random_seed'],
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征工程
    featurizer = MolecularFeaturizer(max_len=config['data']['max_smiles_len'])
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    train_dataset = MolecularDataset(
        train_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    val_dataset = MolecularDataset(
        val_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    test_dataset = MolecularDataset(
        test_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 获取词汇表大小
    vocab_size = len(featurizer.char_dict)
    
    # 创建模型
    if model_name == 'mlp':
        # 对于MLP，我们使用指纹特征
        model = MLPBaseline(
            input_dim=167,  # MACCS指纹维度
            hidden_dims=config['model']['mlp']['hidden_dims'],
            output_dim=1,
            dropout_rate=config['model']['mlp']['dropout_rate']
        )
    elif model_name == 'cnn':
        model = CNNBaseline(
            vocab_size=vocab_size,
            max_len=config['data']['max_smiles_len'],
            embedding_dim=config['model']['cnn']['embedding_dim'],
            hidden_dims=config['model']['cnn']['hidden_dims'],
            output_dim=1,
            dropout_rate=config['model']['cnn']['dropout_rate']
        )
    elif model_name == 'rnn':
        model = RNNBaseline(
            vocab_size=vocab_size,
            max_len=config['data']['max_smiles_len'],
            embedding_dim=config['model']['rnn']['embedding_dim'],
            hidden_dim=config['model']['rnn']['hidden_dim'],
            num_layers=config['model']['rnn']['num_layers'],
            output_dim=1,
            dropout_rate=config['model']['rnn']['dropout_rate'],
            bidirectional=config['model']['rnn']['bidirectional']
        )
    else:
        raise ValueError(f"不支持的基线模型: {model_name}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'task_type': task_info['task'],
        'patience': config['training']['patience'],
        'epochs': config['training']['epochs']
    }
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_baseline_{model_name}")
    visualizer.plot_training_history(
        trainer.history,
        title=f'Baseline ({model_name.upper()}) - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 收集预测结果用于后续分析
    all_predictions = []
    all_targets = []
    all_smiles = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            smiles = batch['smiles']
            fp = batch['fingerprint']
            targets = batch['target']
            
            if model_name == 'mlp':
                inputs = fp
            else:
                inputs = smiles
                
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 收集SMILES用于可视化
            batch_smiles = [featurizer.decode_smiles(smile.numpy()) for smile in smiles]
            all_smiles.extend(batch_smiles)
    
    # 模型解释（新增功能）
    interpreter = ModelInterpreter(model, featurizer, device=trainer.device)
    if len(test_dataset) > 0:
        sample_data = test_dataset[0]
        try:
            if model_name == 'mlp':
                # MLP只需要指纹输入
                importance = interpreter.compute_feature_importance(
                    None,  # smiles_input
                    sample_data['fingerprint'].unsqueeze(0).to(trainer.device)  # fp_input
                )
            else:
                # 其他模型需要SMILES输入
                importance = interpreter.compute_feature_importance(
                    sample_data['smiles'].unsqueeze(0).to(trainer.device), 
                    sample_data['fingerprint'].unsqueeze(0).to(trainer.device)
                )
            
            visualizer.plot_feature_importance(
                importance, 
                title=f'{model_name.upper()} Feature Importance - {dataset_name}',
                save_name='feature_importance.png'
            )
        except Exception as e:
            print(f"特征重要性计算失败: {e}")
    
    # 保存结果
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_val_loss': trainer.best_val_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'smiles': all_smiles
    }
    
    return results


def run_chemixnet_experiment(dataset_name, model_type, config):
    """运行CheMixNet实验"""
    print(f"运行CheMixNet ({model_type.upper()}) - {dataset_name}")
    print("=" * 60)
    
    # 数据预处理
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 划分数据集
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_seed=config['data']['random_seed'],
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征工程
    featurizer = MolecularFeaturizer(max_len=config['data']['max_smiles_len'])
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    train_dataset = MolecularDataset(
        train_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    val_dataset = MolecularDataset(
        val_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    test_dataset = MolecularDataset(
        test_df, featurizer, task_type=task_info['task'], target_col=task_info['target_col']
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # 获取词汇表大小
    vocab_size = len(featurizer.char_dict)
    
    # 创建模型
    if model_type == 'cnn':
        model = CheMixNetCNN(
            vocab_size=vocab_size,
            max_len=config['data']['max_smiles_len'],
            fp_dim=167,  # MACCS指纹维度
            hidden_dim=config['model']['cnn']['hidden_dim'],
            output_dim=1,
            dropout_rate=config['model']['cnn']['dropout_rate']
        )
    elif model_type == 'rnn':
        model = CheMixNetRNN(
            vocab_size=vocab_size,
            max_len=config['data']['max_smiles_len'],
            fp_dim=167,
            hidden_dim=config['model']['rnn']['hidden_dim'],
            lstm_layers=config['model']['rnn']['lstm_layers'],
            output_dim=1,
            dropout_rate=config['model']['rnn']['dropout_rate']
        )
    else:
        raise ValueError(f"不支持的CheMixNet模型类型: {model_type}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'task_type': task_info['task'],
        'patience': config['training']['patience'],
        'epochs': config['training']['epochs']
    }
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_chemixnet_{model_type}")
    visualizer.plot_training_history(
        trainer.history,
        title=f'CheMixNet ({model_type.upper()}) - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 收集预测结果用于后续分析
    all_predictions = []
    all_targets = []
    all_smiles = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            smiles = batch['smiles']
            fp = batch['fingerprint']
            targets = batch['target']
            
            outputs = model(smiles, fp)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 收集SMILES用于可视化
            batch_smiles = [featurizer.decode_smiles(smile.numpy()) for smile in smiles]
            all_smiles.extend(batch_smiles)
    
    # 模型解释（新增功能）
    interpreter = ModelInterpreter(model, featurizer, device=trainer.device)
    if len(test_dataset) > 0:
        sample_data = test_dataset[0]
        if 'smiles' in sample_data and 'fingerprint' in sample_data:
            try:
                importance_scores, feature_names = interpreter.compute_feature_importance(
                    sample_data['smiles'].unsqueeze(0).to(trainer.device), 
                    sample_data['fingerprint'].unsqueeze(0).to(trainer.device)
                )
                if importance_scores is not None and feature_names is not None:
                    visualizer.plot_feature_importance(
                        importance_scores, 
                        feature_names,
                        title=f'{model_type.upper()} Feature Importance - {dataset_name}',
                        save_name='feature_importance.png'
                    )
                else:
                    print("特征重要性计算返回了无效结果")
            except Exception as e:
                print(f"特征重要性计算失败: {e}")
    
    # 保存结果
    results = {
        'dataset': dataset_name,
        'model': f'chemixnet_{model_type}',
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_val_loss': trainer.best_val_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'smiles': all_smiles
    }
    
    return results


def run_comprehensive_experiments_for_dataset(dataset_name, config):
    """为单个数据集运行综合实验"""
    print(f"为数据集 {dataset_name} 运行综合实验")
    print("=" * 60)
    
    results = []
    
    # 运行基线模型
    baseline_models = ['mlp', 'cnn', 'rnn']
    for model_name in baseline_models:
        try:
            result = run_baseline_experiment(dataset_name, model_name, config)
            results.append(result)
        except Exception as e:
            print(f"基线模型 {model_name} 运行失败: {str(e)}")
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'test_loss': float('inf'),
                'test_metrics': {},
                'best_val_loss': float('inf'),
                'predictions': [],
                'targets': [],
                'smiles': []
            })
    
    # 运行CheMixNet模型
    chemixnet_models = ['cnn', 'rnn']
    for model_type in chemixnet_models:
        try:
            result = run_chemixnet_experiment(dataset_name, model_type, config)
            results.append(result)
        except Exception as e:
            print(f"CheMixNet {model_type} 运行失败: {str(e)}")
            results.append({
                'dataset': dataset_name,
                'model': f'chemixnet_{model_type}',
                'test_loss': float('inf'),
                'test_metrics': {},
                'best_val_loss': float('inf'),
                'predictions': [],
                'targets': [],
                'smiles': []
            })
    
    return results


def generate_comprehensive_report(all_results):
    """生成综合实验报告"""
    print("\n生成综合实验报告...")
    print("=" * 60)
    
    # 创建结果DataFrame
    results_data = []
    for r in all_results:
        row = {
            'dataset': r['dataset'],
            'model': r['model'],
            'test_loss': r['test_loss']
        }
        # 展开metrics字典
        if isinstance(r['test_metrics'], dict):
            row.update(r['test_metrics'])
        results_data.append(row)
        
    results_df = pd.DataFrame(results_data)
    
    # 保存详细结果
    results_df.to_csv('./results/comprehensive_experiment_results.csv', index=False)
    print(f"详细结果保存到: ./results/comprehensive_experiment_results.csv")
    
    # 生成汇总表格
    print("\n实验结果汇总:")
    print("-" * 80)
    
    # 按数据集分组显示
    for dataset in results_df['dataset'].unique():
        print(f"\n数据集: {dataset}")
        print("-" * 40)
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        # 根据任务类型选择主要指标
        task_type = DATASET_INFO[dataset]['task']
        if task_type == 'regression':
            # 对于回归任务，主要看MSE/RMSE（越小越好）
            dataset_results = dataset_results.sort_values('mse' if 'mse' in dataset_results.columns else 'rmse')
            # 添加mape到回归指标中
            metric_cols = ['model', 'test_loss', 'mse', 'rmse', 'mae', 'mape']
        else:
            # 对于分类任务，主要看AUC（越大越好）
            dataset_results = dataset_results.sort_values('auc', ascending=False)
            metric_cols = ['model', 'test_loss', 'auc', 'f1', 'precision', 'recall']
        
        # 只显示存在的列
        display_cols = [col for col in metric_cols if col in dataset_results.columns]
        print(dataset_results[display_cols].to_string(index=False))
    
    # 生成对比图表
    visualizer = ResultVisualizer(save_dir='./results/figures/comprehensive_comparison')
    
    # 每个数据集的模型比较
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        task_type = DATASET_INFO[dataset]['task']
        
        if task_type == 'regression':
            primary_metric = 'rmse' if 'rmse' in dataset_results.columns else 'mse'
        else:
            primary_metric = 'auc'
            
        # 检查数据中是否有这个指标
        available_metrics = [col for col in dataset_results.columns if col not in ['dataset', 'model', 'test_loss']]
        if primary_metric not in available_metrics and available_metrics:
            primary_metric = available_metrics[0]
        
        if primary_metric in dataset_results.columns and not dataset_results.empty:
            visualizer.plot_model_comparison(
                dataset_results,
                metric=primary_metric,
                title=f'Model Comparison - {dataset}',
                save_name=f'model_comparison_{dataset}.png'
            )
    
    # 跨数据集比较
    visualizer.plot_cross_dataset_comparison(
        results_df,
        title='Cross-Dataset Model Performance',
        save_name='cross_dataset_comparison.png'
    )
    
    # 如果存在回归数据集，也创建MAPE的专门比较图
    regression_datasets = [ds for ds in results_df['dataset'].unique() 
                          if DATASET_INFO.get(ds, {}).get('task') == 'regression' and 'mape' in results_df.columns]
    if regression_datasets and not results_df[results_df['dataset'].isin(regression_datasets)].empty:
        visualizer.plot_cross_dataset_comparison(
            results_df[results_df['dataset'].isin(regression_datasets)],
            metric='mape',
            title='Cross-Dataset MAPE Comparison',
            save_name='cross_dataset_mape_comparison.png'
        )


def collect_all_results():
    """收集所有实验结果并生成综合报告"""
    print("收集所有实验结果...")
    print("=" * 60)
    
    # 初始化结果分析器和可视化器
    analyzer = ResultAnalyzer()
    visualizer = ResultVisualizer(save_dir="./results/figures/comprehensive_analysis")
    
    # 收集各实验结果
    experiments = ['baseline', 'chemixnet', 'multimodal', 'ablation']
    all_results = {}
    
    for exp_name in experiments:
        df = analyzer.load_results(exp_name, 'results.csv')
        if df is not None:
            all_results[exp_name] = df
            print(f"加载 {exp_name} 实验结果: {len(df)} 条记录")
        else:
            print(f"{exp_name} 实验结果文件不存在")
    
    if not all_results:
        print("没有找到任何实验结果")
        return
    
    # 跨实验比较
    print("\n生成跨实验比较图表...")
    
    # 定义要比较的指标
    metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'mse', 'mae', 'rmse', 'mape', 'r2']
    
    # 为每个指标生成比较图表
    for metric in metrics:
        try:
            comparison_df = analyzer.compare_experiments(list(all_results.keys()), metric)
            if not comparison_df.empty:
                # 创建比较图表
                plt.figure(figsize=(12, 8))
                
                # 绘制柱状图
                x_pos = np.arange(len(all_results))
                means = comparison_df['mean'].values
                stds = comparison_df['std'].values
                
                plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                       color=plt.cm.Set3(np.linspace(0, 1, len(all_results))))
                plt.xticks(x_pos, list(all_results.keys()), rotation=45)
                plt.ylabel(metric.upper())
                plt.title(f'Comparison of {metric.upper()} Across Experiments')
                plt.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for i, (mean, std) in enumerate(zip(means, stds)):
                    plt.text(i, mean + std + 0.01, f'{mean:.4f}', 
                            ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                save_path = f"./results/figures/comprehensive_analysis/{metric}_comparison.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"{metric}比较图已保存至: {save_path}")
                plt.close()
        except Exception as e:
            print(f"生成{metric}比较图时出错: {e}")
    
    # 生成综合性能雷达图
    print("\n生成综合性能雷达图...")
    try:
        # 选择关键指标进行雷达图展示
        radar_metrics = ['auc', 'accuracy', 'f1', 'r2', 'mape']
        radar_data = {}
        
        for exp_name, df in all_results.items():
            exp_metrics = {}
            for metric in radar_metrics:
                if metric in df.columns:
                    exp_metrics[metric] = df[metric].mean()
            if exp_metrics:
                radar_data[exp_name] = exp_metrics
        
        if radar_data:
            visualizer.plot_radar_chart(
                radar_data,
                metrics=radar_metrics,
                title='Comprehensive Performance Radar Chart',
                save_name='comprehensive_performance_radar.png'
            )
    except Exception as e:
        print(f"生成雷达图时出错: {e}")
    
    print("综合分析完成!")


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
    
    # 运行所有数据集的实验
    print("开始运行综合实验...")
    print("=" * 80)
    
    all_results = []
    
    for dataset_name in config['datasets']:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            dataset_results = run_comprehensive_experiments_for_dataset(dataset_name, config)
            all_results.extend(dataset_results)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
    
    # 生成综合报告
    generate_comprehensive_report(all_results)
    
    print("\n综合实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()