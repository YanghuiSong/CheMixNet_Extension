"""运行CheMixNet实验的主脚本"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.paths import DATA_PATHS
from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset, create_dataloaders
from models.chemixnet import CheMixNetCNN, CheMixNetRNN, CheMixNetCNNRNN
from training.trainer import AdvancedTrainer
from evaluation.visualizer import ResultVisualizer
from evaluation.interpretation import ModelInterpreter

def run_original_chemixnet(dataset_name, model_type='cnn_mlp'):
    """运行原始 CheMixNet 实验（基于 Keras/TensorFlow）"""
    print("原始 CheMixNet (基于 TensorFlow/Keras) 已禁用，仅使用 PyTorch 版本")
    return None

def run_chemixnet_experiment(dataset_name, model_type='cnn', config=None, enhanced=False):
    """运行CheMixNet实验"""
    print(f"\n运行CheMixNet ({model_type.upper()}) - {dataset_name}")
    
    # 简化配置处理
    if config is None:
        config = {
            'data': {
                'max_smiles_len': 100,
                'test_size': 0.2,
                'val_size': 0.1,
                'random_seed': 42,
                'batch_size': 32,
                'num_workers': 0  # Windows下多进程可能有问题
            },
            'model': {
                'chemixnet': {
                    'cnn_channels': [64, 128, 256],
                    'fc_dims': [512, 256, 128],
                    'dropout_rate': 0.3
                }
            },
            'training': {
                'epochs': 5,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 3,
                'gradient_clip': 1.0
            },
            'dataset_info': {
                'bace': {'task': 'classification', 'target_col': 'Class'},
                'BBBP': {'task': 'classification', 'target_col': 'p_np'},
                'esol': {'task': 'regression', 'target_col': 'measured log solubility in mols per litre'},
                'HIV': {'task': 'classification', 'target_col': 'HIV_active'},
                'lipophilicity': {'task': 'regression', 'target_col': 'exp'}
            }
        }
    else:
        # 确保必要的配置项存在
        if 'data' not in config:
            config['data'] = {
                'max_smiles_len': 100,
                'test_size': 0.2,
                'val_size': 0.1,
                'random_seed': 42,
                'batch_size': 32,
                'num_workers': 0
            }
            
        if 'model' not in config:
            config['model'] = {
                'chemixnet': {
                    'cnn_channels': [64, 128, 256],
                    'fc_dims': [512, 256, 128],
                    'dropout_rate': 0.3
                }
            }

    # 加载和预处理数据
    preprocessor = DataPreprocessor(dataset_name)  # 修复：传递数据集名称而不是配置
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 划分数据集
    dataset_config = config['dataset_info'][dataset_name]
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df, 
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_seed=config['data'].get('random_seed', 42),
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征生成
    featurizer = MolecularFeaturizer(
        max_len=config['data'].get('max_smiles_len', 100)  # 修复：使用max_len而不是max_smiles_len
    )
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    train_dataset = MolecularDataset(
        train_df, featurizer, 
        task_type=task_info['task'],
        target_col=task_info['target_col']
    )
    val_dataset = MolecularDataset(
        val_df, featurizer,
        task_type=task_info['task'],
        target_col=task_info['target_col']
    )
    test_dataset = MolecularDataset(
        test_df, featurizer,
        task_type=task_info['task'],
        target_col=task_info['target_col']
    )
    
    # 创建数据加载器
    loaders = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data'].get('batch_size', 32),
        num_workers=0  # Windows下设置为0避免多进程问题
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # 获取词汇表大小
    vocab_size = len(featurizer.char_dict)
    fp_dim = 167  # MACCS指纹维度
    
    # 根据任务类型选择损失函数
    if task_info['task'] == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.BCEWithLogitsLoss()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    if model_type == 'cnn':
        # 获取隐藏层维度
        fc_dims = config['model']['chemixnet'].get('fc_dims', [512, 256, 128])
        hidden_dim = fc_dims[0] if isinstance(fc_dims, list) else fc_dims
            
        model = CheMixNetCNN(
            vocab_size=vocab_size,
            max_len=config['data'].get('max_smiles_len', 100),
            fp_dim=167,  # 修复：使用正确的指纹维度
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout_rate=config['model']['chemixnet'].get('dropout_rate', 0.3)
        )
    elif model_type == 'rnn':
        model = CheMixNetRNN(
            vocab_size=vocab_size,
            max_len=config['data'].get('max_smiles_len', 100),
            fp_dim=167,  # 修复：使用正确的指纹维度
            hidden_dim=config['model']['chemixnet'].get('hidden_dim', 128),
            lstm_layers=config['model']['chemixnet'].get('lstm_layers', 2),
            output_dim=1,
            dropout_rate=config['model']['chemixnet'].get('dropout_rate', 0.3)
        )
    elif model_type == 'cnnrnn':
        model = CheMixNetCNNRNN(
            vocab_size=vocab_size,
            max_len=config['data'].get('max_smiles_len', 100),
            fp_dim=167,  # 修复：使用正确的指纹维度
            cnn_channels=config['model']['chemixnet'].get('cnn_channels', [64, 128]),
            lstm_hidden=config['model']['chemixnet'].get('lstm_hidden', 128),
            lstm_layers=config['model']['chemixnet'].get('lstm_layers', 2),
            output_dim=1,
            dropout_rate=config['model']['chemixnet'].get('dropout_rate', 0.3)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 构建训练器配置
    trainer_config = {
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'task_type': task_info['task'],
        'patience': config['training']['patience'],
        'gradient_clip': config['training'].get('gradient_clip', 1.0)
    }
    
    # 初始化训练器
    trainer = AdvancedTrainer(
        model=model,
        config=trainer_config
    )
    
    # 训练模型
    trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    # 测试集评估
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"测试集结果: {test_metrics}")
    
    # 可视化结果
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_chemixnet_{model_type}")
    visualizer.plot_training_history(
        trainer.history,
        title=f'CheMixNet ({model_type.upper()}) Training - {dataset_name}',
        save_name='training_history.png'
    )
    
    # 模型解释（新增功能）
    interpreter = ModelInterpreter(model, featurizer, device=trainer.device)
    if hasattr(test_dataset, '__getitem__') and len(test_dataset) > 0:
        sample_data = test_dataset[0]
        if 'smiles' in sample_data and 'fingerprint' in sample_data:
            try:
                # 将数据移动到正确的设备上
                smiles_tensor = sample_data['smiles'].unsqueeze(0).to(trainer.device)
                fingerprint_tensor = sample_data['fingerprint'].unsqueeze(0).to(trainer.device)
                
                importance_scores, feature_names = interpreter.compute_feature_importance(
                    smiles_tensor,
                    fingerprint_tensor
                )
                # 只有当重要性计算成功时才绘制图表
                if importance_scores is not None and feature_names is not None:
                    visualizer.plot_feature_importance(
                        importance_scores,
                        feature_names,
                        title=f'Feature Importance - {dataset_name}',
                        save_name='feature_importance.png'
                    )
                else:
                    print("特征重要性计算返回了无效结果")
            except Exception as e:
                print(f"特征重要性计算失败: {e}")
    
    return test_metrics


def compare_chemixnet_models(dataset_name, config):
    """比较所有CheMixNet模型"""
    print(f"比较CheMixNet模型 - {dataset_name}")
    print("=" * 60)
    
    # 运行所有CheMixNet模型
    model_types = ['cnn', 'rnn', 'cnnrnn']
    results = {}
    
    for model_type in model_types:
        try:
            model_results = run_chemixnet_experiment(dataset_name, model_type, config)
            results[model_type.upper()] = model_results
        except Exception as e:
            print(f"CheMixNet {model_type} 运行失败: {str(e)}")
            results[model_type.upper()] = {}
    
    # 结果汇总和可视化
    comparison_data = []
    metric_keys = set()
    
    # 确保所有模型都出现在比较数据中，即使它们没有结果
    model_names = list(results.keys())
    for model_name in model_names:
        metrics = results.get(model_name, {})
        row = {'model': model_name}
        
        # 添加所有可用的指标
        for k, v in metrics.items():
            row[k] = v
            metric_keys.add(k)
            
        comparison_data.append(row)
    
    # 只有当有数据时才进行可视化
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # 确定主要评估指标
        task_info = config['dataset_info'][dataset_name]
        primary_metric = 'auc' if task_info['task'] == 'classification' else 'r2'
        if primary_metric not in metric_keys:
            primary_metric = list(metric_keys)[0] if metric_keys else 'loss'
        
        # 绘制模型比较图
        visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_chemixnet_comparison")
        fig = visualizer.plot_model_comparison(
            comparison_df,
            metric=primary_metric,
            title=f'CheMixNet Model Comparison - {dataset_name}',
            save_name='chemixnet_model_comparison.png'
        )
        
        # 保存结果到CSV
        comparison_df.to_csv(f"./results/{dataset_name}_chemixnet_comparison.csv", index=False)
        
        print("\nCheMixNet模型比较结果:")
        print(comparison_df)
    else:
        print("所有CheMixNet模型都运行失败，没有结果可以比较。")
    
    # 返回格式化的结果，用于跨数据集比较
    formatted_results = []
    for model_name in model_names:
        metrics = results.get(model_name, {})
        if metrics:  # 只返回有结果的模型
            result_entry = {'model': model_name}
            result_entry.update(metrics)
            formatted_results.append(result_entry)
        else:
            # 对于失败的模型，添加一个标记
            result_entry = {'model': model_name, 'status': 'failed'}
            formatted_results.append(result_entry)
    
    return formatted_results


def run_all_datasets_chemixnet_comparison(config):
    """在所有数据集上运行CheMixNet实验并生成综合比较"""
    # 数据集配置
    dataset_config = {
        'bace': {'task': 'classification', 'target_col': 'Class'},
        'BBBP': {'task': 'classification', 'target_col': 'p_np'},
        'esol': {'task': 'regression', 'target_col': 'measured log solubility in mols per litre'},
        'HIV': {'task': 'classification', 'target_col': 'HIV_active'},
        'lipophilicity': {'task': 'regression', 'target_col': 'exp'}
    }
    
    # 添加到配置中
    config['dataset_info'] = dataset_config
    
    # 在所有五个数据集上运行
    datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
    
    # 收集所有结果
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*80}")
        
        # 运行CheMixNet模型比较
        results = compare_chemixnet_models(dataset, config)
        all_results[dataset] = results
    
    # 生成跨数据集比较图表
    print(f"\n{'='*80}")
    print("生成跨数据集CheMixNet模型比较图表")
    print(f"{'='*80}")
    
    # 准备跨数据集比较数据
    cross_dataset_data = {}
    for dataset in datasets:
        # compare_chemixnet_models返回的是列表而不是字典
        cross_dataset_data[dataset] = all_results[dataset]
    
    # 检查是否有任何有效的结果
    has_valid_data = any(results for results in all_results.values() if results)
    
    if has_valid_data:
        # 绘制跨数据集比较图
        visualizer = ResultVisualizer(save_dir="./results/figures/cross_dataset_chemixnet_comparison")
        fig = visualizer.plot_cross_dataset_comparison(
            cross_dataset_data,
            metric='auc',  # 默认使用AUC，对于回归任务会自动适配
            title='Cross-Dataset CheMixNet Model Performance',
            save_name='cross_dataset_chemixnet_performance.png'
        )
        
        # 保存综合结果
        with open("./results/all_chemixnet_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n所有数据集的CheMixNet实验已完成！")
        print("结果已保存到 ./results/ 目录中")
    else:
        print("\n所有数据集的CheMixNet实验都失败了，没有结果可以比较。")


if __name__ == "__main__":
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 运行所有数据集的CheMixNet模型比较
    run_all_datasets_chemixnet_comparison(config)