"""运行消融实验"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset, create_dataloaders
from models.chemixnet import CheMixNetCNN, CheMixNetRNN
from models.multimodal import EnhancedCheMixNet
from training.trainer import AdvancedTrainer
from training.metrics import calculate_regression_metrics, calculate_classification_metrics
from config.paths import DATA_PATHS
from evaluation.visualizer import ResultVisualizer
from evaluation.interpretation import ModelInterpreter  # 添加模型解释器导入


def create_ablation_model(model_type, ablation_type, config, vocab_size):
    """创建用于消融实验的模型"""
    fp_dim = config.get('fp_dim', 166)
    max_len = config.get('max_smiles_len', 100)
    
    if model_type == 'chemixnet_cnn':
        if ablation_type == 'no_smiles':
            # 只使用指纹路径
            class FingerprintOnlyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fp_fc = torch.nn.Sequential(
                        torch.nn.Linear(fp_dim, 256),
                        torch.nn.BatchNorm1d(256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(256, 128),
                        torch.nn.BatchNorm1d(128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 1)
                    )
                
                def forward(self, smiles, fp):
                    return self.fp_fc(fp)
            
            return FingerprintOnlyModel()
        elif ablation_type == 'no_fingerprint':
            # 只使用SMILES路径
            class SmilesOnlyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.smiles_conv = torch.nn.Sequential(
                        torch.nn.Conv1d(max_len, 64, kernel_size=3, padding=1),
                        torch.nn.BatchNorm1d(64),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool1d(2),
                        torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
                        torch.nn.BatchNorm1d(128),
                        torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool1d(1)
                    )
                    self.smiles_fc = torch.nn.Linear(128, 256)
                    self.output = torch.nn.Linear(256, 1)
                    
                def forward(self, smiles, fp):
                    smiles_features = self.smiles_conv(smiles).squeeze(-1)
                    smiles_features = self.smiles_fc(smiles_features)
                    return self.output(smiles_features)
            
            return SmilesOnlyModel()
        else:
            # 完整模型
            return CheMixNetCNN(
                vocab_size=vocab_size,
                max_len=max_len,
                fp_dim=fp_dim,
                hidden_dim=256,
                output_dim=1,
                dropout_rate=0.3
            )
    
    elif model_type == 'chemixnet_rnn':
        if ablation_type == 'no_smiles':
            # 只使用指纹路径
            class FingerprintOnlyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fp_fc = torch.nn.Sequential(
                        torch.nn.Linear(fp_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(256, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(128, 1)
                    )
                
                def forward(self, smiles, fp):
                    return self.fp_fc(fp)
            
            return FingerprintOnlyModel()
        elif ablation_type == 'no_fingerprint':
            # 只使用SMILES路径
            class SmilesOnlyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = torch.nn.Embedding(vocab_size, 128)
                    self.lstm = torch.nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
                    self.smiles_fc = torch.nn.Sequential(
                        torch.nn.Linear(128, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3)
                    )
                    self.output = torch.nn.Linear(128, 1)
                    
                def forward(self, smiles, fp):
                    if smiles.dim() == 3:  # one-hot编码
                        smiles_indices = torch.argmax(smiles, dim=-1)
                    else:
                        smiles_indices = smiles
                    
                    embedded = self.embedding(smiles_indices)
                    lstm_out, (hidden, _) = self.lstm(embedded)
                    smiles_features = self.smiles_fc(hidden[-1])  # 使用最后一层隐藏状态
                    return self.output(smiles_features)
            
            return SmilesOnlyModel()
        else:
            # 完整模型
            return CheMixNetRNN(
                vocab_size=vocab_size,
                max_len=max_len,
                fp_dim=fp_dim,
                hidden_dim=128,
                lstm_layers=2,
                output_dim=1,
                dropout_rate=0.3
            )
    
    elif model_type == 'enhanced_chemixnet':
        if ablation_type == 'no_attention':
            # 不使用注意力机制的增强版模型
            return EnhancedCheMixNet(
                smiles_vocab_size=vocab_size,
                smiles_max_len=max_len,
                maccs_dim=fp_dim,
                atom_feature_dim=config.get('atom_feature_dim', 64),
                hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                output_dim=1,
                dropout_rate=0.3,
                use_attention=False
            )
        elif ablation_type == 'no_graph':
            # 不使用图数据的增强版模型
            model = EnhancedCheMixNet(
                smiles_vocab_size=vocab_size,
                smiles_max_len=max_len,
                maccs_dim=fp_dim,
                atom_feature_dim=config.get('atom_feature_dim', 64),
                hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                output_dim=1,
                dropout_rate=0.3,
                use_attention=True
            )
            # 重写forward方法，使其不使用图数据
            original_forward = model.forward
            def forward_without_graph(smiles_input, fp_input, graph_data=None):
                return original_forward(smiles_input, fp_input, None)
            model.forward = forward_without_graph
            return model
        else:
            # 完整的增强版模型
            return EnhancedCheMixNet(
                smiles_vocab_size=vocab_size,
                smiles_max_len=max_len,
                maccs_dim=fp_dim,
                atom_feature_dim=config.get('atom_feature_dim', 64),
                hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                output_dim=1,
                dropout_rate=0.3,
                use_attention=True
            )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def run_single_ablation_experiment(dataset_name, model_type, ablation_type, config):
    """运行单个消融实验"""
    print(f"运行消融实验 - 数据集: {dataset_name}, 模型: {model_type}, 消融类型: {ablation_type}")
    print("=" * 60)
    
    # 数据预处理
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 数据集划分
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df, 
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1),
        random_seed=config.get('random_seed', 42),
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征生成
    featurizer = MolecularFeaturizer(max_len=config.get('max_smiles_len', 100))
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    use_graph = (model_type == 'enhanced_chemixnet' and ablation_type != 'no_graph')
    train_dataset = MolecularDataset(
        train_df, featurizer, 
        task_type=task_info['task'],
        target_col=task_info['target_col'],
        use_graph=use_graph
    )
    val_dataset = MolecularDataset(
        val_df, featurizer,
        task_type=task_info['task'],
        target_col=task_info['target_col'],
        use_graph=use_graph
    )
    test_dataset = MolecularDataset(
        test_df, featurizer,
        task_type=task_info['task'],
        target_col=task_info['target_col'],
        use_graph=use_graph
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 0)
    )
    
    # 创建模型
    vocab_size = len(featurizer.char_dict)
    model = create_ablation_model(model_type, ablation_type, config, vocab_size)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    trainer_config = {
        'learning_rate': config.get('learning_rate', 0.001),
        'weight_decay': config.get('weight_decay', 1e-5),
        'task_type': task_info['task'],
        'patience': config.get('patience', 10),
        'epochs': config.get('epochs', 100)
    }
    trainer = AdvancedTrainer(model, trainer_config)
    
    # 训练模型
    print(f"开始训练消融模型 ({ablation_type})...")
    trainer.train(train_loader, val_loader, epochs=config.get('epochs', 100))
    
    # 在测试集上评估
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"测试集结果 ({ablation_type}): {test_metrics}")
    
    # 绘制训练历史
    visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_ablation_{model_type}_{ablation_type}")
    visualizer.plot_training_history(
        trainer.history,
        title=f'Ablation Study ({ablation_type}) - {dataset_name} ({model_type})',
        save_name='training_history.png'
    )
    
    # 模型解释与特征重要性分析
    try:
        interpreter = ModelInterpreter(model, featurizer, device=trainer.device)
        if len(test_dataset) > 0:
            # 获取一个测试样本进行特征重要性分析
            sample_data = test_dataset[0]
            if hasattr(sample_data, 'smiles') and hasattr(sample_data, 'fingerprint'):
                importance_scores, feature_names = interpreter.compute_feature_importance(
                    sample_data['smiles'].unsqueeze(0).to(trainer.device),
                    sample_data['fingerprint'].unsqueeze(0).to(trainer.device)
                )
                if importance_scores is not None and feature_names is not None:
                    visualizer.plot_feature_importance(
                        importance_scores,
                        feature_names,
                        title=f'Feature Importance - {dataset_name} ({model_type}, {ablation_type})',
                        save_name='feature_importance.png'
                    )
                else:
                    print("特征重要性计算返回了无效结果")
    except Exception as e:
        print(f"特征重要性分析失败: {str(e)}")
    
    return {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'ablation_type': ablation_type,
        'model_type': model_type
    }


def run_ablation_study(dataset_name, model_type, config):
    """运行消融实验"""
    print(f"运行{model_type}消融实验 - 数据集: {dataset_name}")
    print("=" * 60)
    
    # 根据模型类型定义消融类型
    if model_type in ['chemixnet_cnn', 'chemixnet_rnn']:
        ablation_types = ['full', 'no_smiles', 'no_fingerprint']
    elif model_type == 'enhanced_chemixnet':
        ablation_types = ['full', 'no_attention', 'no_graph']
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    results = {}
    
    for ab_type in ablation_types:
        try:
            result = run_single_ablation_experiment(dataset_name, model_type, ab_type, config)
            results[ab_type] = result
        except Exception as e:
            print(f"消融实验 {ab_type} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[ab_type] = {}
    
    return results


def compare_ablation_studies(dataset_name, config):
    """比较所有消融实验结果"""
    print(f"比较消融实验 - {dataset_name}")
    print("=" * 60)
    
    # 定义要比较的模型类型
    model_types = ['chemixnet_cnn', 'chemixnet_rnn', 'enhanced_chemixnet']
    
    all_results = {}
    
    for model_type in model_types:
        # 运行消融实验
        results = run_ablation_study(dataset_name, model_type, config)
        all_results[model_type] = results
    
    # 结果汇总和可视化
    comparison_data = []
    metric_keys = set()
    
    for model_type, ablation_results in all_results.items():
        for ab_type, result_data in ablation_results.items():
            # 只有当实验有结果时才添加到比较数据中
            if result_data and 'test_metrics' in result_data:
                metrics = result_data['test_metrics']
                row = {
                    'model': f"{model_type}_{ab_type}",
                    'model_type': model_type,
                    'ablation_type': ab_type
                }
                for k, v in metrics.items():
                    row[k] = v
                    metric_keys.add(k)
                comparison_data.append(row)
    
    # 只有当有数据时才进行可视化
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # 确定主要评估指标
        task_type = config.get('task_type', 'regression')
        primary_metric = 'auc' if task_type == 'classification' else 'r2'
        if primary_metric not in metric_keys:
            primary_metric = list(metric_keys)[0] if metric_keys else 'loss'
        
        # 绘制模型比较图
        visualizer = ResultVisualizer(save_dir=f"./results/figures/{dataset_name}_ablation_comparison")
        visualizer.plot_model_comparison(
            comparison_df,
            metric=primary_metric,
            title=f'Ablation Study Comparison - {dataset_name}',
            save_name='ablation_comparison.png'
        )
        
        # 保存结果到CSV
        comparison_df.to_csv(f"./results/{dataset_name}_ablation_comparison.csv", index=False)
        
        print("\n消融实验比较结果:")
        print(comparison_df)
    else:
        print("所有消融实验都运行失败，没有结果可以比较。")
    
    # 返回格式化的结果，用于跨数据集比较
    formatted_results = []
    for model_type, ablation_results in all_results.items():
        for ab_type, result_data in ablation_results.items():
            if result_data and 'test_metrics' in result_data:
                metrics = result_data['test_metrics']
                result_entry = {
                    'model': f"{model_type}_{ab_type}",
                    'ablation_type': ab_type,
                    'model_type': model_type
                }
                result_entry.update(metrics)
                formatted_results.append(result_entry)
    
    return formatted_results


def run_all_datasets_ablation_comparison():
    """在所有数据集上运行消融实验并生成综合比较"""
    # 示例配置
    config = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        'dropout': 0.1,
        'max_smiles_len': 100,
        'fp_dim': 166,      # MACCS指纹维度
        'atom_feature_dim': 64,
        'hidden_dims': [256, 128, 64],
        'test_size': 0.2,
        'val_size': 0.1,
        'random_seed': 42,
        'num_workers': 0
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
        dataset_specific_config['task_type'] = dataset_config[dataset]['task_type']
        
        # 运行消融实验比较
        results = compare_ablation_studies(dataset, dataset_specific_config)
        all_results[dataset] = results
    
    # 生成跨数据集比较图表
    print(f"\n{'='*80}")
    print("生成跨数据集消融实验比较图表")
    print(f"{'='*80}")
    
    # 准备跨数据集比较数据
    cross_dataset_data = {}
    for dataset in datasets:
        cross_dataset_data[dataset] = all_results[dataset]
    
    # 确保保存目录存在
    os.makedirs("./results/figures/cross_dataset_ablation_comparison", exist_ok=True)
    
    # 绘制跨数据集比较图
    visualizer = ResultVisualizer(save_dir="./results/figures/cross_dataset_ablation_comparison")
    visualizer.plot_cross_dataset_comparison(
        cross_dataset_data,
        metric=None,  # 自动选择最佳指标
        title='Cross-Dataset Ablation Study Performance',
        save_name='cross_dataset_ablation_performance.png'
    )
    
    # 保存所有结果
    with open("./results/all_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n所有数据集的消融实验已完成！")
    print("结果已保存到 ./results/ 目录中")


if __name__ == "__main__":
    # 运行所有数据集的消融实验比较
    run_all_datasets_ablation_comparison()