"""测试修复后的RNN模型"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from config.paths import DATA_PATHS
from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset, create_dataloaders
from models.chemixnet import CheMixNetRNN

def test_rnn_model():
    print("测试修复后的RNN模型...")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以加快实验速度
    config['data']['batch_size'] = 16
    config['training']['epochs'] = 3  # 只训练3个周期以加快速度
    
    dataset_name = 'esol'
    print(f"实验数据集: {dataset_name}")
    
    # 数据预处理
    print("1. 数据预处理...")
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 使用较小的数据集加快实验速度
    df = df.head(100)
    
    # 划分数据集
    task_info = preprocessor.get_task_info()
    train_df, val_df, test_df = split_dataset(
        df, 
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_seed=config['data']['random_seed'],
        stratify_col=task_info['target_col'] if task_info['task'] == 'classification' else None
    )
    
    # 特征生成
    print("2. 特征生成...")
    featurizer = MolecularFeaturizer(
        max_len=config['data']['max_smiles_len']
    )
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 创建数据集
    print("3. 创建数据集...")
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
    print("4. 创建数据加载器...")
    loaders = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=0  # Windows兼容性
    )
    
    # 测试数据加载
    print("5. 测试数据加载...")
    for batch_idx, batch in enumerate(loaders['train']):
        print(f"批次 {batch_idx}:")
        print(f"  SMILES 形状: {batch['smiles'].shape}")
        print(f"  指纹形状: {batch['fingerprint'].shape}")
        print(f"  目标形状: {batch['target'].shape}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    # 创建模型
    print("6. 创建RNN模型...")
    vocab_size = len(featurizer.char_dict)
    model = CheMixNetRNN(
        vocab_size=vocab_size,
        max_len=config['data']['max_smiles_len'],
        fp_dim=167,  # 修正指纹维度
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    print("7. 测试前向传播...")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loaders['train']):
            smiles = batch['smiles']
            fp = batch['fingerprint']
            
            print(f"输入形状 - SMILES: {smiles.shape}, FP: {fp.shape}")
            
            try:
                output = model(smiles, fp)
                print(f"输出形状: {output.shape}")
                print(f"输出示例: {output.flatten()[:5]}")
            except Exception as e:
                print(f"前向传播出错: {e}")
                import traceback
                traceback.print_exc()
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
    
    print("\nRNN模型测试完成!")

if __name__ == '__main__':
    test_rnn_model()