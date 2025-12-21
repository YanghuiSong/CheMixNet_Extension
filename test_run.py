"""快速测试脚本"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import yaml
import torch
from config.paths import DATA_PATHS
from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from models.chemixnet import CheMixNetCNN
from training.trainer import AdvancedTrainer

def test_system():
    print("开始测试系统功能...")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 选择一个小数据集进行测试
    dataset_name = 'esol'
    print(f"测试数据集: {dataset_name}")
    
    # 数据预处理
    print("1. 数据预处理...")
    preprocessor = DataPreprocessor(dataset_name)
    data_path = DATA_PATHS[dataset_name]
    df = preprocessor.load_and_clean(data_path)
    
    # 只使用一小部分数据进行快速测试
    df = df.head(100)  # 只使用前100个样本
    
    print(f"使用样本数: {len(df)}")
    
    # 特征生成
    print("2. 特征生成...")
    featurizer = MolecularFeaturizer(
        max_len=config['data']['max_smiles_len']
    )
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 测试模型创建
    print("3. 创建模型...")
    vocab_size = len(featurizer.char_dict)
    model = CheMixNetCNN(
        vocab_size=vocab_size,
        max_len=config['data']['max_smiles_len'],
        fp_dim=167,
        hidden_dim=128,
        output_dim=1,
        dropout_rate=0.3
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    print("4. 测试前向传播...")
    
    # 创建一个示例输入
    sample_smiles = torch.randn(4, config['data']['max_smiles_len'], vocab_size)  # 4个样本
    sample_fp = torch.randn(4, 167)
    
    print(f"SMILES输入形状: {sample_smiles.shape}")
    print(f"指纹输入形状: {sample_fp.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(sample_smiles, sample_fp)
    
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output.flatten()[:5]}")  # 显示前5个输出值
    
    print("\n系统测试完成!")

if __name__ == '__main__':
    test_system()