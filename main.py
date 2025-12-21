"""项目主入口"""
import argparse
import yaml
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config.paths import DATA_PATHS
from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from models.multimodal import EnhancedCheMixNet
from training.trainer import AdvancedTrainer
from evaluation.visualizer import ResultVisualizer

def main():
    parser = argparse.ArgumentParser(description='增强版CheMixNet分子性质预测')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity'],
                       help='要使用的数据集')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'analyze'],
                       help='运行模式')
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['mlp', 'cnn', 'rnn', 'chemixnet', 'enhanced'],
                       help='模型类型')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"使用数据集: {args.dataset}")
    print(f"模型类型: {args.model_type}")
    print(f"运行模式: {args.mode}")
    
    # 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = DataPreprocessor(args.dataset)
    data_path = DATA_PATHS[args.dataset]
    df = preprocessor.load_and_clean(data_path)
    
    # 特征生成
    print("\n2. 特征生成...")
    featurizer = MolecularFeaturizer(
        max_len=config['data']['max_smiles_len']
    )
    featurizer.build_char_vocab(df['smiles'].tolist())
    
    # 准备数据加载器
    print("\n3. 准备数据加载器...")
    # 这里需要实现数据加载器的具体逻辑
    # train_loader, val_loader, test_loader = prepare_dataloaders(...)
    
    # 创建模型
    print("\n4. 创建模型...")
    if args.model_type == 'enhanced':
        model = EnhancedCheMixNet(
            smiles_vocab_size=len(featurizer.char_dict),
            smiles_max_len=config['data']['max_smiles_len'],
            maccs_dim=167,
            atom_feature_dim=config['model']['enhanced']['atom_feature_dim'],
            hidden_dims=config['model']['enhanced']['hidden_dims'],
            output_dim=1,
            dropout_rate=config['model']['enhanced']['dropout_rate'],
            use_attention=config['model']['enhanced']['use_attention']
        )
    else:
        # 其他模型类型...
        pass
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练或测试
    if args.mode == 'train':
        print("\n5. 开始训练...")
        trainer_config = {
            'task_type': preprocessor.get_task_info()['task'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'patience': config['training']['patience'],
            'save_dir': config['paths']['checkpoints_dir']
        }
        
        trainer = AdvancedTrainer(model, trainer_config)
        # trainer.train(train_loader, val_loader, 
        #              epochs=config['training']['epochs'])
        
        # 绘制训练历史
        # trainer.plot_training_history()
        print("训练功能待完善...")
        
    elif args.mode == 'test':
        print("\n5. 模型测试...")
        # 测试逻辑...
        print("测试功能待完善...")
        pass
    
    elif args.mode == 'analyze':
        print("\n5. 结果分析...")
        visualizer = ResultVisualizer()
        # 分析逻辑...
        print("分析功能待完善...")
        pass
    
    print("\n任务完成!")

if __name__ == '__main__':
    main()