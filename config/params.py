"""超参数配置"""
import yaml
from pathlib import Path

def get_config(config_file='config.yaml'):
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / config_file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置默认值
    defaults = {
        'data': {
            'max_smiles_len': 100,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_seed': 42,
            'batch_size': 32,
            'num_workers': 4
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'patience': 10,
            'gradient_clip': 1.0
        }
    }
    
    # 合并配置
    for section in defaults:
        if section in config:
            for key in defaults[section]:
                if key not in config[section]:
                    config[section][key] = defaults[section][key]
        else:
            config[section] = defaults[section]
    
    return config

def save_config(config, path):
    """保存配置到文件"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)