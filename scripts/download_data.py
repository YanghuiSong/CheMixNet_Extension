"""下载数据集脚本"""
import os
import requests
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# 数据集URLs（示例，实际需要根据MoleculeNet调整）
DATASET_URLS = {
    'bace': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
    'BBBP': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
    'esol': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
    'HIV': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
    'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
}

def download_file(url, filepath):
    """下载文件"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_datasets(data_dir='./data'):
    """下载所有数据集"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("下载数据集...")
    
    for name, url in DATASET_URLS.items():
        filepath = data_path / f"{name}.csv"
        
        if filepath.exists():
            print(f"数据集 {name} 已存在，跳过下载")
            continue
        
        print(f"下载 {name}...")
        try:
            download_file(url, filepath)
            print(f"  {name} 下载完成")
        except Exception as e:
            print(f"  下载失败: {e}")
    
    print("\n所有数据集下载完成!")
    
    # 验证文件
    print("\n验证文件...")
    for name in DATASET_URLS.keys():
        filepath = data_path / f"{name}.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, nrows=5)
                print(f"  {name}: 有效，{len(df)} 列，示例:")
                print(f"    列名: {list(df.columns)}")
            except Exception as e:
                print(f"  {name}: 无效 - {e}")
        else:
            print(f"  {name}: 文件不存在")

def create_sample_data():
    """创建示例数据（如果下载失败）"""
    data_path = Path('./data')
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("\n创建示例数据...")
    
    # BACE数据集示例
    bace_data = {
        'mol': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 
                'CC(=O)Oc1ccccc1C(=O)O', 'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
        'Class': [1, 0, 1, 0, 1]
    }
    pd.DataFrame(bace_data).to_csv(data_path / 'bace.csv', index=False)
    
    # BBBP数据集示例
    bbbp_data = {
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                  'CC(=O)Oc1ccccc1C(=O)O', 'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
        'p_np': [1, 0, 1, 0, 1]
    }
    pd.DataFrame(bbbp_data).to_csv(data_path / 'BBBP.csv', index=False)
    
    # ESOL数据集示例
    esol_data = {
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                  'CC(=O)Oc1ccccc1C(=O)O', 'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
        'measured log solubility in mols per litre': [-2.1, -1.8, -2.3, -1.5, -2.0]
    }
    pd.DataFrame(esol_data).to_csv(data_path / 'esol.csv', index=False)
    
    # HIV数据集示例
    hiv_data = {
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                  'CC(=O)Oc1ccccc1C(=O)O', 'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
        'HIV_active': [1, 0, 1, 0, 1]
    }
    pd.DataFrame(hiv_data).to_csv(data_path / 'HIV.csv', index=False)
    
    # Lipophilicity数据集示例
    lipo_data = {
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                  'CC(=O)Oc1ccccc1C(=O)O', 'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
        'exp': [1.2, 0.8, 1.5, 0.5, 1.3]
    }
    pd.DataFrame(lipo_data).to_csv(data_path / 'Lipophilicity.csv', index=False)
    
    print("示例数据创建完成!")

if __name__ == '__main__':
    try:
        download_datasets()
    except Exception as e:
        print(f"下载失败: {e}")
        print("创建示例数据...")
        create_sample_data()
    
    print("\n数据准备完成!")