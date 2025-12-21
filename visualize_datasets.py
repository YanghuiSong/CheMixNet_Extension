#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接运行的脚本，用于可视化5个原始数据集
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 尝试导入RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.Draw import IPythonConsole
    rdkit_available = True
except ImportError:
    rdkit_available = False
    print("警告: RDKit未安装，无法进行分子结构可视化")

from config.paths import DATA_PATHS, DATASET_INFO
from data.preprocessing import DataPreprocessor

# 简化版本的分子描述符函数，避免导入问题
def get_molecular_descriptors(mol):
    """计算分子描述符"""
    if mol is None:
        return {}
    
    descriptors = {}
    
    # 基本描述符
    descriptors['mol_weight'] = Descriptors.MolWt(mol)
    descriptors['num_atoms'] = mol.GetNumAtoms()
    descriptors['num_bonds'] = mol.GetNumBonds()
    descriptors['num_heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
    descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['num_h_donors'] = Descriptors.NumHDonors(mol)
    descriptors['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['tpsa'] = Descriptors.TPSA(mol)  # 拓扑极性表面积
    descriptors['logp'] = Descriptors.MolLogP(mol)  # 脂水分配系数
    
    # 环信息
    ring_info = mol.GetRingInfo()
    descriptors['num_rings'] = ring_info.NumRings()
    
    # 电荷信息
    descriptors['formal_charge'] = Chem.GetFormalCharge(mol)
    
    return descriptors

def load_and_process_datasets():
    """加载并处理所有数据集"""
    datasets = {}
    
    for name, path in DATA_PATHS.items():
        if not os.path.exists(path):
            print(f"警告: 数据集文件 {path} 不存在，跳过")
            continue
            
        print(f"加载数据集: {name}")
        try:
            preprocessor = DataPreprocessor(name)
            df = preprocessor.load_and_clean(path)
            datasets[name] = df
            print(f"  成功加载 {len(df)} 个样本")
        except Exception as e:
            print(f"  加载失败: {e}")
    
    return datasets

def visualize_molecular_structures(datasets, output_dir="./visualization_output/molecules"):
    """可视化分子结构"""
    if not rdkit_available:
        print("RDKit不可用，跳过分子结构可视化")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        print(f"可视化 {name} 数据集的分子结构...")
        try:
            # 取前9个分子进行网格显示
            sample_mols = df['mol'].head(9).tolist()
            
            # 过滤掉None值
            sample_mols = [m for m in sample_mols if m is not None]
            
            if sample_mols:
                # 创建分子网格图像
                img = Draw.MolsToGridImage(
                    sample_mols, 
                    molsPerRow=3, 
                    subImgSize=(300, 300),
                    legends=[f"Mol {i+1}" for i in range(len(sample_mols))]
                )
                
                # 保存图像
                save_path = os.path.join(output_dir, f"{name}_molecules.png")
                img.save(save_path)
                print(f"  分子结构图已保存至: {save_path}")
            else:
                print(f"  {name} 数据集中没有有效的分子")
                
        except Exception as e:
            print(f"  可视化分子结构时出错: {e}")

def visualize_target_distributions(datasets, output_dir="./visualization_output/distributions"):
    """可视化目标值分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        print(f"可视化 {name} 数据集的目标值分布...")
        
        # 获取目标列
        target_col = DATASET_INFO[name]['target_col']
        
        if target_col not in df.columns:
            print(f"  未找到目标列 '{target_col}'")
            continue
            
        try:
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 根据任务类型选择可视化方式
            task_type = DATASET_INFO[name]['task']
            
            if task_type == 'classification':
                # 分类任务 - 显示类别分布
                df[target_col].value_counts().plot(kind='bar')
                plt.xlabel('类别')
                plt.ylabel('样本数')
                plt.title(f'{name} - {target_col} 类别分布')
            else:
                # 回归任务 - 显示数值分布
                df[target_col].hist(bins=50, alpha=0.7)
                plt.xlabel(target_col)
                plt.ylabel('频率')
                plt.title(f'{name} - {target_col} 数值分布')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{name}_target_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  目标分布图已保存至: {save_path}")
            
        except Exception as e:
            print(f"  可视化目标分布时出错: {e}")
            plt.close()

def visualize_molecular_descriptors(datasets, output_dir="./visualization_output/descriptors"):
    """可视化分子描述符"""
    if not rdkit_available:
        print("RDKit不可用，跳过分子描述符计算和可视化")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个数据集计算描述符
    descriptor_stats = {}
    
    for name, df in datasets.items():
        print(f"计算 {name} 数据集的分子描述符...")
        
        try:
            # 计算描述符
            descriptors_list = []
            for mol in df['mol']:
                if mol is not None:
                    desc = get_molecular_descriptors(mol)
                    descriptors_list.append(desc)
                else:
                    descriptors_list.append({})
            
            # 转换为DataFrame
            desc_df = pd.DataFrame(descriptors_list)
            
            # 保存描述符统计信息
            descriptor_stats[name] = {
                'count': len(desc_df),
                'descriptors': desc_df.describe()
            }
            
            # 可视化关键描述符分布
            key_descriptors = ['mol_weight', 'logp', 'tpsa', 'num_h_donors', 'num_h_acceptors']
            available_descriptors = [d for d in key_descriptors if d in desc_df.columns]
            
            if available_descriptors:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, desc_name in enumerate(available_descriptors):
                    if desc_name in desc_df.columns:
                        desc_df[desc_name].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_xlabel(desc_name)
                        axes[i].set_ylabel('频率')
                        axes[i].set_title(f'{name} - {desc_name} 分布')
                
                # 隐藏多余的子图
                for j in range(len(available_descriptors), len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(output_dir, f"{name}_descriptors.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  描述符图已保存至: {save_path}")
            
        except Exception as e:
            print(f"  计算或可视化描述符时出错: {e}")
            plt.close()
    
    return descriptor_stats

def generate_summary_report(datasets, descriptor_stats, output_dir="./visualization_output"):
    """生成数据集概要报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("数据集概要报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for name, df in datasets.items():
        report_lines.append(f"数据集: {name}")
        report_lines.append("-" * 40)
        report_lines.append(f"样本数: {len(df)}")
        report_lines.append(f"任务类型: {DATASET_INFO[name]['task']}")
        report_lines.append(f"目标列: {DATASET_INFO[name]['target_col']}")
        
        if name in descriptor_stats:
            stats = descriptor_stats[name]
            report_lines.append(f"有效分子数: {stats['count']}")
        
        report_lines.append(f"列名: {list(df.columns)}")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    report_path = os.path.join(output_dir, "dataset_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"数据集概要报告已保存至: {report_path}")
    print("\n报告内容:")
    print(report_text)

def main():
    """主函数"""
    print("开始可视化原始数据集...")
    print("=" * 60)
    
    # 加载数据集
    datasets = load_and_process_datasets()
    
    if not datasets:
        print("没有成功加载任何数据集，退出")
        return
    
    # 创建输出目录
    output_base = "./visualization_output"
    os.makedirs(output_base, exist_ok=True)
    
    # 可视化分子结构
    print("\n1. 可视化分子结构...")
    visualize_molecular_structures(datasets, os.path.join(output_base, "molecules"))
    
    # 可视化目标分布
    print("\n2. 可视化目标分布...")
    visualize_target_distributions(datasets, os.path.join(output_base, "distributions"))
    
    # 可视化分子描述符
    print("\n3. 可视化分子描述符...")
    descriptor_stats = visualize_molecular_descriptors(datasets, os.path.join(output_base, "descriptors"))
    
    # 生成概要报告
    print("\n4. 生成概要报告...")
    generate_summary_report(datasets, descriptor_stats, output_base)
    
    print(f"\n所有可视化完成! 结果保存在 {output_base} 目录中")

if __name__ == "__main__":
    main()