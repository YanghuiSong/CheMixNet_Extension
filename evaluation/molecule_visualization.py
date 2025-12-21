#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子可视化脚本
用于可视化分子结构、特征重要性等
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.analyzer import ResultAnalyzer
from evaluation.visualizer import ResultVisualizer
from evaluation.interpretation import ModelInterpreter

def visualize_molecule(smiles, title="Molecule"):
    """
    可视化单个分子结构
    
    Args:
        smiles (str): SMILES字符串
        title (str): 图标题
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            plt.show()
        else:
            print(f"无法解析SMILES: {smiles}")
    except ImportError:
        print("RDKit未安装，无法可视化分子结构")

def visualize_molecule_from_smiles(smiles, output_dir="./visualization_output", filename=None):
    """
    从SMILES字符串可视化分子结构并保存到文件
    
    Args:
        smiles (str): SMILES字符串
        output_dir (str): 输出目录
        filename (str): 保存的文件名，默认使用SMILES生成
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 生成分子图像
            img = Draw.MolToImage(mol, size=(300, 300))
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            if filename is None:
                # 使用SMILES的一部分作为文件名
                safe_smiles = "".join(c for c in smiles if c.isalnum() or c in ('-', '_')).rstrip()
                filename = f"molecule_{safe_smiles[:20]}.png"  # 限制长度
            
            # 保存图像
            save_path = os.path.join(output_dir, filename)
            img.save(save_path)
            print(f"分子结构图已保存至: {save_path}")
            return save_path
        else:
            print(f"无法解析SMILES: {smiles}")
            return None
    except ImportError:
        print("RDKit未安装，无法可视化分子结构")
        return None
    except Exception as e:
        print(f"可视化分子结构时出错: {e}")
        return None

def visualize_feature_importance(importance_data, top_k=20, save_path=None):
    """
    可视化特征重要性
    
    Args:
        importance_data (dict): 特征重要性数据
        top_k (int): 显示前K个重要特征
        save_path (str): 保存路径
    """
    if not importance_data:
        print("没有特征重要性数据")
        return
    
    # 获取前K个重要特征
    sorted_features = sorted(importance_data.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(features))
    colors = ['red' if x < 0 else 'blue' for x in importances]
    
    plt.barh(y_pos, importances, color=colors, alpha=0.7)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Feature Importances')
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (pos, imp) in enumerate(zip(y_pos, importances)):
        plt.text(imp + (0.01 if imp >= 0 else -0.01), pos, 
                f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_results_comparison(results_dir="./results"):
    """
    可视化实验结果比较
    
    Args:
        results_dir (str): 结果目录路径
    """
    analyzer = ResultAnalyzer(results_dir)
    visualizer = ResultVisualizer(save_dir=f"{results_dir}/figures/comparison")
    
    # 加载各实验结果
    experiments = ['baseline', 'chemixnet', 'multimodal', 'ablation']
    loaded_experiments = []
    
    for exp_name in experiments:
        df = analyzer.load_results(exp_name, 'results.csv')
        if df is not None:
            loaded_experiments.append(exp_name)
    
    if not loaded_experiments:
        print("没有找到实验结果文件")
        return
    
    # 比较实验结果
    metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'mse', 'mae', 'rmse', 'mape', 'r2']
    
    for metric in metrics:
        comparison_df = analyzer.compare_experiments(loaded_experiments, metric)
        if not comparison_df.empty:
            # 创建比较图表
            plt.figure(figsize=(12, 8))
            
            # 绘制柱状图
            x_pos = np.arange(len(loaded_experiments))
            means = comparison_df['mean'].values
            stds = comparison_df['std'].values
            
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(loaded_experiments))))
            plt.xticks(x_pos, loaded_experiments, rotation=45)
            plt.ylabel(metric.upper())
            plt.title(f'Comparison of {metric.upper()} Across Experiments')
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for i, (mean, std) in enumerate(zip(means, stds)):
                plt.text(i, mean + std + 0.01, f'{mean:.4f}', 
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            save_path = f"{results_dir}/figures/comparison/{metric}_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{metric}比较图已保存至: {save_path}")
            plt.close()

def generate_comparison_plots(output_dir):
    """生成实验结果比较图"""
    print("正在生成实验结果比较图...")
    
    # 查找结果文件
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"结果目录 {results_dir} 不存在")
        return
    
    # 查找所有CSV结果文件
    csv_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_comparison.csv') or file.endswith('_results.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("未找到结果文件")
        return
    
    print(f"找到 {len(csv_files)} 个结果文件")
    
    # 为每个CSV文件生成比较图
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
                
            # 生成简单的条形图比较
            plt.figure(figsize=(10, 6))
            
            # 查找数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                continue
                
            # 选择第一个数值列进行比较
            metric_col = numeric_columns[0]
            
            # 如果有模型列，则按模型分组
            if 'model' in df.columns:
                models = df['model'].tolist()
                values = df[metric_col].tolist()
                
                # 绘制条形图
                plt.bar(models, values)
                plt.xlabel('Models')
                plt.ylabel(metric_col)
                plt.title(f'Model Comparison - {os.path.basename(csv_file)}')
                plt.xticks(rotation=45, ha='right')
            else:
                # 绘制所有数值列的比较
                df[numeric_columns].plot(kind='bar')
                plt.xlabel('Entries')
                plt.ylabel('Metrics')
                plt.title(f'Results Comparison - {os.path.basename(csv_file)}')
                plt.legend(numeric_columns)
            
            plt.tight_layout()
            
            # 保存图像
            output_filename = os.path.splitext(os.path.basename(csv_file))[0] + '_comparison.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"比较图已保存至: {output_path}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

def plot_feature_importance_from_file(importance_file, output_dir="./visualization_output"):
    """
    从特征重要性文件生成可视化图表
    
    Args:
        importance_file (str): 特征重要性文件路径（JSON或CSV格式）
        output_dir (str): 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取特征重要性数据
        if importance_file.endswith('.json'):
            import json
            with open(importance_file, 'r') as f:
                importance_data = json.load(f)
        elif importance_file.endswith('.csv'):
            df = pd.read_csv(importance_file)
            # 假设第一列是特征名，第二列是重要性分数
            if len(df.columns) >= 2:
                importance_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            else:
                print("CSV文件格式不正确，需要至少两列数据")
                return
        else:
            print("不支持的文件格式，请使用JSON或CSV格式")
            return
        
        if not importance_data:
            print("特征重要性数据为空")
            return
        
        # 转换为可视化所需的格式
        features = list(importance_data.keys())
        importances = list(importance_data.values())
        
        # 只取前20个最重要的特征
        top_k = min(20, len(features))
        # 按重要性绝对值排序
        sorted_indices = np.argsort(np.abs(importances))[::-1][:top_k]
        top_features = [features[i] for i in sorted_indices]
        top_importances = [importances[i] for i in sorted_indices]
        
        # 创建可视化图表
        plt.figure(figsize=(12, max(6, top_k * 0.3)))
        y_pos = np.arange(len(top_features))
        colors = ['red' if x < 0 else 'blue' for x in top_importances]
        
        bars = plt.barh(y_pos, top_importances, color=colors, alpha=0.7)
        plt.yticks(y_pos, top_features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importances')
        plt.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, top_importances)):
            plt.text(imp + (abs(imp) * 0.01 if imp >= 0 else -abs(imp) * 0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{imp:.4f}', 
                    ha='left' if imp >= 0 else 'right', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征重要性图已保存至: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"从文件生成特征重要性图时出错: {e}")
        return None


def generate_sample_feature_importance(output_dir="./sample_data", num_features=30):
    """
    生成示例特征重要性数据用于测试可视化功能
    
    Args:
        output_dir (str): 输出目录
        num_features (int): 特征数量
        
    Returns:
        str: 生成的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成示例特征重要性数据
    np.random.seed(42)  # 固定随机种子以获得可重现的结果
    feature_names = [f"Feature_{i}" for i in range(num_features)]
    # 生成一些正负重要性值，模拟真实场景
    importance_scores = np.random.randn(num_features) * 0.5
    
    # 创建字典格式的数据
    importance_data = dict(zip(feature_names, importance_scores))
    
    # 保存为JSON格式
    json_path = os.path.join(output_dir, "sample_feature_importance.json")
    with open(json_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    
    print(f"示例特征重要性数据已保存至: {json_path}")
    
    # 同时保存为CSV格式
    csv_path = os.path.join(output_dir, "sample_feature_importance.csv")
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    df.to_csv(csv_path, index=False)
    print(f"示例特征重要性数据已保存至: {csv_path}")
    
    return json_path

def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='分子可视化工具')
    parser.add_argument('--mode', choices=['molecule', 'feature', 'comparison', 'generate_sample'], 
                       default='molecule', help='可视化模式')
    parser.add_argument('--smiles', type=str, 
                       help='SMILES字符串，用于分子结构可视化')
    parser.add_argument('--csv_file', type=str, 
                       help='包含分子数据的CSV文件路径')
    parser.add_argument('--importance_file', type=str, 
                       help='特征重要性文件路径')
    parser.add_argument('--output_dir', type=str, default='./visualization_output',
                       help='输出目录')
    parser.add_argument('--num_features', type=int, default=30,
                       help='生成示例数据时的特征数量')
    
    # 如果没有传入args，则使用命令行参数
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, list):
        args = parser.parse_args(args)
    # 如果args已经是命名空间对象，则直接使用
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'molecule':
        if not args.smiles:
            print("请提供SMILES字符串用于分子可视化")
            return
        visualize_molecule_from_smiles(args.smiles, args.output_dir)
        
    elif args.mode == 'feature':
        if not args.importance_file:
            print("请提供特征重要性文件路径")
            return
        plot_feature_importance_from_file(args.importance_file, args.output_dir)
        
    elif args.mode == 'generate_sample':
        # 生成示例特征重要性数据
        generated_file = generate_sample_feature_importance(args.output_dir, args.num_features)
        print(f"示例数据已生成: {generated_file}")
        print("您可以使用以下命令进行可视化:")
        print(f"  python evaluation/molecule_visualization.py --mode feature --importance_file {generated_file} --output_dir {args.output_dir}")
        
    elif args.mode == 'comparison':
        # 尝试从已有结果文件生成比较图
        generate_comparison_plots(args.output_dir)

if __name__ == "__main__":
    main()
