"""结果可视化工具"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Union, Optional, Any

class ResultVisualizer:
    """结果可视化器"""
    
    # 定义统一的颜色方案
    COLOR_PALETTE = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }
    
    # 定义统一的字体和样式
    DEFAULT_STYLE = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    }
    
    def __init__(self, save_dir='./results/figures', style='default'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        if style == 'seaborn':
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif style == 'default':
            plt.rcParams.update(self.DEFAULT_STYLE)
        
        # 关闭交互模式，防止图像弹出显示
        plt.ioff()
    
    def _get_unique_save_path(self, save_name):
        """生成唯一的保存路径，避免覆盖已有的图像文件"""
        save_path = self.save_dir / save_name
        counter = 1
        while save_path.exists():
            name_parts = save_name.split('.')
            if len(name_parts) > 1:
                # 如果文件有扩展名
                new_name = '.'.join(name_parts[:-1]) + f'_{counter}.' + name_parts[-1]
            else:
                # 如果文件没有扩展名
                new_name = save_name + f'_{counter}'
            save_path = self.save_dir / new_name
            counter += 1
        return save_path
    
    def _setup_subplot_style(self, ax, title="", xlabel="", ylabel=""):
        """设置子图统一样式"""
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_feature_importance(self, importance_scores, feature_names=None, 
                              top_k=20, title='Feature Importance', 
                              save_name='feature_importance.png'):
        """
        绘制特征重要性图
        
        Args:
            importance_scores: 特征重要性分数数组
            feature_names: 特征名称列表
            top_k: 显示前K个最重要的特征
            title: 图标题
            save_name: 保存文件名
        """
        # 转换为numpy数组
        scores = np.array(importance_scores)
        
        # 如果没有提供特征名称，创建默认名称
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(scores))]
        
        # 确保特征名称和分数长度一致
        if len(feature_names) > len(scores):
            feature_names = feature_names[:len(scores)]
        elif len(feature_names) < len(scores):
            feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), len(scores))])
        
        # 获取top_k个最重要的特征（按绝对值排序）
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(np.abs(scores))[-top_k:][::-1]  # 按绝对值排序
        
        top_scores = scores[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # 创建水平条形图
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        y_pos = np.arange(len(top_scores))
        colors = [self.COLOR_PALETTE['primary'] if score >= 0 else self.COLOR_PALETTE['secondary'] 
                 for score in top_scores]
        
        bars = ax.barh(y_pos, top_scores, color=colors, alpha=0.8)
        
        # 设置标签和标题
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        # 添加网格线
        ax.grid(True, axis='x', alpha=0.3)
        
        # 在条形图上添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            width = bar.get_width()
            ax.text(width + (abs(width) * 0.01 if width >= 0 else -abs(width) * 0.01), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', 
                   ha='left' if width >= 0 else 'right', 
                   va='center',
                   fontsize=9)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征重要性图已保存至: {save_path}")

    def plot_model_comparison(self, results_data, metric='test_loss', 
                           title='Model Comparison', save_name='model_comparison.png'):
        """
        绘制模型比较图
        
        Args:
            results_data: 包含模型结果的列表或DataFrame
            metric: 用于比较的指标名称
            title: 图标题
            save_name: 保存文件名
        """
        # 如果输入是列表，转换为DataFrame
        if isinstance(results_data, list):
            df = pd.DataFrame(results_data)
        else:
            df = results_data
            
        # 检查必要的列是否存在
        if metric not in df.columns:
            raise ValueError(f"指标 '{metric}' 不存在于数据中")
            
        # 创建模型名称列（如果没有的话）
        if 'model' not in df.columns and 'model_name_suffix' in df.columns:
            df['model'] = df['model_name_suffix']
        elif 'model' not in df.columns:
            df['model'] = [f'Model {i}' for i in range(len(df))]
            
        # 按指标值排序
        df_sorted = df.sort_values(by=metric, ascending=True)
        
        # 创建条形图
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.4)))
        
        y_pos = np.arange(len(df_sorted))
        bars = ax.barh(y_pos, df_sorted[metric], 
                      color=self.COLOR_PALETTE['primary'], alpha=0.8)
        
        # 设置标签和标题
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['model'])
        ax.set_xlabel(metric.upper())
        ax.set_title(title, fontweight='bold', pad=20)
        
        # 添加网格线
        ax.grid(True, axis='x', alpha=0.3)
        
        # 在条形图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
            width = bar.get_width()
            ax.text(width + (abs(width) * 0.01), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', 
                   ha='left', 
                   va='center',
                   fontsize=9)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型比较图已保存至: {save_path}")
        return fig

    def plot_comparison_heatmap(self, results_data, metrics=None, 
                              title='Performance Comparison Heatmap', 
                              save_name='comparison_heatmap.png'):
        """
        绘制性能比较热力图
        
        Args:
            results_data: 包含模型结果的列表或DataFrame
            metrics: 要显示的指标列表，默认显示所有数值指标
            title: 图标题
            save_name: 保存文件名
        """
        # 如果输入是列表，转换为DataFrame
        if isinstance(results_data, list):
            df = pd.DataFrame(results_data)
        else:
            df = results_data
            
        # 如果没有指定指标，选择所有数值列
        if metrics is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除test_loss列，因为我们已经有它了
            metrics = [col for col in numeric_columns if col != 'test_loss']
            # 如果没有其他数值列，添加test_loss
            if not metrics and 'test_loss' in df.columns:
                metrics = ['test_loss']
        else:
            # 确保指定的指标都在数据中
            metrics = [m for m in metrics if m in df.columns]
            
        if not metrics:
            print("没有找到合适的指标用于热力图")
            return
            
        # 准备热力图数据
        if 'model' in df.columns:
            heatmap_data = df[['model'] + metrics].set_index('model')
        elif 'model_name_suffix' in df.columns:
            heatmap_data = df[['model_name_suffix'] + metrics].set_index('model_name_suffix')
        else:
            heatmap_data = df[metrics]
            
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_data) * 0.3)))
        
        # 使用seaborn绘制热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"性能比较热力图已保存至: {save_path}")
        return fig

    def plot_attention_weights(self, attention_weights, tokens=None,
                            title='Attention Weights Visualization',
                            save_name='attention_weights.png'):
        """
        绘制注意力权重图
        
        Args:
            attention_weights: 注意力权重矩阵
            tokens: 对应的token序列
            title: 图标题
            save_name: 保存文件名
        """
        # 转换为numpy数组
        weights = np.array(attention_weights)
        
        # 如果是多维数组，取第一个样本
        if weights.ndim > 2:
            weights = weights[0]
            
        # 如果是二维数组，取第一个维度（通常是查询维度）
        if weights.ndim == 2:
            weights = weights[0]
            
        # 确保是一维数组
        weights = weights.flatten()
        
        # 如果没有提供tokens，创建默认标记
        if tokens is None:
            tokens = [f'Token_{i}' for i in range(len(weights))]
        elif len(tokens) > len(weights):
            tokens = tokens[:len(weights)]
        elif len(tokens) < len(weights):
            tokens.extend([f'Token_{i}' for i in range(len(tokens), len(weights))])
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(max(10, len(weights) * 0.5), 6))
        
        # 创建颜色映射
        im = ax.imshow(weights.reshape(1, -1), cmap='Reds', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Attention'])
        
        # 添加数值标注
        for i, weight in enumerate(weights):
            ax.text(i, 0, f'{weight:.3f}', 
                   ha='center', va='center', 
                   color='white' if weight > np.max(weights) / 2 else 'black',
                   fontsize=8)
        
        ax.set_title(title, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力权重图已保存至: {save_path}")
        return fig

    def plot_cross_dataset_comparison(self, results_df, metric='test_loss',
                                   title='Cross Dataset Performance Comparison',
                                   save_name='cross_dataset_comparison.png'):
        """
        绘制跨数据集性能比较图
        
        Args:
            results_df: 包含模型结果的DataFrame
            metric: 用于比较的指标名称
            title: 图标题
            save_name: 保存文件名
        """
        # 检查必要的列是否存在
        if metric not in results_df.columns:
            raise ValueError(f"指标 '{metric}' 不存在于数据中")
        if 'dataset' not in results_df.columns:
            raise ValueError("数据集中缺少'dataset'列")
        if 'model' not in results_df.columns:
            raise ValueError("数据集中缺少'model'列")
            
        # 只选择数值型列进行处理
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = ['dataset', 'model']
        
        # 确保需要的列存在
        columns_to_include = categorical_columns.copy()
        if metric in numeric_columns:
            columns_to_include.append(metric)
            
        # 过滤数据框只包含需要的列
        filtered_df = results_df[columns_to_include].copy()
            
        # 按数据集和模型分组，只对数值列进行聚合
        if metric in numeric_columns:
            grouped = filtered_df.groupby(['dataset', 'model'])[metric].mean().reset_index()
        else:
            raise ValueError(f"指标 '{metric}' 不是数值型列")
        
        # 检查是否有足够的数据进行绘图
        if len(grouped) == 0:
            print("警告: 没有足够的数据绘制跨数据集性能比较图")
            return None
            
        # 创建透视表用于热力图
        try:
            pivot_table = grouped.pivot_table(index='dataset', columns='model', values=metric, fill_value=0)
        except Exception as e:
            print(f"创建透视表时出错: {e}")
            return None
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用seaborn绘制热力图
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"跨数据集性能比较图已保存至: {save_path}")
        return fig

    def plot_training_history(self, history, title='Training History', 
                             save_name='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, 
                       marker='o', markersize=3, color=self.COLOR_PALETTE['primary'])
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, 
                       marker='s', markersize=3, color=self.COLOR_PALETTE['secondary'])
        self._setup_subplot_style(axes[0, 0], 'Training and Validation Loss', 'Epoch', 'Loss')
        axes[0, 0].legend()
        
        # 学习率变化
        if 'lr' in history and len(history['lr']) > 0:
            axes[0, 1].plot(history['lr'], color=self.COLOR_PALETTE['tertiary'], 
                           linewidth=2, marker='d', markersize=3)
            self._setup_subplot_style(axes[0, 1], 'Learning Rate Schedule', 'Epoch', 'Learning Rate')
        else:
            axes[0, 1].text(0.5, 0.5, 'Learning rate\nnot available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Learning Rate Schedule', fontweight='bold')
        
        # 关键指标 - 先尝试获取训练和验证指标
        metrics_to_plot = ['auc', 'r2', 'mape', 'accuracy', 'f1', 'mse']
        plotted_metrics = 0
        
        for metric in metrics_to_plot:
            if plotted_metrics >= 4:  # 最多绘制4个指标
                break
                
            train_vals = None
            val_vals = None
            
            # 从训练指标历史中查找
            if 'train_metrics' in history and len(history['train_metrics']) > 0:
                # 提取所有epoch中该指标的值
                train_vals = [metrics.get(metric, None) for metrics in history['train_metrics']]
                # 过滤掉None值
                train_vals = [v for v in train_vals if v is not None]
            
            # 从验证指标历史中查找
            if 'val_metrics' in history and len(history['val_metrics']) > 0:
                # 提取所有epoch中该指标的值
                val_vals = [metrics.get(metric, None) for metrics in history['val_metrics']]
                # 过滤掉None值
                val_vals = [v for v in val_vals if v is not None]
            
            # 如果找到了至少一个指标，则绘制
            if (train_vals and len(train_vals) > 0) or (val_vals and len(val_vals) > 0):
                row = (plotted_metrics + 2) // 3  # 转换为subplot的行索引 (从第2行开始)
                col = (plotted_metrics + 2) % 3   # 转换为subplot的列索引
                
                # 确保x轴长度一致
                max_epochs = max(len(train_vals) if train_vals else 0, 
                                len(val_vals) if val_vals else 0)
                epochs = list(range(1, max_epochs + 1))
                
                if train_vals and len(train_vals) > 0:
                    # 补齐长度
                    if len(train_vals) < max_epochs:
                        train_vals.extend([train_vals[-1]] * (max_epochs - len(train_vals)))
                    axes[row, col].plot(epochs, train_vals, label=f'Train {metric.upper()}', 
                                       linewidth=2, marker='o', markersize=3,
                                       color=self.COLOR_PALETTE['primary'])
                if val_vals and len(val_vals) > 0:
                    # 补齐长度
                    if len(val_vals) < max_epochs:
                        val_vals.extend([val_vals[-1]] * (max_epochs - len(val_vals)))
                    axes[row, col].plot(epochs, val_vals, label=f'Val {metric.upper()}', 
                                       linewidth=2, marker='s', markersize=3,
                                       color=self.COLOR_PALETTE['secondary'])
                
                # 设置子图样式
                ylabel = metric.upper()
                if metric == 'mape':
                    ylabel = 'MAPE (%)'
                self._setup_subplot_style(axes[row, col], f'{metric.upper()} Metrics', 'Epoch', ylabel)
                axes[row, col].legend()
                
                plotted_metrics += 1
        
        # 隐藏未使用的子图
        total_subplots = 6
        for i in range(plotted_metrics + 2, total_subplots):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self._get_unique_save_path(save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练历史图已保存至: {save_path}")