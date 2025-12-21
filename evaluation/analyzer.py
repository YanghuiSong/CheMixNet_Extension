"""结果分析工具"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from pathlib import Path

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir='./results'):
        self.results_dir = Path(results_dir)
        self.results = {}
    
    def load_results(self, experiment_name, results_file='results.csv'):
        """加载实验结果"""
        results_path = self.results_dir / experiment_name / results_file
        if results_path.exists():
            df = pd.read_csv(results_path)
            self.results[experiment_name] = df
            return df
        else:
            print(f"结果文件不存在: {results_path}")
            return None
    
    def compare_experiments(self, experiment_names, metric='auc'):
        """比较多个实验的结果"""
        comparison = {}
        
        for exp_name in experiment_names:
            if exp_name in self.results:
                df = self.results[exp_name]
                if metric in df.columns:
                    comparison[exp_name] = {
                        'mean': df[metric].mean(),
                        'std': df[metric].std(),
                        'min': df[metric].min(),
                        'max': df[metric].max(),
                        'median': df[metric].median()
                    }
        
        comparison_df = pd.DataFrame(comparison).T
        return comparison_df
    
    def perform_statistical_test(self, experiment1, experiment2, 
                                metric='auc', test_type='t_test'):
        """执行统计检验"""
        if experiment1 not in self.results or experiment2 not in self.results:
            print("实验数据不存在")
            return None
            
        df1 = self.results[experiment1][metric].dropna()
        df2 = self.results[experiment2][metric].dropna()
        
        if test_type == 't_test':
            stat, p_value = stats.ttest_ind(df1, df2)
            test_name = "独立样本t检验"
        elif test_type == 'wilcoxon':
            if len(df1) != len(df2):
                print("Wilcoxon要求样本量相同，改用Mann-Whitney U检验")
                test_type = 'mannwhitney'
            else:
                stat, p_value = stats.wilcoxon(df1, df2)
                test_name = "Wilcoxon符号秩检验"
        elif test_type == 'mannwhitney':
            stat, p_value = stats.mannwhitneyu(df1, df2)
            test_name = "Mann-Whitney U检验"
        else:
            print(f"不支持的检验类型: {test_type}")
            return None
        
        return {
            'test_name': test_name,
            'experiment1': experiment1,
            'experiment2': experiment2,
            'metric': metric,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean1': df1.mean(),
            'mean2': df2.mean(),
            'std1': df1.std(),
            'std2': df2.std()
        }
    
    def analyze_error_patterns(self, predictions, targets, smiles_list=None):
        """分析错误模式"""
        errors = np.abs(np.array(predictions).flatten() - np.array(targets).flatten())
        
        analysis = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'error_distribution': errors
        }
        
        # 找出最大错误的样本
        max_error_idx = np.argmax(errors)
        analysis['max_error_sample'] = {
            'prediction': predictions[max_error_idx],
            'target': targets[max_error_idx],
            'error': errors[max_error_idx]
        }
        
        if smiles_list is not None and len(smiles_list) == len(predictions):
            analysis['max_error_sample']['smiles'] = smiles_list[max_error_idx]
        
        # 错误分类（对于分类任务）
        if len(np.unique(targets)) <= 2:  # 二分类
            pred_binary = (np.array(predictions).flatten() > 0.5).astype(int)
            target_binary = (np.array(targets).flatten() > 0.5).astype(int)
            
            false_positives = np.where((pred_binary == 1) & (target_binary == 0))[0]
            false_negatives = np.where((pred_binary == 0) & (target_binary == 1))[0]
            
            analysis['false_positives'] = false_positives
            analysis['false_negatives'] = false_negatives
            analysis['fp_count'] = len(false_positives)
            analysis['fn_count'] = len(false_negatives)
        
        return analysis
    
    def calculate_correlation(self, predictions, targets, features=None):
        """计算相关性"""
        preds = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        correlations = {
            'pearson': stats.pearsonr(preds, targets)[0] if len(preds) > 1 else 0,
            'spearman': stats.spearmanr(preds, targets)[0] if len(preds) > 1 else 0,
            'r2': 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        }
        
        # 特征相关性（如果提供了特征）
        if features is not None:
            feature_corrs = {}
            for i, feature in enumerate(features.T):
                if len(np.unique(feature)) > 1:  # 避免常数特征
                    corr = stats.pearsonr(feature, errors)[0]
                    feature_corrs[f'feature_{i}'] = corr
            
            correlations['feature_error_correlations'] = feature_corrs
        
        return correlations
    
    def generate_summary_report(self, experiment_results):
        """生成摘要报告"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_count': len(experiment_results),
            'summary': {}
        }
        
        for exp_name, metrics in experiment_results.items():
            report['summary'][exp_name] = {
                'best_model': metrics.get('best_model', 'N/A'),
                'best_score': metrics.get('best_score', 0),
                'training_time': metrics.get('training_time', 0),
                'parameters': metrics.get('parameters', 0)
            }
        
        return report