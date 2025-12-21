"""模型解释工具"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from typing import List, Dict, Any, Optional
import warnings

# 检查RDKit可用性
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import SimilarityMaps
    rdkit_available = True
    similarity_maps_available = hasattr(SimilarityMaps, 'GetSimilarityMapFromWeights')
except ImportError:
    rdkit_available = False
    similarity_maps_available = False
    warnings.warn("RDKit is not installed. Some visualization features will be disabled.")

class ModelInterpreter:
    """模型解释器"""
    
    def __init__(self, model, featurizer, device='cpu'):
        self.model = model
        self.featurizer = featurizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute_feature_importance(self, smiles_input, fp_input, graph_data=None, 
                                  n_repeats=10, random_state=42):
        """计算特征重要性（基于排列）"""
        from sklearn.metrics import mean_squared_error
        
        try:
            # 确保输入在正确的设备上并需要梯度
            if hasattr(smiles_input, 'to'):
                smiles_input = smiles_input.to(self.device)
            if hasattr(fp_input, 'to'):
                fp_input = fp_input.to(self.device)
            
            # 确保输入张量需要梯度
            if not smiles_input.requires_grad:
                smiles_input = smiles_input.detach().requires_grad_(True)
            if not fp_input.requires_grad:
                fp_input = fp_input.detach().requires_grad_(True)
            
            # 基准性能
            with torch.no_grad():
                # 正确调用模型的forward方法，传递正确的参数数量
                y_pred = self.model(smiles_input, fp_input).cpu().numpy()
                # 创建虚拟目标（用于计算基准分数）
                baseline_target = np.zeros_like(y_pred)
                baseline_score = mean_squared_error(baseline_target, y_pred)
            
            # 分别对SMILES和指纹特征进行重要性计算
            smiles_features_count = smiles_input.shape[1] if smiles_input.dim() == 2 else smiles_input.shape[1]
            fp_features_count = fp_input.shape[1]
            
            importance_scores = np.zeros(smiles_features_count + fp_features_count)
            
            # 对SMILES特征进行排列重要性计算（简化处理）
            for i in range(min(10, smiles_features_count)):  # 限制计算量
                for _ in range(n_repeats):
                    np.random.seed(random_state + _)
                    # 创建扰动输入
                    smiles_perturbed = smiles_input.clone()
                    if smiles_perturbed.dim() == 3:
                        # 对one-hot编码进行扰动
                        perturb_idx = np.random.randint(0, smiles_perturbed.shape[1])
                        smiles_perturbed[:, perturb_idx, :] = torch.rand_like(smiles_perturbed[:, perturb_idx, :])
                    else:
                        # 对索引进行扰动
                        perturb_idx = np.random.randint(0, smiles_perturbed.shape[1])
                        smiles_perturbed[:, perturb_idx] = torch.randint(0, smiles_perturbed.max().int()+1, 
                                                                       (smiles_perturbed.shape[0],))
                    
                    # 确保扰动后的输入也需要梯度
                    if not smiles_perturbed.requires_grad:
                        smiles_perturbed = smiles_perturbed.detach().requires_grad_(True)
                    
                    with torch.no_grad():
                        # 正确调用模型的forward方法，传递正确的参数数量
                        y_pred_perm = self.model(smiles_perturbed, fp_input).cpu().numpy()
                        perm_score = mean_squared_error(baseline_target, y_pred_perm)
                    
                    importance_scores[i] += (perm_score - baseline_score)
                
                importance_scores[i] /= n_repeats
            
            # 对指纹特征进行排列重要性计算（简化处理）
            for i in range(min(10, fp_features_count)):
                feature_idx = smiles_features_count + i
                for _ in range(n_repeats):
                    np.random.seed(random_state + _)
                    # 创建扰动输入
                    fp_perturbed = fp_input.clone()
                    # 确保fp_perturbed和fp_input形状相同
                    if fp_perturbed.dim() > 1:
                        fp_perturbed[:, i] = torch.rand_like(fp_perturbed[:, i])
                    else:
                        fp_perturbed[i] = torch.rand_like(fp_perturbed[i])
                    
                    # 确保扰动后的输入也需要梯度
                    if not fp_perturbed.requires_grad:
                        fp_perturbed = fp_perturbed.detach().requires_grad_(True)
                    
                    with torch.no_grad():
                        # 确保图数据格式正确
                        if graph_data is not None:
                            if isinstance(graph_data, list) and len(graph_data) > 0:
                                # 如果graph_data是列表，取第一个元素
                                graph_input = graph_data[0]
                            else:
                                graph_input = graph_data
                        else:
                            graph_input = None
                        
                        try:
                            y_pred_perm = self.model(smiles_input, fp_perturbed, graph_input).cpu().numpy()
                        except Exception as e:
                            print(f"模型推理错误: {e}")
                            # 回退到不使用图数据
                            y_pred_perm = self.model(smiles_input, fp_perturbed).cpu().numpy()
                        perm_score = mean_squared_error(baseline_target, y_pred_perm)
                    
                    importance_scores[feature_idx] += (perm_score - baseline_score)
                
                importance_scores[feature_idx] /= n_repeats
            
            # 创建结果列表而不是字典
            feature_names = ([f'SMILES_Feature_{i}' for i in range(min(10, smiles_features_count))] + 
                            [f'FP_Feature_{i}' for i in range(min(10, fp_features_count))])
            
            # 返回重要性分数数组和特征名称
            return np.array(importance_scores[:len(feature_names)]), feature_names
            
        except Exception as e:
            print(f"计算特征重要性时发生错误: {e}")
            return None, None
    
    def visualize_feature_importance(self, importance_scores, feature_names, top_n=20,
                                   title='Feature Importance', 
                                   save_name='feature_importance.png'):
        """可视化特征重要性"""
        if importance_scores is None or len(importance_scores) == 0:
            print("没有特征重要性数据可供可视化")
            return
            
        # 获取前N个最重要的特征
        top_indices = np.argsort(np.abs(importance_scores))[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = [importance_scores[i] for i in top_indices]
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if x < 0 else 'blue' for x in top_importance]
        bars = plt.barh(range(len(top_features)), top_importance, color=colors)
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score (Increase in MSE)')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, top_importance)):
            plt.text(imp + (0.01 if imp >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{imp:.4f}', va='center',
                   color='black', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()  # 避免显示图形
    
    def compute_shap_values(self, X, background_samples=100):
        """计算SHAP值"""
        # 创建背景数据
        if background_samples < len(X):
            background = X[np.random.choice(len(X), background_samples, replace=False)]
        else:
            background = X
        
        # 创建SHAP解释器
        def model_wrapper(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                return self.model(x_tensor).cpu().numpy()
        
        explainer = shap.KernelExplainer(model_wrapper, background)
        
        # 计算SHAP值（为了速度，只计算部分样本）
        n_shap_samples = min(100, len(X))
        shap_samples = X[np.random.choice(len(X), n_shap_samples, replace=False)]
        
        shap_values = explainer.shap_values(shap_samples)
        
        return shap_values, shap_samples
    
    def visualize_shap_summary(self, shap_values, feature_names=None,
                             save_name='shap_summary.png'):
        """可视化SHAP摘要图"""
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_molecular_substructures(self, smiles_list, predictions,
                                       n_examples=5):
        """分析重要分子子结构"""
        results = []
        
        for i in range(min(n_examples, len(smiles_list))):
            smiles = smiles_list[i]
            pred = predictions[i]
            
            mol = Chem.MolFromSmiles(smiles)
            
            # 分析子结构（简化版）
            substructures = {
                'aromatic_rings': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ccccc1'))),
                'hydroxyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
                'amine_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2]'))),
                'carbonyl_groups': len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=O'))),
                'halogens': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[F,Cl,Br,I]'))),
            }
            
            results.append({
                'smiles': smiles,
                'prediction': pred,
                'substructures': substructures,
                'mol': mol
            })
        
        return results
    
    def visualize_molecule_with_importance(self, mol, atom_importance=None,
                                         bond_importance=None,
                                         title='Molecule with Importance',
                                         save_name='molecule_importance.png'):
        """可视化分子及其重要性"""
        if not rdkit_available:
            warnings.warn("RDKit is not available. Cannot visualize molecules.")
            return None
            
        if atom_importance is not None and similarity_maps_available:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 创建相似性地图
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, atom_importance, colorMap='bwr', 
                contourLines=10, size=(300, 300), ax=ax
            )
            
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.show()
        
        elif Draw is not None:
            # 绘制普通分子
            img = Draw.MolToImage(mol, size=(400, 300))
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            warnings.warn("RDKit Draw module is not available. Cannot visualize molecules.")
    
    def generate_interpretation_report(self, X, y, feature_names=None,
                                      smiles_list=None, save_path='interpretation_report.txt'):
        """生成完整的解释报告"""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("MODEL INTERPRETATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of samples: {len(X)}")
        report_lines.append(f"Number of features: {X.shape[1]}")
        report_lines.append("")
        
        # 特征重要性
        report_lines.append("1. FEATURE IMPORTANCE")
        report_lines.append("-" * 40)
        
        importance = self.compute_feature_importance(X, y, feature_names)
        top_features = list(importance.items())[:10]
        
        for feature, score in top_features:
            report_lines.append(f"{feature:30s}: {score:10.6f}")
        
        report_lines.append("")
        
        # 子结构分析（如果有SMILES）
        if smiles_list is not None:
            report_lines.append("2. SUBSTRUCTURE ANALYSIS")
            report_lines.append("-" * 40)
            
            substructure_results = self.analyze_molecular_substructures(
                smiles_list, y[:5]
            )
            
            for i, result in enumerate(substructure_results):
                report_lines.append(f"Example {i+1}: {result['smiles']}")
                report_lines.append(f"  Prediction: {result['prediction']:.4f}")
                report_lines.append("  Substructure counts:")
                for substruct, count in result['substructures'].items():
                    report_lines.append(f"    {substruct}: {count}")
                report_lines.append("")
        
        # 模型洞察
        report_lines.append("3. MODEL INSIGHTS")
        report_lines.append("-" * 40)
        
        # 分析正负相关的特征
        positive_features = [(f, s) for f, s in importance.items() if s > 0][:5]
        negative_features = [(f, s) for f, s in importance.items() if s < 0][:5]
        
        report_lines.append("Most positively influential features:")
        for feature, score in positive_features:
            report_lines.append(f"  {feature}: +{score:.6f}")
        
        report_lines.append("")
        report_lines.append("Most negatively influential features:")
        for feature, score in negative_features:
            report_lines.append(f"  {feature}: {score:.6f}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"解释报告已保存到: {save_path}")
        
        return report_text