"""自定义PyTorch数据集"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem

def collate_fn(batch):
    # 检查是否是字典格式
    if isinstance(batch[0], dict):
        # 注释掉调试打印语句，避免输出过于冗长
        # print(f"Collate function called with batch of size: {len(batch)}")
        # print("处理字典格式的batch")
        
        collated_batch = {}
        for key in batch[0].keys():
            if key == 'smiles':
                # Stack SMILES tensors
                collated_batch[key] = torch.stack([item[key] for item in batch])
            elif key in ['fingerprint', 'target']:
                # Stack fingerprint and target tensors
                collated_batch[key] = torch.stack([item[key] for item in batch])
            elif key == 'original_smiles':
                # Keep original SMILES as list
                collated_batch[key] = [item[key] for item in batch]
            elif key == 'graph_data':
                # Keep graph data as list
                collated_batch[key] = [item[key] for item in batch]
            else:
                # Handle any other keys by stacking
                try:
                    collated_batch[key] = torch.stack([item[key] for item in batch])
                except:
                    # If stacking fails, keep as list
                    collated_batch[key] = [item[key] for item in batch]
        
        # 注释掉调试打印语句，避免输出过于冗长
        # print(f"Collated batch keys: {collated_batch.keys()}")
        # for k, v in collated_batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor with shape {v.shape}")
        #     else:
        #         print(f"  {k}: <class '{type(v).__name__}'> with length {len(v) if hasattr(v, '__len__') else 'N/A'}")
        
        return collated_batch
    
    # Handle non-dictionary format (fallback)
    return default_collate(batch)

class MolecularDataset(Dataset):
    """分子数据集类"""
    
    def __init__(self, df, featurizer, task_type='regression', 
                 target_col=None, use_graph=False):
        """
        初始化数据集
        
        Args:
            df: 包含SMILES和标签的DataFrame
            featurizer: 特征生成器
            task_type: 任务类型 ('regression' 或 'classification')
            target_col: 目标列名
            use_graph: 是否使用图数据
        """
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer
        self.task_type = task_type
        self.target_col = target_col
        self.use_graph = use_graph
        
        # 确保SMILES列存在
        if 'smiles' not in self.df.columns and 'mol' in self.df.columns:
            self.df['smiles'] = self.df['mol'].apply(lambda x: Chem.MolToSmiles(x) if x else '')
        
        # 提取目标值
        if target_col and target_col in self.df.columns:
            self.targets = self.df[target_col].values
        else:
            self.targets = np.zeros(len(self.df))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 确保idx在有效范围内
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.df)})")
            
        smiles = str(self.df.loc[idx, 'smiles'])
        target = self.targets[idx]
        
        # 生成SMILES特征，增加错误处理
        try:
            smiles_onehot = self.featurizer.smiles_to_onehot(smiles)
        except Exception as e:
            print(f"SMILES特征生成错误 (索引 {idx}): {e}")
            # 确保featurizer有正确的vocab_size
            if not hasattr(self.featurizer, 'vocab_size') or self.featurizer.vocab_size <= 0:
                # 创建一个最小的one-hot向量
                smiles_onehot = np.zeros((self.featurizer.max_len, 1))
            else:
                smiles_onehot = np.zeros((self.featurizer.max_len, self.featurizer.vocab_size))
        
        # 生成指纹特征
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                maccs_fp = self.featurizer.get_maccs_fingerprint(mol)
            except Exception as e:
                print(f"生成MACCS指纹时出错 (索引 {idx}): {e}")
                # 返回默认的零向量
                maccs_fp = np.zeros(167)  # 确保使用正确的维度
        else:
            # 如果无法生成分子对象，返回零向量
            maccs_fp = np.zeros(167)  # 确保使用正确的维度
        
        # 转换为张量
        try:
            smiles_tensor = torch.FloatTensor(smiles_onehot)
        except Exception as e:
            print(f"SMILES张量转换错误 (索引 {idx}): {e}")
            smiles_tensor = torch.FloatTensor(1)  # 返回最小的张量
            
        try:
            fp_tensor = torch.FloatTensor(maccs_fp)
        except Exception as e:
            print(f"指纹张量转换错误 (索引 {idx}): {e}")
            fp_tensor = torch.FloatTensor(167)  # 确保使用正确的维度
            
        # 目标值处理
        try:
            if self.task_type == 'regression':
                target_tensor = torch.FloatTensor([float(target)])
            else:
                target_tensor = torch.FloatTensor([float(target)])
        except Exception as e:
            print(f"目标值转换错误 (索引 {idx}): {e}")
            target_tensor = torch.FloatTensor([0.0])
        
        result = {
            'smiles': smiles_tensor,
            'fingerprint': fp_tensor,
            'target': target_tensor,
            'original_smiles': smiles
        }
        
        # 添加图数据（如果启用）
        if self.use_graph and mol is not None:
            try:
                # 生成图数据
                graph_data = get_mol_graph(smiles)
                result['graph_data'] = graph_data
            except Exception as e:
                # 注释掉详细的错误信息打印，避免输出过于冗长
                # print(f"生成图数据时出错 (索引 {idx}): {e}")
                # 即使图数据生成失败，也要确保字典中有这个键
                result['graph_data'] = None
        elif self.use_graph:
            result['graph_data'] = None
        
        return result
