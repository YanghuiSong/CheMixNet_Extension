"""多模态特征生成"""
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import warnings
import torch

# 尝试导入RDKit模块，如果失败则设置为None
try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    rdkit_available = True
except ImportError:
    Chem = None
    MACCSkeys = None
    rdkit_available = False
    warnings.warn("RDKit is not available. Some molecular fingerprint features will be disabled.")

try:
    from rdkit.Chem import rdmolops
    rdmolops_available = True
except ImportError:
    rdmolops = None
    rdmolops_available = False

class MolecularFeaturizer:
    """分子特征化工具"""
    
    def __init__(self, max_len: int = 100):
        self.max_len = max_len
        self.char_dict = {}
        self.unknown_token = 0
    
    def build_char_vocab(self, smiles_list: List[str]):
        """构建字符词典"""
        # 收集所有字符
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(smiles)
        
        # 构建字符到索引的映射
        self.char_dict = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
        self.char_dict['<UNK>'] = self.unknown_token
        self.vocab_size = len(self.char_dict)  # 确保 vocab_size 正确设置
    
    def encode_smiles(self, smiles: str) -> np.ndarray:
        """编码单个SMILES字符串"""
        # 确保输入是字符串类型
        smiles = str(smiles) if smiles is not None else ""
        encoded = np.zeros(self.max_len, dtype=int)
        for i, char in enumerate(smiles[:self.max_len]):
            # 增加安全检查，确保char_dict存在且不为空
            if hasattr(self, 'char_dict') and self.char_dict:
                encoded[i] = self.char_dict.get(char, self.unknown_token)
            else:
                encoded[i] = self.unknown_token
        return encoded
    
    def encode_smiles_batch(self, smiles_list: List[str]) -> np.ndarray:
        """批量编码SMILES字符串"""
        return np.array([self.encode_smiles(smiles) for smiles in smiles_list])
    
    def smiles_to_onehot(self, smiles: str) -> np.ndarray:
        """将SMILES字符串转换为one-hot编码"""
        # 先编码为索引
        encoded = self.encode_smiles(smiles)
        
        # 确保词汇表大小大于0，避免索引错误
        if not hasattr(self, 'vocab_size') or self.vocab_size <= 0:
            # 如果词汇表未建立或为空，返回零向量
            print(f"警告: 词汇表大小为 {getattr(self, 'vocab_size', 'N/A')}，返回最小维度的one-hot向量")
            return np.zeros((self.max_len, 1))
        
        # 转换为one-hot编码，增加安全检查
        try:
            # 确保encoded中的索引不超过vocab_size
            clipped_encoded = np.clip(encoded, 0, self.vocab_size - 1)
            onehot = np.eye(self.vocab_size)[clipped_encoded]
            return onehot
        except (IndexError, ValueError) as e:
            print(f"One-hot编码时发生索引错误: {e}")
            print(f"  词汇表大小: {self.vocab_size}")
            print(f"  编码序列长度: {len(encoded)}")
            print(f"  编码序列内容: {encoded[:10]}...")  # 只打印前10个元素
            # 出错时返回零向量
            return np.zeros((self.max_len, self.vocab_size))

    def smiles_to_onehot_batch(self, smiles_list: List[str]) -> np.ndarray:
        """批量将SMILES字符串转换为one-hot编码"""
        return np.array([self.smiles_to_onehot(smiles) for smiles in smiles_list])
    
    def get_maccs_fingerprint(self, smiles: str) -> np.ndarray:
        """获取MACCS指纹"""
        if not rdkit_available:
            warnings.warn("RDKit is not available. Cannot compute MACCS fingerprints.")
            # 返回一个默认的167维向量
            return np.zeros(167)
        
        from rdkit import Chem
        # 确保smiles是字符串类型
        if isinstance(smiles, Chem.Mol):
            mol = smiles
        else:
            mol = Chem.MolFromSmiles(str(smiles))
            
        if mol is None:
            return np.zeros(167)
        
        try:
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            # 转换为numpy数组
            arr = np.array(list(fingerprint.ToBitString()), dtype=int)
            # 确保返回167维向量
            if len(arr) < 167:
                padded = np.zeros(167, dtype=int)
                padded[:len(arr)] = arr
                return padded
            elif len(arr) > 167:
                return arr[:167]
            return arr
        except Exception as e:
            print(f"生成MACCS指纹时出错: {e}")
            return np.zeros(167)
    
    def get_morgan_fingerprint(self, smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """获取Morgan指纹"""
        if not rdkit_available:
            warnings.warn("RDKit is not available. Cannot compute Morgan fingerprints.")
            # 返回一个默认的向量
            return np.zeros(n_bits)
        
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        # 转换为numpy数组
        return np.array(list(fingerprint.ToBitString()), dtype=int)
    
    def add_molecular_features(self, df):
        """
        为数据框添加分子特征
        
        Args:
            df: 包含SMILES列的DataFrame
            
        Returns:
            添加了分子特征的DataFrame
        """
        # 生成分子指纹
        fingerprints = []
        valid_smiles = []  # 存储有效的SMILES
        invalid_indices = []  # 存储无效SMILES的索引
        
        for idx, smiles in enumerate(df['smiles']):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 生成MACCS指纹
                    fingerprint = MACCSkeys.GenMACCSKeys(mol)
                    fingerprints.append(fingerprint.ToList())  # 转换为列表格式
                    valid_smiles.append(smiles)
                else:
                    invalid_indices.append(idx)
            except:
                invalid_indices.append(idx)
        
        # 输出统计信息
        print(f"总SMILES数: {len(df)}")
        print(f"有效SMILES数: {len(valid_smiles)}")
        print(f"无效SMILES数: {len(invalid_indices)}")
        
        if len(invalid_indices) > 0:
            print(f"无效SMILES索引: {invalid_indices[:10]}...")  # 只显示前10个
        
        # 为DataFrame添加指纹特征列
        df_copy = df.copy()
        df_copy['maccs_fingerprint'] = fingerprints
        
        return df_copy
    
    def get_mol_graph(self, mol):
        """生成分子图表示（用于GNN）"""
        from rdkit.Chem import rdmolops
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic()
            ]
            atom_features.append(features)
        
        # 邻接矩阵
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        
        # 边特征
        bond_features = []
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # 无向图
            
            bond_type = bond.GetBondType()
            bond_feat = [
                bond_type == Chem.rdchem.BondType.SINGLE,
                bond_type == Chem.rdchem.BondType.DOUBLE,
                bond_type == Chem.rdchem.BondType.TRIPLE,
                bond_type == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            bond_features.append(bond_feat)
            bond_features.append(bond_feat)  # 双向
        
        # 创建一个简单的图对象，具有.x, .edge_index等属性
        class GraphData:
            def __init__(self, x, edge_index):
                self.x = x  # 节点特征
                self.edge_index = edge_index  # 边索引
        
        # 转换为张量
        x = torch.FloatTensor(atom_features)
        edge_index = torch.LongTensor(edge_indices).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        
        return GraphData(x, edge_index)
