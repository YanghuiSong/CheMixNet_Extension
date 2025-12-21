"""化学工具"""
import warnings

# 尝试导入RDKit模块，如果失败则设置为None
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
    rdkit_available = True
except ImportError:
    Chem = None
    AllChem = None
    Descriptors = None
    rdMolDescriptors = None
    Draw = None
    rdkit_available = False
    warnings.warn("RDKit is not available. Some chemical processing features will be disabled.")

# 尝试导入RDKit的高级绘图功能
try:
    from rdkit.Chem.Draw import SimilarityMaps
    similarity_maps_available = True
except ImportError:
    SimilarityMaps = None
    similarity_maps_available = False
    if rdkit_available:
        warnings.warn("RDKit SimilarityMaps not available. Some advanced visualization features will be disabled.")

import numpy as np
from typing import List, Dict, Any, Optional

def validate_smiles(smiles: str) -> bool:
    """验证SMILES字符串是否有效"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot validate SMILES strings.")
        return True  # 如果没有RDKit，则跳过验证
    
    if not isinstance(smiles, str):
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def canonicalize_smiles(smiles: str) -> str:
    """规范化SMILES字符串"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot canonicalize SMILES strings.")
        return smiles  # 如果没有RDKit，则返回原字符串
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)

def get_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """计算分子描述符"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot compute molecular descriptors.")
        return {}
    
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
    descriptors['mr'] = Descriptors.MolMR(mol)  # 摩尔折射率
    
    # 环信息
    ring_info = mol.GetRingInfo()
    descriptors['num_rings'] = ring_info.NumRings()
    
    # 电荷信息
    descriptors['formal_charge'] = Chem.GetFormalCharge(mol)
    
    # 芳香性
    descriptors['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
    descriptors['num_aliphatic_rings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    
    # 杂原子计数
    descriptors['num_nitrogen'] = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7])
    descriptors['num_oxygen'] = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8])
    descriptors['num_sulfur'] = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16])
    descriptors['num_halogen'] = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]])
    
    return descriptors

def generate_fingerprints(mol: Chem.Mol, fp_type: str = 'maccs', 
                         n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """生成分子指纹"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot generate fingerprints.")
        return np.zeros(n_bits if fp_type == 'morgan' else 167)
    
    if mol is None:
        return np.zeros(n_bits if fp_type == 'morgan' else 167)
    
    if fp_type == 'maccs':
        fp = Chem.MACCSkeys.GenMACCSKeys(mol)
        bits = fp.ToBitString()
        # MACCS指纹是167位，但第一位总是1（忽略）
        return np.array([int(b) for b in bits[1:]])
    
    elif fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array([int(b) for b in fp.ToBitString()])
    
    elif fp_type == 'rdkit':
        fp = Chem.RDKFingerprint(mol)
        return np.array([int(b) for b in fp.ToBitString()])
    
    elif fp_type == 'atom_pair':
        fp = Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array([int(b) for b in fp.ToBitString()])
    
    elif fp_type == 'topological_torsion':
        fp = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array([int(b) for b in fp.ToBitString()])
    
    else:
        raise ValueError(f"不支持的指纹类型: {fp_type}")

def calculate_similarity(fp1: np.ndarray, fp2: np.ndarray, 
                        metric: str = 'tanimoto') -> float:
    """计算指纹相似度"""
    if metric == 'tanimoto':
        # Tanimoto系数（Jaccard相似度）
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0
    
    elif metric == 'dice':
        # Dice系数
        intersection = np.sum(fp1 & fp2)
        return 2 * intersection / (np.sum(fp1) + np.sum(fp2))
    
    elif metric == 'cosine':
        # 余弦相似度
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    else:
        raise ValueError(f"不支持的相似度度量: {metric}")

def draw_molecule(mol: Chem.Mol, size: tuple = (300, 300), 
                 highlight_atoms: List[int] = None,
                 highlight_bonds: List[int] = None,
                 legend: str = '') -> np.ndarray:
    """绘制分子结构图"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot draw molecules.")
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    if mol is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    img = Draw.MolToImage(mol, size=size, 
                         highlightAtoms=highlight_atoms,
                         highlightBonds=highlight_bonds)
    
    if legend:
        # 添加图例（需要PIL）
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        # 这里可以添加文本，但需要字体文件
    
    return np.array(img)

def find_similar_molecules(query_smiles: str, smiles_list: List[str], 
                          top_k: int = 10, fp_type: str = 'morgan') -> List[Dict]:
    """在列表中查找相似分子"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot find similar molecules.")
        return []
    
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return []
    
    query_fp = generate_fingerprints(query_mol, fp_type)
    
    similarities = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        target_fp = generate_fingerprints(mol, fp_type)
        similarity = calculate_similarity(query_fp, target_fp)
        
        similarities.append({
            'index': i,
            'smiles': smiles,
            'similarity': similarity,
            'mol': mol
        })
    
    # 按相似度排序
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_k]

def analyze_scaffold(mol: Chem.Mol) -> Dict[str, Any]:
    """分析分子骨架"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot analyze molecular scaffold.")
        return {
            'scaffold_smiles': '',
            'scaffold_num_atoms': 0,
            'scaffold_num_rings': 0,
            'scaffold_mol': None
        }
        
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    
    return {
        'scaffold_smiles': Chem.MolToSmiles(scaffold) if scaffold else '',
        'scaffold_num_atoms': scaffold.GetNumAtoms() if scaffold else 0,
        'scaffold_num_rings': scaffold.GetRingInfo().NumRings() if scaffold else 0,
        'scaffold_mol': scaffold
    }

def calculate_3d_coordinates(mol: Chem.Mol, force_field: str = 'MMFF') -> Chem.Mol:
    """计算分子的3D坐标"""
    if not rdkit_available:
        warnings.warn("RDKit is not available. Cannot calculate 3D coordinates.")
        return mol
    
    mol_3d = Chem.AddHs(mol)
    
    if force_field == 'MMFF':
        success = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if success == 0:  # 嵌入成功
            AllChem.MMFFOptimizeMolecule(mol_3d)
    elif force_field == 'ETKDG':
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol_3d, params)
    else:
        raise ValueError(f"不支持的力场: {force_field}")
    
    return Chem.RemoveHs(mol_3d)