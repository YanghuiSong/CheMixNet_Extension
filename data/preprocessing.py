"""统一数据预处理模块"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')

# 数据集元信息
DATASET_INFO = {
    'bace': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'Class'},
    'BBBP': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'p_np'},
    'esol': {'task': 'regression', 'metric': 'rmse', 'target_col': 'measured log solubility in mols per litre'},
    'HIV': {'task': 'classification', 'metric': 'roc_auc', 'target_col': 'HIV_active'},
    'lipophilicity': {'task': 'regression', 'metric': 'rmse', 'target_col': 'exp'}
}

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        # 从配置中提取数据集名称
        if isinstance(config, dict) and 'dataset_name' in config:
            self.dataset_name = config['dataset_name']
        elif isinstance(config, str):
            self.dataset_name = config
        else:
            raise ValueError(f"无效的配置类型: {type(config)}")
        self.smiles_col = self._get_smiles_column()
        
    def _get_smiles_column(self):
        """获取SMILES列名"""
        # dataset_name现在是一个字符串
        dataset_name = self.dataset_name
            
        smiles_columns = {
            'bace': 'mol',
            'BBBP': 'smiles',
            'esol': 'smiles',
            'HIV': 'smiles',
            'lipophilicity': 'smiles'
        }
        return smiles_columns.get(dataset_name, 'smiles')
    
    def load_and_clean(self, filepath):
        """加载并清洗数据"""
        df = pd.read_csv(filepath)
        
        # 确保SMILES列存在
        if self.smiles_col not in df.columns:
            raise ValueError(f"SMILES列 '{self.smiles_col}' 不存在于数据集中")
        
        # 移除无效SMILES
        original_len = len(df)
        df['mol'] = df[self.smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)))
        df = df[df['mol'].notnull()].copy()
        cleaned_len = len(df)
        
        print(f"数据集: {self.dataset_name}")
        print(f"原始样本数: {original_len}, 有效样本数: {cleaned_len}")
        print(f"移除无效SMILES: {original_len - cleaned_len} 个")
        
        # 确保存在'smiles'列
        if 'smiles' not in df.columns:
            if self.smiles_col == 'mol':
                # 从mol对象生成smiles列
                df['smiles'] = df['mol'].apply(lambda x: Chem.MolToSmiles(x) if x else '')
            else:
                # 复制smiles列
                df['smiles'] = df[self.smiles_col]
        
        return df
    
    def add_molecular_features(self, df):
        """添加分子描述符和指纹作为额外特征"""
        from rdkit.Chem import MACCSkeys
        
        # 基本描述符
        df['mol_weight'] = df['mol'].apply(Descriptors.MolWt)
        df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
        df['num_bonds'] = df['mol'].apply(lambda x: x.GetNumBonds())
        df['num_rings'] = df['mol'].apply(lambda x: x.GetRingInfo().NumRings())
        
        # 添加MACCS指纹
        maccs_fps = []
        for mol in df['mol']:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                maccs_fps.append(np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0'))
            except:
                maccs_fps.append(np.zeros(166))  # MACCS指纹是166位
                
        df['maccs_fingerprint'] = maccs_fps
        
        return df
    
    def get_task_info(self, dataset_name=None):
        """获取任务信息"""
        if dataset_name is None:
            dataset_name = self.dataset_name
        if isinstance(dataset_name, dict):
            dataset_name = list(dataset_name.keys())[0]
        return DATASET_INFO.get(dataset_name, {})
    
    def split_data(self, df, target_col, val_ratio=0.1, test_ratio=0.2, task_type='regression'):
        """划分数据集"""
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 判断是否使用分层抽样
        use_stratify = task_type == 'classification' and y.nunique() >= 2
        
        # 先划分训练集和临时集（验证+测试）
        stratify_param = y if use_stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=42, stratify=stratify_param
        )
        
        # 再将临时集划分为验证集和测试集
        stratify_param_temp = y_temp if use_stratify else None
        temp_val_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=temp_val_ratio, random_state=42, stratify=stratify_param_temp
        )
        
        # 重建数据框
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        return train_df, val_df, test_df
    
    def split_features(self, features, targets, task_type='regression', test_size=0.2, val_size=0.1):
        """
        划分特征数据集
        
        Args:
            features: 特征数组
            targets: 目标值数组
            task_type: 任务类型
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            train_features, val_features, test_features, train_targets, val_targets, test_targets
        """
        from sklearn.model_selection import train_test_split
        
        # 第一次划分：分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, targets, 
            test_size=test_size, 
            random_state=42,
            stratify=targets if task_type == 'classification' and len(targets) >= 2 else None
        )
        
        # 第二次划分：从剩余数据中分离出验证集
        val_ratio = val_size / (1 - test_size)  # 调整验证集比例
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=42,
            stratify=y_temp if task_type == 'classification' and len(y_temp) >= 2 else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
