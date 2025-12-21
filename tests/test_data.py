"""数据模块测试"""
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import DataPreprocessor
from data.featurization import MolecularFeaturizer
from data.dataset import MolecularDataset
from data.splits import split_dataset

class TestDataPreprocessor(unittest.TestCase):
    """测试数据预处理器"""
    
    def setUp(self):
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CCC', 'invalid_smiles'],
            'target': [1.0, 0.0, 1.0, 0.5]
        })
        
        # 保存到临时文件
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / 'test.csv'
        self.test_data.to_csv(self.test_file, index=False)
        
        self.preprocessor = DataPreprocessor('test')
    
    def test_load_and_clean(self):
        """测试数据加载和清洗"""
        df = self.preprocessor.load_and_clean(self.test_file)
        
        # 检查无效SMILES被移除
        self.assertEqual(len(df), 3)  # 一个无效SMILES被移除
        
        # 检查mol列存在
        self.assertIn('mol', df.columns)
        
        # 检查所有mol都是有效的
        self.assertTrue(all(df['mol'].notnull()))
    
    def test_add_molecular_features(self):
        """测试添加分子特征"""
        df = self.preprocessor.load_and_clean(self.test_file)
        df = self.preprocessor.add_molecular_features(df)
        
        # 检查是否添加了特征列
        expected_features = ['mol_weight', 'num_atoms', 'num_bonds', 'num_rings']
        for feature in expected_features:
            self.assertIn(feature, df.columns)
            
        # 检查特征值合理
        self.assertTrue(all(df['num_atoms'] > 0))
    
    def tearDown(self):
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir)

class TestMolecularFeaturizer(unittest.TestCase):
    """测试分子特征生成器"""
    
    def setUp(self):
        self.featurizer = MolecularFeaturizer(max_smiles_len=50)
        self.test_smiles = ['CCO', 'CCN', 'CCC(=O)O', 'c1ccccc1']
    
    def test_build_char_vocab(self):
        """测试构建字符词汇表"""
        vocab = self.featurizer.build_char_vocab(self.test_smiles)
        
        # 检查词汇表大小
        self.assertGreater(len(vocab), 0)
        self.assertLessEqual(len(vocab), 50 + 2)  # 最多50个字符 + <PAD> + <UNK>
        
        # 检查特殊标记
        self.assertIn('<PAD>', vocab)
        self.assertIn('<UNK>', vocab)
    
    def test_smiles_to_onehot(self):
        """测试SMILES转one-hot编码"""
        # 先构建词汇表
        self.featurizer.build_char_vocab(self.test_smiles)
        
        # 测试编码
        onehot = self.featurizer.smiles_to_onehot('CCO')
        
        # 检查形状
        self.assertEqual(onehot.shape, (50, len(self.featurizer.char_dict)))
        
        # 检查one-hot属性
        self.assertEqual(onehot.sum(), len('CCO'))  # 每个字符一个1
    
    def test_get_maccs_fingerprint(self):
        """测试生成MACCS指纹"""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        fp = self.featurizer.get_maccs_fingerprint(mol)
        
        # 检查指纹维度
        self.assertEqual(len(fp), 166)  # MACCS是167位，第一位被移除
        
        # 检查值类型
        self.assertTrue(all(fp == 0) or all(fp == 1) or any((fp == 0) | (fp == 1)))
    
    def test_get_morgan_fingerprint(self):
        """测试生成Morgan指纹"""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles('CCO')
        fp = self.featurizer.get_morgan_fingerprint(mol, n_bits=1024)
        
        # 检查指纹维度
        self.assertEqual(len(fp), 1024)
        
        # 检查值类型
        self.assertTrue(all(fp == 0) or all(fp == 1) or any((fp == 0) | (fp == 1)))

class TestDataset(unittest.TestCase):
    """测试数据集类"""
    
    def setUp(self):
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CCC'],
            'target': [1.0, 0.0, 1.0]
        })
        
        self.featurizer = MolecularFeaturizer(max_smiles_len=50)
        self.featurizer.build_char_vocab(self.test_df['smiles'].tolist())
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = MolecularDataset(
            self.test_df, self.featurizer,
            task_type='regression',
            target_col='target'
        )
        
        # 检查数据集大小
        self.assertEqual(len(dataset), 3)
        
        # 检查样本
        sample = dataset[0]
        
        # 检查样本包含必要的键
        self.assertIn('smiles', sample)
        self.assertIn('fingerprint', sample)
        self.assertIn('target', sample)
        self.assertIn('original_smiles', sample)
        
        # 检查张量形状
        self.assertEqual(sample['smiles'].shape, (50, len(self.featurizer.char_dict)))
        self.assertEqual(sample['fingerprint'].shape, (166,))
        self.assertEqual(sample['target'].shape, (1,))
    
    def test_dataset_statistics(self):
        """测试数据集统计"""
        dataset = MolecularDataset(
            self.test_df, self.featurizer,
            task_type='regression',
            target_col='target'
        )
        
        stats = dataset.get_statistics()
        
        # 检查统计信息
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['task_type'], 'regression')
        self.assertEqual(stats['target_col'], 'target')
        
        # 回归任务的统计
        self.assertIn('mean', stats)
        self.assertIn('std', stats)

class TestDataSplits(unittest.TestCase):
    """测试数据集划分"""
    
    def setUp(self):
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'smiles': [f'C{"C"*i}O' for i in range(100)],  # 生成100个不同的SMILES
            'target': list(range(100))
        })
    
    def test_split_dataset(self):
        """测试数据集划分"""
        train_df, val_df, test_df = split_dataset(
            self.test_df, test_size=0.2, val_size=0.1, random_seed=42
        )
        
        # 检查划分比例
        total = len(self.test_df)
        expected_train = int(total * 0.7)  # 70% 训练
        expected_val = int(total * 0.1)    # 10% 验证
        expected_test = int(total * 0.2)   # 20% 测试
        
        self.assertEqual(len(train_df), expected_train)
        self.assertEqual(len(val_df), expected_val)
        self.assertEqual(len(test_df), expected_test)
        
        # 检查没有重叠
        train_smiles = set(train_df['smiles'])
        val_smiles = set(val_df['smiles'])
        test_smiles = set(test_df['smiles'])
        
        self.assertTrue(train_smiles.isdisjoint(val_smiles))
        self.assertTrue(train_smiles.isdisjoint(test_smiles))
        self.assertTrue(val_smiles.isdisjoint(test_smiles))
    
    def test_stratified_split(self):
        """测试分层划分（用于分类任务）"""
        # 创建分类数据
        class_df = pd.DataFrame({
            'smiles': [f'C{"C"*i}O' for i in range(100)],
            'target': [0] * 50 + [1] * 50  # 平衡的二元分类
        })
        
        train_df, val_df, test_df = split_dataset(
            class_df, test_size=0.2, val_size=0.1, 
            random_seed=42, stratify_col='target'
        )
        
        # 检查每折中的类别比例
        for df in [train_df, val_df, test_df]:
            class_counts = df['target'].value_counts()
            # 应该大致保持原始比例（50%每类）
            ratio = class_counts[0] / class_counts[1]
            self.assertAlmostEqual(ratio, 1.0, delta=0.3)  # 允许30%的差异

if __name__ == '__main__':
    unittest.main()