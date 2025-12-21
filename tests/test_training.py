"""训练模块测试"""
import unittest
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import AdvancedTrainer
from training.metrics import MetricsCalculator
from training.callbacks import EarlyStopping, ModelCheckpoint
from models.base_models import MLPBaseline

class MockDataset(torch.utils.data.Dataset):
    """模拟数据集用于测试"""
    def __init__(self, n_samples=100, input_dim=167):
        self.data = torch.randn(n_samples, input_dim)
        self.targets = torch.randn(n_samples, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'fingerprint': self.data[idx],
            'target': self.targets[idx]
        }

class MockDataLoader:
    """模拟数据加载器"""
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx = 0
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        
        batch = {}
        end_idx = min(self.idx + self.batch_size, len(self.dataset))
        
        batch_data = []
        batch_targets = []
        for i in range(self.idx, end_idx):
            item = self.dataset[i]
            batch_data.append(item['fingerprint'])
            batch_targets.append(item['target'])
        
        batch['fingerprint'] = torch.stack(batch_data)
        batch['target'] = torch.stack(batch_targets)
        
        self.idx = end_idx
        return batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class TestMetricsCalculator(unittest.TestCase):
    """测试指标计算器"""
    
    def setUp(self):
        self.metrics_calc = MetricsCalculator()
    
    def test_regression_metrics(self):
        """测试回归指标计算"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = self.metrics_calc.calculate_regression_metrics(y_true, y_pred)
        
        # 检查所有指标都存在
        expected_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # 检查指标值合理
        self.assertGreaterEqual(metrics['r2'], 0.9)  # 预测应该很好
        self.assertLess(metrics['rmse'], 0.2)
        self.assertLess(metrics['mae'], 0.2)
    
    def test_classification_metrics(self):
        """测试分类指标计算"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        metrics = self.metrics_calc.calculate_classification_metrics(
            y_true, y_pred=None, y_prob=y_prob
        )
        
        # 检查所有指标都存在
        expected_metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'specificity']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # 检查指标值合理
        self.assertGreaterEqual(metrics['accuracy'], 0.8)
        self.assertGreaterEqual(metrics['auc'], 0.8)
    
    def test_format_metrics(self):
        """测试指标格式化"""
        metrics = {
            'mse': 0.123456789,
            'rmse': 0.351363,
            'r2': 0.87654321
        }
        
        formatted = self.metrics_calc.format_metrics(metrics, decimal_places=4)
        
        # 检查格式化
        self.assertEqual(formatted['mse'], 0.1235)  # 四舍五入到4位小数
        self.assertEqual(formatted['rmse'], 0.3514)
        self.assertEqual(formatted['r2'], 0.8765)

class TestEarlyStopping(unittest.TestCase):
    """测试早停回调"""
    
    def test_early_stopping_min(self):
        """测试最小化模式的早停"""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 模拟损失值
        losses = [10.0, 9.0, 8.0, 8.5, 8.6, 8.7, 8.8]
        
        should_stop = False
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss)
            if should_stop:
                print(f"在第 {i+1} 轮早停")
                break
        
        # 检查是否在正确的时间停止
        self.assertTrue(should_stop)
        self.assertEqual(early_stopping.counter, 3)
        self.assertEqual(early_stopping.best_score, 8.0)
    
    def test_early_stopping_max(self):
        """测试最大化模式的早停"""
        early_stopping = EarlyStopping(patience=3, mode='max')
        
        # 模拟准确率值
        accuracies = [0.7, 0.8, 0.85, 0.83, 0.82, 0.81, 0.80]
        
        should_stop = False
        for i, acc in enumerate(accuracies):
            should_stop = early_stopping(acc)
            if should_stop:
                print(f"在第 {i+1} 轮早停")
                break
        
        # 检查是否在正确的时间停止
        self.assertTrue(should_stop)
        self.assertEqual(early_stopping.counter, 3)
        self.assertEqual(early_stopping.best_score, 0.85)
    
    def test_early_stopping_reset(self):
        """测试早停重置"""
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        # 第一次序列
        early_stopping(10.0)
        early_stopping(9.0)  # 改善
        early_stopping(9.5)  # 变差
        early_stopping(9.6)  # 变差
        
        self.assertTrue(early_stopping.early_stop)
        
        # 重置
        early_stopping.reset()
        
        self.assertFalse(early_stopping.early_stop)
        self.assertEqual(early_stopping.counter, 0)
        self.assertIsNone(early_stopping.best_score)

class TestModelCheckpoint(unittest.TestCase):
    """测试模型检查点"""
    
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建简单模型
        self.model = MLPBaseline(input_dim=10, output_dim=1)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def test_model_checkpoint_save(self):
        """测试模型检查点保存"""
        checkpoint = ModelCheckpoint(
            save_dir=self.temp_dir,
            filename='test_model.pth',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        # 模拟训练过程
        metrics1 = {'val_loss': 10.0, 'val_accuracy': 0.7}
        metrics2 = {'val_loss': 9.0, 'val_accuracy': 0.8}  # 改善
        metrics3 = {'val_loss': 9.5, 'val_accuracy': 0.75}  # 变差
        
        # 第一次保存（应该保存）
        is_better1 = checkpoint(self.model, epoch=1, metrics=metrics1, 
                               optimizer=self.optimizer)
        self.assertTrue(is_better1)
        
        # 第二次保存（改善，应该保存）
        is_better2 = checkpoint(self.model, epoch=2, metrics=metrics2,
                               optimizer=self.optimizer)
        self.assertTrue(is_better2)
        
        # 第三次保存（变差，不应该保存为最佳）
        is_better3 = checkpoint(self.model, epoch=3, metrics=metrics3,
                               optimizer=self.optimizer)
        self.assertFalse(is_better3)
        
        # 检查文件是否存在
        import os
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_model.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'latest_model.pt')))
    
    def test_model_checkpoint_load(self):
        """测试模型检查点加载"""
        checkpoint = ModelCheckpoint(
            save_dir=self.temp_dir,
            filename='test_model.pth',
            monitor='val_loss',
            mode='min'
        )
        
        # 先保存一个检查点
        metrics = {'val_loss': 10.0, 'val_accuracy': 0.7}
        checkpoint(self.model, epoch=1, metrics=metrics, optimizer=self.optimizer)
        
        # 创建新模型
        new_model = MLPBaseline(input_dim=10, output_dim=1)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # 加载检查点
        loaded_checkpoint = checkpoint.load_best_model(
            new_model, new_optimizer
        )
        
        # 检查加载的检查点
        self.assertIsNotNone(loaded_checkpoint)
        self.assertEqual(loaded_checkpoint['epoch'], 1)
        self.assertEqual(loaded_checkpoint['current_score'], 10.0)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

class TestAdvancedTrainer(unittest.TestCase):
    """测试高级训练器"""
    
    def setUp(self):
        # 创建模拟数据
        self.train_dataset = MockDataset(n_samples=50, input_dim=167)
        self.val_dataset = MockDataset(n_samples=20, input_dim=167)
        
        self.train_loader = MockDataLoader(self.train_dataset, batch_size=4)
        self.val_loader = MockDataLoader(self.val_dataset, batch_size=4)
        
        # 创建模型
        self.model = MLPBaseline(input_dim=167, output_dim=1)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = AdvancedTrainer(
            self.model,
            config={
                'task_type': 'regression',
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 5
            }
        )
        
        # 检查设备
        self.assertIn(trainer.device.type, ['cpu', 'cuda'])
        
        # 检查优化器
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        
        # 检查损失函数
        self.assertIsInstance(trainer.criterion, torch.nn.MSELoss)
        
        # 检查历史记录
        self.assertEqual(len(trainer.history['train_loss']), 0)
        self.assertEqual(len(trainer.history['val_loss']), 0)
    
    def test_train_epoch(self):
        """测试训练一个epoch"""
        trainer = AdvancedTrainer(
            self.model,
            config={'task_type': 'regression'}
        )
        
        train_loss, train_metrics = trainer.train_epoch(self.train_loader)
        
        # 检查损失值
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0)
        
        # 检查指标
        self.assertIsInstance(train_metrics, dict)
        self.assertIn('mse', train_metrics)
        self.assertIn('rmse', train_metrics)
        self.assertIn('mae', train_metrics)
        self.assertIn('r2', train_metrics)
    
    def test_validate(self):
        """测试验证"""
        trainer = AdvancedTrainer(
            self.model,
            config={'task_type': 'regression'}
        )
        
        val_loss, val_metrics = trainer.validate(self.val_loader)
        
        # 检查损失值
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0)
        
        # 检查指标
        self.assertIsInstance(val_metrics, dict)
    
    def test_calculate_metrics_regression(self):
        """测试回归指标计算"""
        trainer = AdvancedTrainer(
            self.model,
            config={'task_type': 'regression'}
        )
        
        preds = [1.0, 2.0, 3.0]
        targets = [1.1, 1.9, 3.1]
        
        metrics = trainer.calculate_metrics(preds, targets)
        
        # 检查回归指标
        expected_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_calculate_metrics_classification(self):
        """测试分类指标计算"""
        trainer = AdvancedTrainer(
            self.model,
            config={'task_type': 'classification'}
        )
        
        preds = [0.1, 0.9, 0.2, 0.8]
        targets = [0, 1, 0, 1]
        
        metrics = trainer.calculate_metrics(preds, targets)
        
        # 检查分类指标
        expected_metrics = ['accuracy', 'f1', 'precision', 'recall']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_save_checkpoint(self):
        """测试保存检查点"""
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        
        trainer = AdvancedTrainer(
            self.model,
            config={
                'task_type': 'regression',
                'save_dir': temp_dir
            }
        )
        
        # 模拟一些训练历史
        trainer.history['train_loss'] = [1.0, 0.9, 0.8]
        trainer.history['val_loss'] = [1.1, 1.0, 0.9]
        
        # 保存检查点
        trainer.save_checkpoint('test_checkpoint.pth')
        
        # 检查文件是否存在
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()