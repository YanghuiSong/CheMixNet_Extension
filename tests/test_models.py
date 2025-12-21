"""模型测试"""
import unittest
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_models import MLPBaseline, CNNBaseline, RNNBaseline
from models.chemixnet import CheMixNetCNN, CheMixNetRNN, CheMixNetCNNRNN
from models.multimodal import EnhancedCheMixNet

class TestBaseModels(unittest.TestCase):
    """测试基线模型"""
    
    def test_mlp_baseline(self):
        """测试MLP基线模型"""
        model = MLPBaseline(input_dim=167, output_dim=1)
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, 167)
        output = model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, 1))
        
        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # 检查所有参数都要求梯度
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_cnn_baseline(self):
        """测试CNN基线模型"""
        vocab_size = 50
        max_len = 100
        
        model = CNNBaseline(vocab_size=vocab_size, max_len=max_len, output_dim=1)
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, max_len, vocab_size)  # one-hot编码
        output = model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_rnn_baseline(self):
        """测试RNN基线模型"""
        vocab_size = 50
        max_len = 100
        
        model = RNNBaseline(vocab_size=vocab_size, max_len=max_len, output_dim=1)
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, max_len, vocab_size)  # one-hot编码
        output = model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, 1))
        
        # 测试注意力权重获取
        attention_weights = model.get_attention_weights(x)
        self.assertEqual(attention_weights.shape, (batch_size, max_len))

class TestCheMixNetModels(unittest.TestCase):
    """测试CheMixNet模型"""
    
    def setUp(self):
        self.batch_size = 4
        self.vocab_size = 50
        self.max_len = 100
        self.fp_dim = 167
    
    def test_chemixnet_cnn(self):
        """测试CheMixNet CNN架构"""
        model = CheMixNetCNN(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            fp_dim=self.fp_dim,
            output_dim=1
        )
        
        # 测试前向传播
        smiles_input = torch.randn(self.batch_size, self.max_len, self.vocab_size)
        fp_input = torch.randn(self.batch_size, self.fp_dim)
        
        output = model(smiles_input, fp_input)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_chemixnet_rnn(self):
        """测试CheMixNet RNN架构"""
        model = CheMixNetRNN(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            fp_dim=self.fp_dim,
            output_dim=1
        )
        
        # 测试前向传播
        smiles_input = torch.randn(self.batch_size, self.max_len, self.vocab_size)
        fp_input = torch.randn(self.batch_size, self.fp_dim)
        
        output = model(smiles_input, fp_input)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_chemixnet_cnn_rnn(self):
        """测试CheMixNet CNN-RNN架构"""
        model = CheMixNetCNNRNN(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            fp_dim=self.fp_dim,
            output_dim=1
        )
        
        # 测试前向传播
        smiles_input = torch.randn(self.batch_size, self.max_len, self.vocab_size)
        fp_input = torch.randn(self.batch_size, self.fp_dim)
        
        output = model(smiles_input, fp_input)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestEnhancedMultimodalModel(unittest.TestCase):
    """测试增强版多模态模型"""
    
    def setUp(self):
        self.batch_size = 4
        self.vocab_size = 50
        self.max_len = 100
        self.fp_dim = 167
    
    def test_enhanced_chemixnet(self):
        """测试增强版CheMixNet"""
        model = EnhancedCheMixNet(
            smiles_vocab_size=self.vocab_size,
            smiles_max_len=self.max_len,
            maccs_dim=self.fp_dim,
            output_dim=1,
            use_attention=True
        )
        
        # 测试前向传播（不带图数据）
        smiles_input = torch.randn(self.batch_size, self.max_len, self.vocab_size)
        fp_input = torch.randn(self.batch_size, self.fp_dim)
        
        output = model(smiles_input, fp_input, graph_data=None)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # 测试注意力权重获取
        attention_weights = model.get_attention_weights(smiles_input, fp_input)
        
        # 检查注意力权重
        self.assertIn('smiles_weight', attention_weights)
        self.assertIn('fp_weight', attention_weights)
        self.assertIn('graph_weight', attention_weights)
        
        self.assertEqual(attention_weights['smiles_weight'].shape, (self.batch_size,))
        self.assertEqual(attention_weights['fp_weight'].shape, (self.batch_size,))
        self.assertEqual(attention_weights['graph_weight'].shape, (self.batch_size,))
    
    def test_model_without_attention(self):
        """测试不带注意力的模型"""
        model = EnhancedCheMixNet(
            smiles_vocab_size=self.vocab_size,
            smiles_max_len=self.max_len,
            maccs_dim=self.fp_dim,
            output_dim=1,
            use_attention=False
        )
        
        # 测试前向传播
        smiles_input = torch.randn(self.batch_size, self.max_len, self.vocab_size)
        fp_input = torch.randn(self.batch_size, self.fp_dim)
        
        output = model(smiles_input, fp_input, graph_data=None)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestModelDeviceCompatibility(unittest.TestCase):
    """测试模型设备兼容性"""
    
    def test_model_to_device(self):
        """测试模型移动到不同设备"""
        model = MLPBaseline(input_dim=167, output_dim=1)
        
        # 测试CPU
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(2, 167, device='cpu')
        output_cpu = model_cpu(x_cpu)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # 测试GPU（如果可用）
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            x_gpu = torch.randn(2, 167, device='cuda')
            output_gpu = model_gpu(x_gpu)
            self.assertEqual(output_gpu.device.type, 'cuda')
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        import tempfile
        import os
        
        # 创建原始模型
        original_model = CheMixNetCNN(
            vocab_size=50,
            max_len=100,
            fp_dim=167,
            output_dim=1
        )
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(original_model.state_dict(), tmp.name)
            
            # 加载到新模型
            new_model = CheMixNetCNN(
                vocab_size=50,
                max_len=100,
                fp_dim=167,
                output_dim=1
            )
            new_model.load_state_dict(torch.load(tmp.name))
            
            # 测试两个模型输出相同
            smiles_input = torch.randn(2, 100, 50)
            fp_input = torch.randn(2, 167)
            
            original_output = original_model(smiles_input, fp_input)
            new_output = new_model(smiles_input, fp_input)
            
            # 检查输出是否相同（允许小的数值差异）
            self.assertTrue(torch.allclose(original_output, new_output, rtol=1e-6))
            
            # 清理临时文件
            os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()