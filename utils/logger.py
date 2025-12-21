"""日志记录工具"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog

def setup_logger(name='CheMixNet', log_dir='./logs', level=logging.INFO):
    """设置日志记录器"""
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'{name}_{timestamp}.log'
    
    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 控制台处理器（带颜色）
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name='CheMixNet'):
    """获取日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger

class LoggerWrapper:
    """日志记录器包装器，用于记录训练过程"""
    
    def __init__(self, logger, log_file=None):
        self.logger = logger
        self.log_file = log_file
        self.metrics_history = []
        
    def log_metrics(self, epoch, metrics, stage='train'):
        """记录指标"""
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch:03d} - {stage} - {metrics_str}')
        
        # 保存到历史
        self.metrics_history.append({
            'epoch': epoch,
            'stage': stage,
            **metrics
        })
    
    def log_model_info(self, model):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型总参数量: {total_params:,}")
        self.logger.info(f"可训练参数量: {trainable_params:,}")
        
        # 记录模型结构
        self.logger.debug("模型结构:")
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            self.logger.debug(f"  {name}: {module.__class__.__name__} ({num_params:,} params)")
    
    def save_history(self, filepath):
        """保存历史记录"""
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        self.logger.info(f"指标历史保存到: {filepath}")
    
    def log_experiment_start(self, experiment_name, config):
        """记录实验开始"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始实验: {experiment_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"配置: {config}")
    
    def log_experiment_end(self, experiment_name, results):
        """记录实验结束"""
        self.logger.info("=" * 60)
        self.logger.info(f"实验结束: {experiment_name}")
        self.logger.info(f"结果: {results}")
        self.logger.info("=" * 60)