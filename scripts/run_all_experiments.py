"""运行所有实验的脚本"""
import subprocess
import sys
from pathlib import Path
import yaml
import pandas as pd

def run_experiment_script(script_name):
    """运行实验脚本"""
    cmd = [
        sys.executable, script_name
    ]
    
    print(f"\n{'='*60}")
    print(f"运行脚本: {script_name}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 保存日志
    log_dir = Path('results/logs')
    log_dir.mkdir(ex_parent=True, exist_ok=True)
    
    log_file = log_dir / f"{script_name.replace('/', '_').replace('.py', '')}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    return result.returncode

def run_main_experiment(dataset, model_type, mode='train'):
    """运行单个实验（使用main.py）"""
    cmd = [
        sys.executable, 'main.py',
        '--dataset', dataset,
        '--model_type', model_type,
        '--mode', mode,
        '--config', 'config.yaml'
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: 数据集={dataset}, 模型={model_type}, 模式={mode}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 保存日志
    log_dir = Path('results/logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{dataset}_{model_type}_{mode}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    return result.returncode

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 实验配置
    datasets = ['bace', 'BBBP', 'esol', 'HIV', 'lipophilicity']
    results = []
    
    # 运行专门的实验脚本
    if config['experiments']['run_baselines']:
        retcode = run_experiment_script('experiments/run_baselines.py')
        results.append({
            'experiment': 'baselines',
            'success': retcode == 0
        })
    
    if config['experiments']['run_chemixnet']:
        retcode = run_experiment_script('experiments/run_chemixnet.py')
        results.append({
            'experiment': 'chemixnet',
            'success': retcode == 0
        })
    
    if config['experiments']['run_enhanced']:
        retcode = run_experiment_script('experiments/run_multimodal.py')
        results.append({
            'experiment': 'multimodal',
            'success': retcode == 0
        })
    
    if config['experiments']['run_ablation']:
        retcode = run_experiment_script('experiments/run_ablation.py')
        results.append({
            'experiment': 'ablation',
            'success': retcode == 0
        })
    
    # 运行使用main.py的实验
    main_experiments = []
    
    # 基线模型
    if config['experiments']['run_baselines']:
        main_experiments.extend([
            (dataset, 'mlp', 'train') for dataset in datasets
        ])
        main_experiments.extend([
            (dataset, 'cnn', 'train') for dataset in datasets
        ])
        main_experiments.extend([
            (dataset, 'rnn', 'train') for dataset in datasets
        ])
    
    # CheMixNet
    if config['experiments']['run_chemixnet']:
        main_experiments.extend([
            (dataset, 'chemixnet', 'train') for dataset in datasets
        ])
    
    # 增强版
    if config['experiments']['run_enhanced']:
        main_experiments.extend([
            (dataset, 'enhanced', 'train') for dataset in datasets
        ])
    
    # 运行main.py实验
    for dataset, model_type, mode in main_experiments:
        retcode = run_main_experiment(dataset, model_type, mode)
        results.append({
            'dataset': dataset,
            'model_type': model_type,
            'mode': mode,
            'success': retcode == 0
        })
    
    # 汇总结果
    print("\n\n实验完成汇总:")
    print("-" * 60)
    success_count = sum(1 for r in results if r['success'])
    print(f"总实验数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")
    
    # 保存汇总
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/experiment_summary.csv', index=False, encoding='utf-8')
    print("\n结果已保存到: results/experiment_summary.csv")

if __name__ == '__main__':
    main()