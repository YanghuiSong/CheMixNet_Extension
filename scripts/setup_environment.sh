#!/bin/bash

# CheMixNet项目环境设置脚本

echo "设置CheMixNet项目环境..."
echo "================================"

# 创建conda环境
ENV_NAME="chemixnet"
PYTHON_VERSION="3.8"

echo "1. 创建conda环境: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo "2. 激活conda环境"
conda activate $ENV_NAME

# 安装PyTorch (根据CUDA版本选择)
echo "3. 安装PyTorch和相关库"

# 检测CUDA版本
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
if [ -z "$CUDA_VERSION" ]; then
    echo "  未检测到CUDA，安装CPU版本PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "  检测到CUDA $CUDA_VERSION，安装GPU版本PyTorch"
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == "11.7" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    elif [[ "$CUDA_VERSION" == "10.2" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu102
    else
        echo "  不支持的CUDA版本，安装CPU版本"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# 安装PyTorch Geometric
echo "4. 安装PyTorch Geometric"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# 安装基本依赖
echo "5. 安装基本依赖"
pip install numpy pandas scikit-learn matplotlib seaborn plotly jupyter tqdm pyyaml joblib pillow

# 安装化学信息学库
echo "6. 安装化学信息学库"
conda install -c conda-forge rdkit -y
pip install deepchem

# 安装项目依赖
echo "7. 安装项目依赖"
pip install -r requirements.txt

# 创建必要的目录
echo "8. 创建项目目录"
mkdir -p data
mkdir -p results/figures
mkdir -p results/checkpoints
mkdir -p results/logs
mkdir -p results/tables

# 下载数据（如果需要）
echo "9. 下载数据集"
python scripts/download_data.py

echo "================================"
echo "环境设置完成!"
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 运行实验: python main.py --dataset bace --model_type enhanced --mode train"
echo "3. 运行所有实验: python experiments/run_all_experiments.py"
echo "4. 打开Jupyter: jupyter notebook"