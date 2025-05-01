# ktransformers的ROCm支持（测试版）

## 介绍

### 概述
为了扩展NVIDIA之外的GPU架构支持，我们很高兴在ktransformers中引入**通过ROCm支持AMD GPU**（测试版）。此实现已使用EPYC 9274F处理器和AMD Radeon 7900xtx GPU进行测试和开发。

## 安装指南

### 1. 安装ROCm驱动
首先为您的AMD GPU安装ROCm驱动：
- [Radeon GPU的官方ROCm安装指南](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html)

### 2. 设置Conda环境
我们建议使用Miniconda3/Anaconda3进行环境管理：

```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create --name ktransformers python=3.11
conda activate ktransformers

# 安装所需库
conda install -c conda-forge libstdcxx-ng

# 验证GLIBCXX版本（应包含3.4.32）
strings ~/anaconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
```

> **注意：** 如果您的安装目录与`~/anaconda3`不同，请调整Anaconda路径

### 3. 安装支持ROCm的PyTorch
安装支持ROCm 6.2.4的PyTorch：

```bash
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2.4
pip3 install packaging ninja cpufeature numpy
```

> **提示：** 对于其他ROCm版本，请访问[PyTorch以前的版本](https://pytorch.org/get-started/previous-versions/)

### 4. 构建ktransformers

```bash
# 克隆仓库
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init

# 可选：编译Web界面
# 参见：api/server/website.md

# 安装依赖
bash install.sh
```

## 运行DeepSeek-R1模型

### 24GB VRAM GPU的配置
使用我们针对有限VRAM优化的配置：

```bash
python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-R1 \
  --gguf_path <gguf文件路径> \
  --optimize_config_path ktransformers/optimize/optimize_rules/rocm/DeepSeek-V3-Chat.yaml \
  --cpu_infer <cpu核心数 + 1>
```

> **测试版说明：** 当前的Q8线性实现（Marlin替代方案）性能不佳。预计在未来版本中会进行优化。

### 40GB+VRAM GPU的配置
对于高VRAM GPU，可获得更好性能：

1. 修改`DeepSeek-V3-Chat.yaml`：
   ```yaml
   # 替换所有以下实例：
   KLinearMarlin → KLinearTorch
   ```

2. 执行：
   ```bash
   python ktransformers/local_chat.py \
     --model_path deepseek-ai/DeepSeek-R1 \
     --gguf_path <gguf文件路径> \
     --optimize_config_path <修改后的yaml路径> \
     --cpu_infer <cpu核心数 + 1>
   ```
> **提示：** 如果您有2个24GB AMD GPU，您也可以进行相同的修改并改为运行`ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml`。

## 已知限制
- ROCm平台不支持Marlin操作
- 当前Q8线性实现显示性能降低（测试版限制） 