系统环境：PyTorch  2.5.1，Python  3.12(ubuntu22.04)，CUDA  12.4
GPU：RTX 4090D(24GB) * 5
CPU：75 vCPU Intel(R) Xeon(R) Platinum 8474CPU @ 2.50GHz
内存：400GB
硬盘系统盘：30 GB
数据盘：430GB SSD

## 在AutoDL开启学术资源加速，也就是针对特定外网网站中的资源下载进行了加速
```bash
# 服务器运行命令开启（针对HuggingFace等境外资源加速）https://www.autodl.com/docs/network_turbo/
source /etc/network_turbo
```

## 使用 huggingface_hub 下载模型：
```bash
pip install huggingface_hub
```

## 由于实际下载时间可能持续4-6个小时，因此最好使用screen开启持久化会话，避免因为关闭会话导致下载中断。
```bash
# 安装screen工具用于持久化会话
sudo apt install screen -y
# 开启持久会话
screen -S kt
# 如果会话断开，可以输入如下命令回到之前的会话：
screen -r kt
```

## 默认情况下，Huggingface会将下载文件保存在/root/.cache文件夹中，在 /root/autodl-tmp 下创建名为 HF_download 文件夹作为huggingface下载文件保存文件夹，这样模型就保存在磁盘而非系统盘
```bash
cd /root/autodl-tmp
mkdir -p HF_download DeepSeek-V3-0324-GGUF DeepSeek-V3-0324
```
##  配置HuggingFace下载目录环境变量，找到root文件夹下的 .bashrc 文件
```bash
echo 'export HF_HOME="/root/autodl-tmp/HF_download"' >> ~/.bashrc
source ~/.bashrc
```

## 方式一：使用python代码下载特定量化模型
```bash
# 使用Python代码下载特定量化模型
# unsloth/DeepSeek-V3-0324-GGUF是最新版本的DeepSeek-V3-0324量化模型
# Q4_K_M是量化格式，平衡了性能和质量的良好选择
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = 'unsloth/DeepSeek-V3-0324-GGUF',
    local_dir = '/root/autodl-tmp/DeepSeek-V3-0324-GGUF',
    allow_patterns = ['*Q4_K_M*'],
)
"
```

## 方式二：使用jupyter下载模型权重：
```bash
# 使环境变量生效
source ~/.bashrc

# 进入下载目录：
cd /root/autodl-tmp

# 启动Jupyter
jupyter lab --allow-root

root@autodl-container-8f1147b36e-a9e9d8a4:~/autodl-tmp# jupyter lab --allow-root
[I 2025-04-27 18:59:20.140 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2025-04-27 18:59:20.144 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2025-04-27 18:59:20.150 ServerApp] jupyterlab | extension was successfully linked.
[I 2025-04-27 18:59:20.393 ServerApp] notebook_shim | extension was successfully linked.
[I 2025-04-27 18:59:20.412 ServerApp] notebook_shim | extension was successfully loaded.
[I 2025-04-27 18:59:20.414 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2025-04-27 18:59:20.416 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2025-04-27 18:59:20.417 LabApp] JupyterLab extension loaded from /root/miniconda3/lib/python3.12/site-packages/jupyterlab
[I 2025-04-27 18:59:20.417 LabApp] JupyterLab application directory is /root/miniconda3/share/jupyter/lab
[I 2025-04-27 18:59:20.417 LabApp] Extension Manager is 'pypi'.
[I 2025-04-27 18:59:20.526 LabApp] Extensions will be fetched using proxy, proxy host and port: ('172.20.0.113', '12798')
[I 2025-04-27 18:59:20.530 ServerApp] jupyterlab | extension was successfully loaded.
[I 2025-04-27 18:59:20.530 ServerApp] The port 8888 is already in use, trying another port.
[I 2025-04-27 18:59:20.531 ServerApp] Serving notebooks from local directory: /root/autodl-tmp/HF_download
[I 2025-04-27 18:59:20.531 ServerApp] Jupyter Server 2.14.2 is running at:
[I 2025-04-27 18:59:20.531 ServerApp] http://localhost:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
[I 2025-04-27 18:59:20.531 ServerApp]     http://127.0.0.1:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
[I 2025-04-27 18:59:20.531 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 2025-04-27 18:59:20.535 ServerApp] No web browser found: Error('could not locate runnable browser').
[C 2025-04-27 18:59:20.535 ServerApp] 
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-21597-open.html
    Or copy and paste one of these URLs:
        http://localhost:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
        http://127.0.0.1:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
[I 2025-04-27 18:59:20.557 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
```

## 开启AutoDL隧道工具可以让本地访问服务器的jupyter，并使用jupyter下载模型权重
## 访问Jupyter Web：http://127.0.0.1:8889/lab?token=d05632341ad9255a31ed844b5d1737604d5d1983b1fa918a
```bash
# 页面创建一个ipynb文件运行以下代码：
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/DeepSeek-V3-0324-GGUF",
    local_dir = "DeepSeek-V3-0324-GGUF",
    allow_patterns = ["*Q4_K_M*"],
)
```

## 下载DeepSeek V3原版模型的配置文件：
```bash
根据KTransformer的要求，还需要下载DeepSeek V3原版模型的除了模型权重文件外的其他配置文件，方便进行灵活的专家加载。因此我们还需要使用modelscope下载DeepSeek V3模型除了模型权重（.safetensor）外的其他全部文件，可以按照如下方式进行下载

# 安装魔搭
pip install modelscope

# 进入下载目录：
cd /root/autodl-tmp

# 下载模型其他文件到文件夹
modelscope download --model deepseek-ai/DeepSeek-V3-0324 --exclude '*.safetensors' --local_dir /root/autodl-tmp/DeepSeek-V3-0324

这里最终我们是下载了 DeepSeek-V3-0324-GGUF-Q4_K_M 模型权重和 DeepSeek-V3-0324 的模型配置文件，并分别保存在两个文件夹中：
DeepSeek-V3-0324-GGUF-Q4_K_M 模型权重地址：/root/autodl-tmp/DeepSeek-V3-0324-GGUF
DeepSeek-V3-0324 的模型配置文件地址：/root/autodl-tmp/DeepSeek-V3-0324
```

## 安装基础依赖：
```bash
# 1. 创建独立Python环境
# 创建Python 3.11环境（KTransformers在3.11下测试更充分）
conda init bash # 初始化conda环境
source ~/.bashrc # 使环境变量生效
# 取消学术加速，避免对conda源造成影响
unset http_proxy && unset https_proxy
conda create -n kt311 python=3.11 -y # 创建Python 3.11环境
conda activate kt311 # 激活环境

# 2. 安装系统依赖
# 基础编译工具
sudo apt-get update
sudo apt-get install -y gcc g++ cmake ninja-build libnuma-dev
# 多并发支持依赖（balance_serve后端需要）
sudo apt install -y libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libgflags-dev zlib1g-dev libfmt-dev patchelf

# 3.升级libstdc++（关键步骤，避免GLIBCXX版本错误）
# 安装软件源工具
sudo apt-get install -y software-properties-common
# 添加Ubuntu Toolchain PPA
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
# 升级libstdc++6
sudo apt-get install -y --only-upgrade libstdc++6
# 验证libstdc++版本，确保包含GLIBCXX_3.4.30及以上版本
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail
# 确保conda环境的libstdc++也是新版本，确保了conda环境中安装最新版本的C++标准库，这对KTransformers的安装和运行非常重要
conda install -c conda-forge libstdcxx-ng -y

#4. 安装Python依赖
# 安装PyTorch和基本依赖
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install packaging ninja cpufeature numpy openai

# 安装Flash-Attention，（提前装可以避免后续某些编译依赖出错，flash attention实现比标准注意力机制更高效）
# 先验证 torch 版本
python -c "import torch; print(torch.__version__)"
(kt311) root@autodl-container-8f1147b36e-a9e9d8a4:~/autodl-tmp# python -c "import torch; print(torch.__version__)"
2.5.1+cu124
# 验证 CUDA 版本
nvcc --version
# 适配环境：CUDA 12.4 Python 3.11 PyTorch 2.5.1

# 先卸载可能已经安装的版本
pip uninstall -y flash-attn

# Flash-Attention 安装地址：https://github.com/Dao-AILab/flash-attention/releases
# 最合适的 flash_attn 地址：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# 这个版本有以下特点：
# cu12: 适用于CUDA 12.x（包括您的CUDA 12.4）
# torch2.5: 专门为PyTorch 2.5构建（完全匹配您的PyTorch 2.5.1）
# cp311: 针对Python 3.11构建（与您的环境完全匹配）
# cxx11abiFALSE: 与PyTorch 2.5.1版本的ABI兼容
# linux_x86_64: 适用于Linux x86_64平台
# 下载适合PyTorch 2.5的预编译wheel
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install ./flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# 验证安装
python -c "import flash_attn; print(flash_attn.__version__)"
```

## ktransformers安装步骤
```bash
# 卸载现有ktransformers
pip uninstall ktransformers
# 删除旧仓库文件
rm -rf /root/autodl-tmp/ktransformers
# 开启学术加速
source /etc/network_turbo
# 克隆KTransformers仓库
# 进入数据盘目录
cd /root/autodl-tmp
# 克隆KTransformers仓库
git clone https://github.com/kvcache-ai/ktransformers.git
# 进入KTransformers目录
cd ktransformers
# 初始化和更新子模块
git submodule update --init --recursive

# 确保conda环境激活
conda activate kt311

# 根据CPU配置选择安装方式，获取详细的GPU计算能力信息
nvidia-smi --query-gpu=name,compute_cap --format=csv
# NVIDIA vGPU都具有8.9的计算能力，这对应于Hopper架构（如H100 GPU）。
(kt311) root@autodl-container-8f1147b36e-a9e9d8a4:~/autodl-tmp# nvidia-smi --query-gpu=name,compute_cap --format=csv
name, compute_cap
NVIDIA vGPU-32GB, 8.9
NVIDIA vGPU-32GB, 8.9
NVIDIA vGPU-32GB, 8.9
NVIDIA vGPU-32GB, 8.9
NVIDIA vGPU-32GB, 8.9

# 设置环境变量 - 针对Hopper架构(8.9)优化
export TORCH_CUDA_ARCH_LIST="8.9;9.0+PTX"

# 查看系统NUMA节点配置，确定CPU拓扑结构
lscpu | grep NUMA
# 确认系统NUMA状态，如果没有需要安装
numactl --hardware
# 安装numactl工具
apt-get install -y numactl
# 输出结果：
# 系统有2个NUMA节点，这通常对应于2个物理CPU插槽
# 第一个CPU插槽控制着逻辑CPU核心0-31和64-95
# 第二个CPU插槽控制着逻辑CPU核心32-63和96-127
NUMA node(s):                    2
NUMA node0 CPU(s):               0-31,64-95
NUMA node1 CPU(s):               32-63,96-127

# NUMA优化对于多插槽服务器至关重要
# NUMA（Non-Uniform Memory Access）是一种内存架构，在多插槽服务器中，每个CPU插槽都有自己的本地内存
# 当一个CPU核心访问另一个CPU插槽的内存时，会产生额外的延迟，这称为NUMA惩罚
# KTransformers的NUMA优化可以确保计算任务尽可能在本地内存上执行，减少跨NUMA节点访问
# NUMA优化配置选择指南：
# 1. 对于双插槽或多插槽服务器：启用NUMA支持（USE_NUMA=1）
#    - 这将使KTransformers根据CPU拓扑结构优化线程分配，使其合理分配计算任务
#    - 对于MoE（混合专家）模型如DeepSeek-V3，NUMA优化可以显著提高推理性能
#    - 特别是在处理多并发请求时，减少跨NUMA节点访问，可以减少内存访问延迟
# 2. 对于单插槽服务器：不需要启用NUMA支持
#    - 单插槽服务器没有NUMA架构，所有CPU核心访问内存的延迟相同
#    - 在这种情况下，启用NUMA支持不会带来性能提升，反而可能增加不必要的开销

# 1. 对于双槽服务器(60+ vCPU)，启用NUMA支持和多并发功能
# 注意：环境变量必须在同一个命令中设置并执行安装，否则环境变量不会在编译时生效
# 错误做法（环境变量只在当前shell有效，不会传递给install.sh内部的编译过程）：
# export USE_NUMA=1 # 启用NUMA支持，优化多核CPU性能
# export USE_BALANCE_SERVE=1 # 启用多并发支持，实现高效请求处理
# export KTRANSFORMERS_FORCE_BUILD=TRUE # 强制重新编译，确保使用最新的代码和配置
# bash install.sh # 执行安装脚本
# 正确做法（在同一命令行中设置环境变量并执行安装脚本）：
USE_NUMA=1 USE_BALANCE_SERVE=1 KTRANSFORMERS_FORCE_BUILD=TRUE bash ./install.sh
# 打印环境变量确认设置成功
echo $USE_NUMA
echo $USE_BALANCE_SERVE

# 安装完成后验证NUMA配置是否生效
# 如果返回True，说明NUMA支持已成功启用
# 如果返回False，可能是因为在容器环境中NUMA支持受限，或者libnuma-dev未正确安装
python -c "import ktransformers; print('NUMA支持：', hasattr(ktransformers, 'numa_available') and ktransformers.numa_available())"

# 2. 如果是单槽CPU，则不用设置USE_NUMA=1
# 正确做法（在同一命令行中设置环境变量并执行安装脚本）：
USE_BALANCE_SERVE=1 KTRANSFORMERS_FORCE_BUILD=TRUE bash ./install.sh

# 安装完成后验证安装是否成功
python -c "import ktransformers; print(ktransformers.__version__)"

# 对于单NUMA节点系统，不需要验证NUMA支持
```

## 验证模型文件完整性
```bash
# 检查GGUF模型文件
ls -lh /root/autodl-tmp/DeepSeek-V3-0324-GGUF/
# 检查模型配置文件
ls -lh /root/autodl-tmp/DeepSeek-V3-0324/
# 验证KTransformers安装
# 命令1: 使用pip包管理器查询KTransformers的安装信息
# 显示包的元数据信息，包括版本号、安装位置、依赖库、作者等
# 此命令不会实际加载模块，只查询pip数据库中的信息
pip show ktransformers
# 命令2: 通过Python实际导入KTransformers并打印其版本号
# 此命令会真正加载KTransformers包及其所有依赖和C++扩展
# 如果包含本地编译的扩展模块有任何问题，会在此步骤中显示错误
# 这是更全面的测试，确保包不仅已安装，而且可以正常使用
# 特别适合像KTransformers这样包含C++/CUDA扩展的复杂包
python -c "import ktransformers; print(ktransformers.__version__)"
```

## 启动服务
- 单用户模式测试（先确认模型能正常加载）
```bash
# 确保conda环境激活
conda activate kt311
# 进入KTransformers目录
cd /root/autodl-tmp/ktransformers
# 启动服务
python -m ktransformers.local_chat \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF/Q4_K_M \
  --cpu_infer 64 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
  --max_new_tokens 1024

  # 如果是R1模型，则需要添加参数，用于打印思考过程
  --force_think true
  # 如果是单槽CPU，则不用添加参数
  --cpu_infer 55

  - `--model_path`: safetensor配置路径（仅需要配置，不需要模型safetensors）。
  - `--max_new_tokens`: 每个请求生成的最大令牌数。
  - `--chunk_size`: 引擎在单次运行中处理的最大令牌数。对应32768个令牌，占用的空间将在请求完成后释放。
  - `--max_batch_size`: 引擎在单次运行中处理的最大请求数（预填充+解码）。（仅被`balance_serve`支持）
  - `--cache_lens`: 调度器分配的kvcache总长度。所有请求共享kvcache空间。
  - `--backend_type`: `balance_serve`是v0.2.4版本中引入的多并发后端引擎。原始的单并发引擎是`ktransformers`。
    请注意，自`ver 0.2.4`起，`${model_path}`目录名称的最后一段**必须**是包含模型配置文件的本地目录。目前不支持Hugging Face链接（例如deepseek-ai/DeepSeek-R1）。
  - `--force_think`: 强制响应`DeepSeek R1`的推理标签。

  # max_batch_size、cache_lens和max_new_tokens之间的关系应满足：cache_lens > max_batch_size * max_new_tokens，否则并发度将降低。
```

- 自定义配置启动服务
```bash
# 确保conda环境激活
conda activate kt311

# 进入KTransformers目录
cd /root/autodl-tmp/ktransformers

# 使用5 GPU配置启动多用户服务
python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF/Q4_K_M \
  --cpu_infer 64 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu-5.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 1024 \
  --max_batch_size 4 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --use_cuda_graph False \
  --web True

  --model_path：指向您在/root/autodl-tmp/DeepSeek-V3-0324的模型配置文件路径
  --gguf_path：指向您在/root/autodl-tmp/DeepSeek-V3-0324-GGUF/Q4_K_M的GGUF权重文件
  --cpu_infer 55：设为55，给您的60个vCPU留一些余量
  --optimize_config_path：使用DeepSeek-V3-Chat-multi-gpu-5.yaml多GPU配置文件
  --port：服务端口，默认10002
  --chunk_size：预填充块大小，根据内存和上下文需求调整
  --max_new_tokens：每个请求生成的最大令牌数
  --max_batch_size：引擎在单次运行中处理的最大请求数（预填充+解码）
  --cache_lens: KV缓存总长度，根据内存和上下文需求调整
  --backend_type balance_serve：启用多并发支持，实现高效请求处理
  --use_cuda_graph False：由于配置将部分专家放在GPU上，需要禁用CUDA图优化
  --web True：启用Web界面，可通过浏览器访问
```

- 多用户服务模式（balance_serve后端）
```bash
# 确保conda环境激活
conda activate kt311
# 进入KTransformers目录
cd /root/autodl-tmp/ktransformers

# 针对NUMA架构的优化启动配置
# 对于多插槽服务器，可以使用numactl工具进一步优化服务启动
# 以下是针对双插槽服务器的优化启动命令

# 方式1：使用numactl绑定到特定NUMA节点（推荐用于双插槽服务器）
# 这种方式可以确保所有计算都在同一个NUMA节点上进行，减少跨节点内存访问
# 在某些容器化环境（如 AutoDL）中，容器可能没有被赋予修改 NUMA 内存策略的权限，所以--membind=0可能用不了，需要去掉
numactl --cpunodebind=0 --membind=0 python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF/Q4_K_M \
  --cpu_infer 64 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 1024 \
  --max_batch_size 4 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --web True

# 方式2：不使用numactl，依赖KTransformers内部NUMA优化（适用于已启用USE_NUMA=1编译的情况）
# 这种方式依赖KTransformers在编译时启用的NUMA支持
# 注意：如果在安装时没有正确启用NUMA支持，这种方式不会生效
# 可以通过以下命令验证NUMA支持是否已启用：
# python -c "import ktransformers; print('NUMA支持：', hasattr(ktransformers, 'numa_available') and ktransformers.numa_available())"
# 如果返回False，建议使用方式1（numactl）或重新安装KTransformers
python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF/Q4_K_M \
  --cpu_infer 64 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 1024 \
  --max_batch_size 4 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --web True

# 注意：对于有多个NUMA节点的服务器，可以尝试不同的NUMA绑定策略
# 例如，如果模型较大，可以考虑跨NUMA节点分配，但这可能会增加内存访问延迟
# numactl --interleave=all python ktransformers/server/main.py ...
```

## 长上下文支持配置（可选）
```bash
# 确认系统DRAM足够（450GB足以支持至少128K上下文）
# 在~/.ktransformers目录下会自动创建config.yaml，可以调整以下参数

# 修改config.yaml中的关键参数
cat > ~/.ktransformers/config.yaml << EOF
chunk_size: 4096         # 预填充块大小
max_seq_len: 128000      # KVCache长度，根据DRAM大小调整
block_size: 128          # KVCache块大小
local_windows_len: 4096  # 存储在GPU上的KVCache长度
second_select_num: 96    # 预选后每次选择的KVCache块数量
threads_num: 58          # CPU线程数，设为略低于CPU核心数
anchor_type: DYNAMIC     # KVCache块代表性token选择方法
kv_type: FP16            # KV缓存类型
dense_layer_num: 0       # 不需要填充或选择KVCache的前几层
anchor_num: 1            # KVCache块内代表性token的数量
preselect_block: True    # 启用预选
head_select_mode: SHARED # 所有kv_heads联合选择
preselect_block_count: 96 # 预选的块数量
layer_step: 1            # 每隔几层选择一次
token_step: 1            # 每隔几个token选择一次
absorb_for_prefill: True  # 启用矩阵吸收减少内存使用
EOF

# 启动服务时添加mode参数
python -m ktransformers.local_chat \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF \
  --cpu_infer 55 \
  --max_new_tokens 1000 \
  --mode="long_context"
```

## 多GPU优化配置（可选）
```bash
# 创建自定义多GPU配置文件
cat > ./ktransformers/optimize/optimize_rules/DeepSeek-V3-0324-Multi-GPU.yaml << EOF
# 模型传输映射配置
- match:
    name: "^model$"
  replace:
    class: "ktransformers.operators.models.KDeepseekV2Model"
    kwargs:
      transfer_map: 
        30: "cuda:1"  # 第30层后切换到cuda:1
        40: "cuda:2"  # 第40层后切换到cuda:2
        50: "cuda:3"  # 第50层后切换到cuda:3
        60: "cuda:4"  # 第60层后切换到cuda:4

# 0-29层专家分配到cuda:0
- match:
    name: "^model\\.layers\\.(0|[1-9]|1[0-9]|2[0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda:0"
  recursive: False

# 30-39层专家分配到cuda:1
- match:
    name: "^model\\.layers\\.(3[0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda:1"
  recursive: False

# 40-49层专家分配到cuda:2
- match:
    name: "^model\\.layers\\.(4[0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda:2"
  recursive: False

# 50-59层专家分配到cuda:3
- match:
    name: "^model\\.layers\\.(5[0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda:3"
  recursive: False

# 60+层专家分配到cuda:4
- match:
    name: "^model\\.layers\\.(6[0-9]|7[0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda:4"
  recursive: False

# 其他注意力层和线性层配置（省略部分配置）
EOF

# 使用多GPU配置运行
python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF \
  --cpu_infer 55 \
  --optimize_config_path ./ktransformers/optimize/optimize_rules/DeepSeek-V3-0324-Multi-GPU.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 2048 \
  --max_batch_size 8 \  # 由于使用多GPU，可以增大批处理大小
  --cache_lens 65536 \  # 增大缓存长度
  --backend_type balance_serve \
  --web True
```

## FP8内核配置（可选，需要支持FP8的GPU）
```bash
# 首先安装flashinfer
pip install git+https://github.com/flashinfer-ai/flashinfer.git

# 使用FP8优化配置启动
python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF \
  --cpu_infer 55 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-fp8-linear-ggml-experts.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 2048 \
  --max_batch_size 6 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --web True
```

## 访问KTransformers Web UI
```bash
# 1. 内存监控
# 监控系统内存使用情况
watch -n 2 free -h
# 如果内存不足，可以调整以下参数：
# --chunk_size 128  # 降低单次处理的标记数
# --max_batch_size 4  # 降低批处理大小
# --absorb_for_prefill True  # 启用矩阵吸收减少内存使用

# 2. GPU内存监控
watch -n 2 nvidia-smi
# 5个32GB vGPU足够运行DeepSeek-V3，如果GPU内存不足，考虑：
# 1. 使用更低精度量化模型(Q3_K_M或Q2_K)
# 2. 减小batch_size和cache_lens参数

# 3.CPU使用监控
# 监控CPU使用情况
htop
# 优化参数：
# --cpu_infer: 可在50-58之间调整，找到最佳性能点
# 如果观察到NUMA节点间不平衡，可以尝试numactl绑定

# 4.多并发优化
# --max_batch_size: 对于5个32GB GPU，可以设置为6-8
# --chunk_size: 256是平衡速度和内存的好选择
# --cache_lens: 确保 cache_lens > max_batch_size * max_new_tokens
# 确保USE_BALANCE_SERVE=1已启用，以支持高效多并发

# 5. libstdc++问题排查：
# 如果出现version 'GLIBCXX_X.X.XX' not found错误，需重新执行libstdc++升级步骤
# 确保conda环境和系统环境都有最新版本
```

## 访问Web UI
```bash
# 通过浏览器访问Web UI
http://服务器IP:10002/web/index.html#/chat

# 或者使用API访问服务
curl -X POST http://服务器IP:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，介绍一下DeepSeek-V3-0324模型的特点"}
    ],
    "model": "DeepSeek-V3-0324",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```

## 完整一键部署脚本
```bash
#!/bin/bash
# DeepSeek-V3-0324一键部署脚本

# 1. 环境准备
echo "===== 准备环境 ====="
# 开启学术加速
source /etc/network_turbo
# 确认系统环境
lscpu  # 检查CPU信息
free -h  # 检查内存
nvidia-smi  # 检查GPU
# 配置conda
conda init bash
source ~/.bashrc

# 2. 创建下载目录
echo "===== 准备下载目录 ====="
cd /root/autodl-tmp
mkdir -p HF_download DeepSeek-V3-0324-GGUF DeepSeek-V3-0324
echo 'export HF_HOME="/root/autodl-tmp/HF_download"' >> ~/.bashrc
source ~/.bashrc

# 3. 创建并激活Python环境
echo "===== 创建Python环境 ====="
unset http_proxy && unset https_proxy
conda create -n kt311 python=3.11 -y
conda activate kt311

# 4. 安装依赖
echo "===== 安装依赖 ====="
# 系统依赖
sudo apt-get update
sudo apt-get install -y gcc g++ cmake ninja-build libnuma-dev screen
sudo apt-get install -y libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libgflags-dev zlib1g-dev libfmt-dev patchelf

# libstdc++升级
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y --only-upgrade libstdc++6
conda install -c conda-forge libstdcxx-ng -y

# Python依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install packaging ninja cpufeature numpy openai
pip install flash-attn==2.5.5 或 pip install flash-attn
pip install huggingface_hub modelscope

# 5. 下载模型文件
echo "===== 下载模型文件 ====="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = 'unsloth/DeepSeek-V3-0324-GGUF', 
    local_dir = '/root/autodl-tmp/DeepSeek-V3-0324-GGUF',
    allow_patterns = ['*Q4_K_M*'],
)
"

# 使用modelscope下载模型配置
modelscope download --model deepseek-ai/DeepSeek-V3-0324 --exclude '*.safetensors' --local_dir /root/autodl-tmp/DeepSeek-V3-0324

# 6. 安装KTransformers
echo "===== 安装KTransformers ====="
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

# 根据NUMA配置安装
export USE_NUMA=1
export USE_BALANCE_SERVE=1
bash install.sh

# 7. 启动服务
echo "===== 启动服务 ====="
python ktransformers/server/main.py \
  --model_path /root/autodl-tmp/DeepSeek-V3-0324 \
  --gguf_path /root/autodl-tmp/DeepSeek-V3-0324-GGUF \
  --cpu_infer 55 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --port 10002 \
  --chunk_size 256 \
  --max_new_tokens 2048 \
  --max_batch_size 6 \
  --cache_lens 32768 \
  --backend_type balance_serve \
  --web True

echo "===== 部署完成 ====="
echo "请访问: http://服务器IP:10002/web/index.html#/chat"
```